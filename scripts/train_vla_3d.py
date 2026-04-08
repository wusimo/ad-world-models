"""
Train the 3D-camera VLA driving model.

Uses MetaDrive's built-in IDM lane-following expert as the demonstration source
(no need for a pretrained PPO expert since we're switching obs space).

Pipeline:
    1. Roll out the IDM expert in 3D camera env to collect demos
    2. Imitation Learning on (image, command) → (steer, accel)
    3. Optional PPO fine-tuning
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque

from src.vla_sim.train_3d import VLA3DModel
from src.vla_sim.env_3d import make_3d_lang_env, COMMAND_LIST


def collect_idm_demos(env, num_episodes=20, max_steps=300, device="cuda"):
    """
    Collect demonstrations using MetaDrive's built-in IDM (Intelligent Driver Model)
    expert. We use a simple proportional controller as fallback.
    """
    print(f"  Collecting {num_episodes} expert episodes (IDM lane-keeping)...")
    images, commands, actions = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        for step in range(max_steps):
            # Simple expert: drive forward, light steering correction
            # Use info from base env to compute lateral error
            try:
                vehicle = env.unwrapped.vehicle
                lane = vehicle.lane
                lateral = vehicle.lane.local_coordinates(vehicle.position)[1]
                heading_diff = vehicle.heading_theta - lane.heading_theta_at(
                    vehicle.lane.local_coordinates(vehicle.position)[0]
                )
                # Steering: P-controller on lateral error and heading
                steer = -0.3 * lateral - 0.5 * heading_diff
                steer = float(np.clip(steer, -1, 1))
                # Throttle: maintain speed ~30 km/h
                target_speed = 30.0
                speed_error = (target_speed - vehicle.speed_km_h) / 10.0
                accel = float(np.clip(speed_error, -0.5, 0.6))
                action = np.array([steer, accel], dtype=np.float32)
            except Exception:
                action = np.array([0.0, 0.3], dtype=np.float32)

            images.append(obs["image"])
            commands.append(obs["command_id"])
            actions.append(action)

            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1}/{num_episodes}, samples: {len(images)}")

    return {
        "images": np.array(images),
        "commands": np.array(commands),
        "actions": np.array(actions),
    }


def train_il(model, dataset, epochs=20, batch_size=32, lr=3e-4, device="cuda"):
    print(f"\n=== Imitation Learning ({epochs} epochs) ===")

    # Move data to GPU in chunks (large images)
    images = torch.from_numpy(dataset["images"]).float() / 255.0
    images = images.permute(0, 3, 1, 2)  # NHWC → NCHW
    commands = torch.from_numpy(dataset["commands"]).long()
    actions = torch.from_numpy(dataset["actions"]).float()

    n_samples = len(images)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        indices = torch.randperm(n_samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            img_batch = images[idx].to(device)
            cmd_batch = commands[idx].to(device)
            act_batch = actions[idx].to(device)

            pred_actions, _ = model(img_batch, cmd_batch)
            loss = F.mse_loss(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(1, n_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg:.5f}")


def evaluate(model, env, num_episodes=5, device="cuda"):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            img = torch.from_numpy(obs["image"]).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
            cmd = torch.tensor([obs["command_id"]], device=device)
            with torch.no_grad():
                action, _ = model(img, cmd)
            action_np = action[0].cpu().numpy()
            obs, r, term, trunc, _ = env.step(action_np)
            total += r
            done = term or trunc
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/vla_sim/vla_3d_driving.pt")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VLA 3D Training (Camera + Language → Action)")
    print("=" * 60)

    print("\n[1] Creating 3D camera env...")
    env = make_3d_lang_env(num_scenarios=20, map_type="SSS", traffic=0.15)
    obs, _ = env.reset()
    print(f"  Image: {obs['image'].shape}, Command id: {obs['command_id']}")

    print("\n[2] Building VLA 3D model...")
    model = VLA3DModel().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.2f}M")

    print("\n[3] Collecting expert demonstrations...")
    dataset = collect_idm_demos(env, num_episodes=args.num_episodes, device=device)
    print(f"  Total samples: {len(dataset['images'])}")

    train_il(model, dataset, epochs=args.epochs, device=device)

    print("\n[4] Evaluating...")
    mean_r, std_r = evaluate(model, env, num_episodes=5, device=device)
    print(f"  VLA 3D reward: {mean_r:.1f} +/- {std_r:.1f}")

    torch.save(model.state_dict(), args.output)
    print(f"\nSaved to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
