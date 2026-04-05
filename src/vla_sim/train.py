"""
Train a VLA (Vision-Language-Action) model for driving.

Three training modes:
    1. IL (Imitation Learning): Learn from expert demonstrations
    2. RL (Reinforcement Learning): Learn from reward via PPO
    3. IL+RL: Pretrain with IL, fine-tune with RL

Expert demonstrations are collected from the trained PPO agent
in MetaDrive, paired with language navigation commands.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque

from src.vla_sim.model import VLADrivingModel
from src.vla_sim.env import LanguageDrivingEnv, COMMAND_LIST


def make_env(map_type="SSS", traffic=0.15, num_scenarios=50):
    """Create language-conditioned MetaDrive environment."""
    from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv

    base_env = TopDownSingleFrameMetaDriveEnv(config={
        "num_scenarios": num_scenarios,
        "map": map_type,
        "traffic_density": traffic,
    })
    return LanguageDrivingEnv(base_env)


def collect_expert_data(env, expert_model, num_episodes=50, device="cuda"):
    """Collect demonstrations from trained PPO expert."""
    from stable_baselines3 import PPO

    print(f"  Collecting {num_episodes} expert episodes...")
    dataset = {"images": [], "commands": [], "actions": []}

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_steps = 0

        while not done and ep_steps < 300:
            # Expert predicts action from image only (ignores language)
            img = obs["image"].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            # Use expert model
            with torch.no_grad():
                action, _ = expert_model.predict(
                    obs["image"][np.newaxis], deterministic=True
                )
            action = action[0]

            dataset["images"].append(obs["image"])
            dataset["commands"].append(obs["command_id"])
            dataset["actions"].append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1

        if (ep + 1) % 10 == 0:
            print(f"    Episode {ep+1}/{num_episodes}, total samples: {len(dataset['images'])}")

    # Convert to numpy
    dataset["images"] = np.array(dataset["images"])
    dataset["commands"] = np.array(dataset["commands"])
    dataset["actions"] = np.array(dataset["actions"])

    print(f"  Collected {len(dataset['images'])} samples")
    return dataset


def train_il(model, dataset, epochs=30, batch_size=64, lr=3e-4, device="cuda"):
    """Train VLA with imitation learning on expert data."""
    print(f"\n=== Imitation Learning ({epochs} epochs) ===")

    images = torch.from_numpy(dataset["images"]).float().to(device) / 255.0
    images = images.permute(0, 3, 1, 2)  # NHWC → NCHW
    commands = torch.from_numpy(dataset["commands"]).long().to(device)
    actions = torch.from_numpy(dataset["actions"]).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_samples = len(images)

    for epoch in range(epochs):
        indices = torch.randperm(n_samples, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            pred_actions, _ = model(images[idx], commands[idx])
            loss = F.mse_loss(pred_actions, actions[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.5f}")


def train_rl(model, env, total_steps=50000, lr=3e-4, gamma=0.99,
             gae_lambda=0.95, clip_range=0.2, batch_size=64,
             n_steps=256, device="cuda"):
    """Train VLA with PPO reinforcement learning."""
    print(f"\n=== RL Training (PPO, {total_steps} steps) ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reward_history = deque(maxlen=20)
    step = 0

    while step < total_steps:
        # Collect rollout
        obs_images, obs_cmds, actions_list = [], [], []
        rewards_list, values_list, log_probs_list, dones_list = [], [], [], []

        obs, info = env.reset()
        for t in range(n_steps):
            img = torch.from_numpy(obs["image"]).float().to(device) / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            cmd = torch.tensor([obs["command_id"]], device=device)

            with torch.no_grad():
                action, value, log_prob = model.get_action(img, cmd)

            action_np = action[0].cpu().numpy()
            obs_images.append(obs["image"])
            obs_cmds.append(obs["command_id"])
            actions_list.append(action_np)
            values_list.append(value.item())
            log_probs_list.append(log_prob.item())

            obs, reward, terminated, truncated, info = env.step(action_np)
            rewards_list.append(reward)
            dones_list.append(terminated or truncated)

            if terminated or truncated:
                reward_history.append(sum(rewards_list[-int(info.get("episode_length", len(rewards_list))):]))
                obs, info = env.reset()

        step += n_steps

        # Compute advantages (GAE)
        advantages = np.zeros(n_steps)
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
            else:
                next_value = values_list[t + 1]
            delta = rewards_list[t] + gamma * next_value * (1 - dones_list[t]) - values_list[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones_list[t]) * last_gae
        returns = advantages + np.array(values_list)

        # PPO update
        imgs = torch.from_numpy(np.array(obs_images)).float().to(device) / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        cmds = torch.tensor(obs_cmds, device=device)
        acts = torch.from_numpy(np.array(actions_list)).float().to(device)
        old_log_probs = torch.tensor(log_probs_list, device=device)
        advs = torch.tensor(advantages, dtype=torch.float32, device=device)
        rets = torch.tensor(returns, dtype=torch.float32, device=device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            indices = torch.randperm(n_steps, device=device)
            for i in range(0, n_steps, batch_size):
                idx = indices[i:i+batch_size]
                values, log_probs, entropy = model.evaluate_action(
                    imgs[idx], cmds[idx], acts[idx]
                )

                ratio = (log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advs[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), rets[idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if step % 5000 == 0 and reward_history:
            print(f"  Step {step:>6d}  mean_reward={np.mean(reward_history):+.1f}  "
                  f"policy_loss={policy_loss.item():.4f}")


def evaluate(model, env, num_episodes=10, device="cuda"):
    """Evaluate VLA model."""
    rewards = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            img = torch.from_numpy(obs["image"]).float().to(device) / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            cmd = torch.tensor([obs["command_id"]], device=device)

            with torch.no_grad():
                action, _ = model(img, cmd)
            action_np = action[0].cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser(description="Train VLA for Driving")
    parser.add_argument("--mode", type=str, default="il+rl",
                        choices=["il", "rl", "il+rl"])
    parser.add_argument("--expert_path", type=str,
                        default="outputs/metadrive/highway/highway_final.zip")
    parser.add_argument("--output", type=str, default="outputs/vla_sim/vla_driving.pt")
    parser.add_argument("--il_epochs", type=int, default=30)
    parser.add_argument("--rl_steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VLA Training for Autonomous Driving")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60)

    # Create environment
    print("\n[1] Creating language-conditioned driving environment...")
    env = make_env(map_type="SSS", traffic=0.15)
    obs, _ = env.reset()
    print(f"  Image obs: {obs['image'].shape}")
    print(f"  Command: '{env.command_text}' (id={obs['command_id']})")
    print(f"  Action space: {env.action_space}")

    # Create VLA model
    print("\n[2] Creating VLA model...")
    model = VLADrivingModel().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.2f}M")

    if "il" in args.mode:
        # Load expert for demonstration collection
        print("\n[3] Loading expert model for demonstrations...")
        from stable_baselines3 import PPO
        expert = PPO.load(args.expert_path, device=device)
        print(f"  Loaded expert from {args.expert_path}")

        # Collect demonstrations
        dataset = collect_expert_data(env, expert, num_episodes=30, device=device)

        # Train with IL
        train_il(model, dataset, epochs=args.il_epochs, device=device)

        # Evaluate after IL
        mean_r, std_r = evaluate(model, env, device=device)
        print(f"  After IL: reward = {mean_r:.1f} +/- {std_r:.1f}")

    if "rl" in args.mode:
        # Fine-tune with RL
        env.reset()
        train_rl(model, env, total_steps=args.rl_steps, device=device)

        # Evaluate after RL
        mean_r, std_r = evaluate(model, env, device=device)
        print(f"  After RL: reward = {mean_r:.1f} +/- {std_r:.1f}")

    # Save
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved VLA model to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
