"""
Demo for VLA-trained driving agent.

Shows the agent following language commands with visualization of:
    - Bird's-eye view simulation frames
    - Current language command
    - Agent's action (steering + acceleration)
    - Cumulative reward comparison (random vs IL vs IL+RL)
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path

from src.vla_sim.model import VLADrivingModel
from src.vla_sim.env import LanguageDrivingEnv, COMMANDS, COMMAND_LIST


TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def rollout_vla(env, model=None, device="cuda", max_steps=200):
    """Run episode with VLA model, collecting frames + commands + actions."""
    obs, info = env.reset()
    frames, commands, actions_taken, rewards = [], [], [], []

    for step in range(max_steps):
        frames.append(obs["image"].copy())
        commands.append(env.command_text)

        if model is not None:
            img = torch.from_numpy(obs["image"]).float().to(device) / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            cmd = torch.tensor([obs["command_id"]], device=device)
            with torch.no_grad():
                action, _ = model(img, cmd)
            action_np = action[0].cpu().numpy()
        else:
            action_np = env.action_space.sample()

        actions_taken.append(action_np)
        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(reward)

        if terminated or truncated:
            break

    return frames, commands, actions_taken, rewards


def main():
    parser = argparse.ArgumentParser(description="VLA Driving Demo")
    parser.add_argument("--weights", type=str, default="outputs/vla_sim/vla_driving.pt")
    parser.add_argument("--save_dir", type=str, default="outputs/vla_sim")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VLA (Vision-Language-Action) Driving Demo")
    print("=" * 60)

    # Create env
    from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv
    base_env = TopDownSingleFrameMetaDriveEnv(config={
        "num_scenarios": 50, "map": "SSS", "traffic_density": 0.15,
    })
    env = LanguageDrivingEnv(base_env)

    # Load model
    model = VLADrivingModel().to(device)
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded VLA model from {args.weights}")
    else:
        print(f"  No weights found. Run: python -m src.vla_sim.train")
        return

    # Run episodes
    print("\nRunning episodes...")

    # Random agent
    frames_r, cmds_r, acts_r, rewards_r = rollout_vla(env, model=None, device=device)
    print(f"  Random:  reward={sum(rewards_r):.1f}, steps={len(rewards_r)}")

    # VLA agent
    frames_v, cmds_v, acts_v, rewards_v = rollout_vla(env, model=model, device=device)
    print(f"  VLA:     reward={sum(rewards_v):.1f}, steps={len(rewards_v)}")

    # === Visualization ===
    n_show = min(8, len(frames_v))
    indices = np.linspace(0, len(frames_v) - 1, n_show, dtype=int)

    fig = plt.figure(figsize=(4 * n_show, 12), facecolor="#1a1a1a")
    gs = fig.add_gridspec(4, n_show, height_ratios=[1, 0.15, 1, 0.5])

    # Row 1: VLA agent frames with commands
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames_v[idx])
        ax.axis("off")
        if col == 0:
            ax.set_title("VLA Agent", fontsize=12, color="#00ff88", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#00ff88")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 2: Language commands for each frame
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        cmd_text = cmds_v[idx] if idx < len(cmds_v) else ""
        # Shorten for display
        short_cmd = cmd_text.split(".")[0] if cmd_text else ""
        ax.text(0.5, 0.5, f'"{short_cmd}"', transform=ax.transAxes,
               fontsize=7, color="#64b5f6", ha="center", va="center",
               fontstyle="italic", path_effects=TEXT_OUTLINE)
        steer, accel = acts_v[idx] if idx < len(acts_v) else (0, 0)
        ax.text(0.5, 0.0, f"steer={steer:+.2f} accel={accel:+.2f}",
               transform=ax.transAxes, fontsize=6, color="#aaa",
               ha="center", va="bottom", fontfamily="monospace")

    # Row 3: Random agent frames
    rand_indices = np.linspace(0, len(frames_r) - 1, n_show, dtype=int)
    for col, idx in enumerate(rand_indices):
        ax = fig.add_subplot(gs[2, col])
        if idx < len(frames_r):
            ax.imshow(frames_r[idx])
        ax.axis("off")
        if col == 0:
            ax.set_title("Random Agent", fontsize=12, color="#ff6b6b", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#ff6b6b")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 4: Reward curve
    ax_r = fig.add_subplot(gs[3, :])
    ax_r.set_facecolor("#2d2d2d")
    cr = np.cumsum(rewards_r)
    cv = np.cumsum(rewards_v)
    ax_r.plot(cr, color="#ff6b6b", lw=2, label=f"Random (total={cr[-1]:.1f})")
    ax_r.plot(cv, color="#00ff88", lw=2, label=f"VLA (total={cv[-1]:.1f})")
    ax_r.fill_between(range(len(cr)), cr, alpha=0.15, color="#ff6b6b")
    ax_r.fill_between(range(len(cv)), cv, alpha=0.15, color="#00ff88")
    ax_r.set_xlabel("Step", color="white")
    ax_r.set_ylabel("Cumulative Reward", color="white")
    ax_r.legend(fontsize=9, facecolor="#333", edgecolor="white", labelcolor="white")
    ax_r.tick_params(colors="white")
    for sp in ax_r.spines.values():
        sp.set_color("white")
    ax_r.grid(True, alpha=0.2, color="white")

    fig.suptitle("VLA Agent: Vision + Language Command → Driving Action",
                fontsize=15, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(save_dir / "vla_sim_demo.png"), dpi=150,
               bbox_inches="tight", facecolor="#1a1a1a")
    print(f"\nSaved to {save_dir}/vla_sim_demo.png")

    env.close()


if __name__ == "__main__":
    main()
