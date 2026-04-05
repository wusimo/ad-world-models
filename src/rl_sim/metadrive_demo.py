"""
Demo for MetaDrive RL-trained driving agents.

Renders 3D simulation frames for trained vs random agents,
produces side-by-side comparison with top-down BEV and reward curves.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path
from PIL import Image
from stable_baselines3 import PPO

from src.rl_sim.metadrive_train import SCENARIOS


TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def rollout(env, model=None, max_steps=300, obs_as_frame=False):
    """Run one episode, collect frames and rewards.

    If obs_as_frame=True, the observation itself is used as the rendered frame
    (for TopDownSingleFrameMetaDriveEnv where obs is a bird's-eye image).
    """
    obs, info = env.reset()
    frames, rewards = [], []

    if obs_as_frame and obs.ndim == 3:
        frame = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        frames.append(frame)

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if obs_as_frame and step % 3 == 0 and obs.ndim == 3:
            frame = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
            if frame.shape[2] == 1:
                frame = np.repeat(frame, 3, axis=2)
            frames.append(frame)

        if terminated or truncated:
            break

    return frames, rewards, info


def create_comparison(scenario, frames_random, rewards_random,
                     frames_trained, rewards_trained, save_path):
    """Create side-by-side random vs trained comparison."""
    cfg = SCENARIOS[scenario]
    n_show = min(8, len(frames_random), len(frames_trained))

    def select_frames(frames, n):
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    rf = select_frames(frames_random, n_show)
    tf = select_frames(frames_trained, n_show)

    fig = plt.figure(figsize=(3.5 * n_show, 9), facecolor="#1a1a1a")
    gs = fig.add_gridspec(3, n_show, height_ratios=[1, 1, 0.5])

    # Row 1: Random
    for col in range(n_show):
        ax = fig.add_subplot(gs[0, col])
        if col < len(rf) and rf[col] is not None:
            ax.imshow(rf[col])
        ax.axis("off")
        if col == 0:
            ax.set_title("Random Agent", fontsize=11, color="#ff6b6b", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#ff6b6b")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 2: Trained
    for col in range(n_show):
        ax = fig.add_subplot(gs[1, col])
        if col < len(tf) and tf[col] is not None:
            ax.imshow(tf[col])
        ax.axis("off")
        if col == 0:
            ax.set_title("Trained Agent (PPO)", fontsize=11, color="#00ff88", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#00ff88")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 3: Reward curve
    ax_r = fig.add_subplot(gs[2, :])
    ax_r.set_facecolor("#2d2d2d")
    cr = np.cumsum(rewards_random)
    ct = np.cumsum(rewards_trained)
    ax_r.plot(cr, color="#ff6b6b", lw=2, label=f"Random (total={cr[-1]:.1f})")
    ax_r.plot(ct, color="#00ff88", lw=2, label=f"Trained (total={ct[-1]:.1f})")
    ax_r.fill_between(range(len(cr)), cr, alpha=0.15, color="#ff6b6b")
    ax_r.fill_between(range(len(ct)), ct, alpha=0.15, color="#00ff88")
    ax_r.set_xlabel("Step", color="white")
    ax_r.set_ylabel("Cumulative Reward", color="white")
    ax_r.legend(fontsize=9, facecolor="#333", edgecolor="white", labelcolor="white")
    ax_r.tick_params(colors="white")
    for sp in ax_r.spines.values():
        sp.set_color("white")
    ax_r.grid(True, alpha=0.2, color="white")

    fig.suptitle(f"MetaDrive RL: {scenario.upper()} — {cfg['description']}",
                fontsize=14, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"  Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MetaDrive RL Demo")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--model_dir", type=str, default="outputs/metadrive")
    parser.add_argument("--save_dir", type=str, default="outputs/metadrive")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    print("=" * 60)
    print("  MetaDrive RL Demo — 3D Driving Simulation")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print("=" * 60)

    for scenario in scenarios:
        cfg = SCENARIOS[scenario]
        print(f"\n--- {scenario.upper()}: {cfg['description']} ---")

        # Load model
        model_path = Path(args.model_dir) / scenario / "best_model.zip"
        final_path = Path(args.model_dir) / scenario / f"{scenario}_final.zip"

        if model_path.exists():
            model = PPO.load(str(model_path))
            print(f"  Loaded model from {model_path}")
        elif final_path.exists():
            model = PPO.load(str(final_path))
            print(f"  Loaded model from {final_path}")
        else:
            print(f"  No model found. Run: python -m src.rl_sim.metadrive_train --scenario {scenario}")
            continue

        # Create env with top-down rendering
        # Create same env type as training
        from src.rl_sim.metadrive_train import make_env as make_md_env
        use_cnn = cfg.get("policy", "MlpPolicy") == "CnnPolicy"
        env = make_md_env(cfg["env_cls"], cfg["config"], seed=0, use_cnn=use_cnn)

        # Check if observations are images (TopDown env)
        obs_is_image = len(env.observation_space.shape) == 3

        # Random agent
        env.reset(seed=0)
        frames_r, rewards_r, info_r = rollout(env, model=None, max_steps=200, obs_as_frame=obs_is_image)

        # Trained agent
        env.reset(seed=0)
        frames_t, rewards_t, info_t = rollout(env, model=model, max_steps=200, obs_as_frame=obs_is_image)

        print(f"  Random:  reward={sum(rewards_r):.1f}, steps={len(rewards_r)}")
        print(f"  Trained: reward={sum(rewards_t):.1f}, steps={len(rewards_t)}")

        save_dir = Path(args.save_dir) / scenario
        save_dir.mkdir(parents=True, exist_ok=True)

        if frames_r and frames_t:
            create_comparison(
                scenario, frames_r, rewards_r, frames_t, rewards_t,
                str(save_dir / f"{scenario}_comparison.png"),
            )

        env.close()

    print(f"\nAll results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
