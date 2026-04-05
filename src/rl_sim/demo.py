"""
Demo script for RL-trained driving agents in simulation.

Visualizes trained agents across multiple driving scenarios:
    - Highway: lane changing at high speed
    - Intersection: navigating cross traffic
    - Roundabout: entering/exiting circular flow
    - Merge: on-ramp highway merging

Produces side-by-side comparison of random vs trained agents
with rendered simulation frames and reward curves.
"""

import argparse
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path
from PIL import Image
from stable_baselines3 import PPO, DQN

from src.rl_sim.train import ENV_CONFIGS

TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def rollout_episode(env, model=None, max_steps=200):
    """Run one episode, return frames, rewards, and info."""
    obs, info = env.reset()
    frames = [env.render()]
    rewards = []
    actions = []

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        rewards.append(reward)
        actions.append(action)

        if terminated or truncated:
            break

    return frames, rewards, actions, info


def visualize_scenario(scenario: str, frames_random, rewards_random,
                      frames_trained, rewards_trained, save_path: str):
    """Create comparison visualization for one scenario."""
    cfg = ENV_CONFIGS[scenario]

    # Select key frames evenly spaced
    n_show = 6
    def select_frames(frames, n):
        if len(frames) <= n:
            return frames, list(range(len(frames)))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices], indices.tolist()

    rand_frames, rand_idx = select_frames(frames_random, n_show)
    trained_frames, trained_idx = select_frames(frames_trained, n_show)

    fig = plt.figure(figsize=(4 * n_show, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(3, n_show, height_ratios=[1, 1, 0.6])

    # Row 1: Random agent frames
    for col in range(n_show):
        ax = fig.add_subplot(gs[0, col])
        if col < len(rand_frames):
            ax.imshow(rand_frames[col])
        ax.axis("off")
        if col == 0:
            ax.set_title("Random Agent", fontsize=12, color="#ff6b6b", fontweight="bold")
        else:
            step = rand_idx[col] if col < len(rand_idx) else 0
            ax.set_title(f"step {step}", fontsize=9, color="gray")
        for sp in ax.spines.values():
            sp.set_color("#ff6b6b")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 2: Trained agent frames
    for col in range(n_show):
        ax = fig.add_subplot(gs[1, col])
        if col < len(trained_frames):
            ax.imshow(trained_frames[col])
        ax.axis("off")
        if col == 0:
            ax.set_title("Trained Agent (RL)", fontsize=12, color="#00ff88", fontweight="bold")
        else:
            step = trained_idx[col] if col < len(trained_idx) else 0
            ax.set_title(f"step {step}", fontsize=9, color="gray")
        for sp in ax.spines.values():
            sp.set_color("#00ff88")
            sp.set_linewidth(2)
            sp.set_visible(True)

    # Row 3: Reward curves
    ax_reward = fig.add_subplot(gs[2, :])
    ax_reward.set_facecolor("#2d2d2d")

    cum_random = np.cumsum(rewards_random)
    cum_trained = np.cumsum(rewards_trained)

    ax_reward.plot(cum_random, color="#ff6b6b", linewidth=2, label=f"Random (total={cum_random[-1]:.1f})")
    ax_reward.plot(cum_trained, color="#00ff88", linewidth=2, label=f"Trained (total={cum_trained[-1]:.1f})")
    ax_reward.fill_between(range(len(cum_random)), cum_random, alpha=0.15, color="#ff6b6b")
    ax_reward.fill_between(range(len(cum_trained)), cum_trained, alpha=0.15, color="#00ff88")
    ax_reward.set_xlabel("Step", fontsize=11, color="white")
    ax_reward.set_ylabel("Cumulative Reward", fontsize=11, color="white")
    ax_reward.legend(fontsize=10, facecolor="#333", edgecolor="white", labelcolor="white")
    ax_reward.tick_params(colors="white")
    for sp in ax_reward.spines.values():
        sp.set_color("white")
    ax_reward.grid(True, alpha=0.2, color="white")

    fig.suptitle(f"RL Driving: {scenario.upper()} — {cfg['description']}",
                fontsize=15, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="RL Driving Demo")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["highway", "intersection", "roundabout", "merge", "all"])
    parser.add_argument("--model_dir", type=str, default="outputs/rl_sim")
    parser.add_argument("--save_dir", type=str, default="outputs/rl_sim")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes per scenario")
    args = parser.parse_args()

    scenarios = list(ENV_CONFIGS.keys()) if args.scenario == "all" else [args.scenario]

    print("=" * 60)
    print("  RL Autonomous Driving Simulation Demo")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print("=" * 60)

    save_dir = Path(args.save_dir)

    for scenario in scenarios:
        cfg = ENV_CONFIGS[scenario]
        env_id = cfg["env_id"]
        algo_name = cfg["algo"]
        env_config = cfg["config"]

        print(f"\n--- {scenario.upper()}: {cfg['description']} ---")

        # Load trained model
        model_path = Path(args.model_dir) / scenario / f"{scenario}_final.zip"
        best_path = Path(args.model_dir) / scenario / "best_model.zip"

        model = None
        if best_path.exists():
            AlgoClass = DQN if algo_name == "DQN" else PPO
            model = AlgoClass.load(str(best_path))
            print(f"  Loaded best model from {best_path}")
        elif model_path.exists():
            AlgoClass = DQN if algo_name == "DQN" else PPO
            model = AlgoClass.load(str(model_path))
            print(f"  Loaded model from {model_path}")
        else:
            print(f"  No trained model found at {model_path}")
            print(f"  Run: python -m src.rl_sim.train --scenario {scenario}")
            continue

        # Create environment
        env = gym.make(env_id, render_mode="rgb_array")
        env.unwrapped.configure(env_config)

        # Run episodes
        best_random_reward = -float("inf")
        best_trained_reward = -float("inf")
        best_random_data = None
        best_trained_data = None

        for ep in range(args.episodes):
            # Random agent
            env.reset(seed=ep)
            frames_r, rewards_r, actions_r, info_r = rollout_episode(env, model=None)
            total_r = sum(rewards_r)
            if total_r > best_random_reward:
                best_random_reward = total_r
                best_random_data = (frames_r, rewards_r)

            # Trained agent
            env.reset(seed=ep)
            frames_t, rewards_t, actions_t, info_t = rollout_episode(env, model=model)
            total_t = sum(rewards_t)
            if total_t > best_trained_reward:
                best_trained_reward = total_t
                best_trained_data = (frames_t, rewards_t)

        print(f"  Random agent:  best reward = {best_random_reward:.2f}")
        print(f"  Trained agent: best reward = {best_trained_reward:.2f}")
        print(f"  Improvement:   {best_trained_reward - best_random_reward:+.2f}")

        # Visualize best episodes
        scenario_dir = save_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        visualize_scenario(
            scenario,
            best_random_data[0], best_random_data[1],
            best_trained_data[0], best_trained_data[1],
            str(scenario_dir / f"{scenario}_comparison.png"),
        )

        # Save individual frames as filmstrip
        n_strip = 8
        trained_frames = best_trained_data[0]
        indices = np.linspace(0, len(trained_frames) - 1, n_strip, dtype=int)
        strip_frames = [trained_frames[i] for i in indices]

        fig_strip, axes = plt.subplots(1, n_strip, figsize=(3 * n_strip, 3), facecolor="#1a1a1a")
        for i, (ax, frame) in enumerate(zip(axes, strip_frames)):
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(f"t={indices[i]}", fontsize=9, color="#00ff88")
        fig_strip.suptitle(f"{scenario.upper()} — Trained RL Agent", fontsize=13,
                          fontweight="bold", color="white")
        plt.tight_layout()
        fig_strip.savefig(str(scenario_dir / f"{scenario}_filmstrip.png"),
                         dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig_strip)

        try:
            env.close()
        except Exception:
            pass

    # Create combined overview
    print(f"\nCreating combined overview...")
    available = [s for s in scenarios if (save_dir / s / f"{s}_comparison.png").exists()]

    if available:
        fig_overview = plt.figure(figsize=(20, 5 * len(available)), facecolor="#1a1a1a")

        for i, scenario in enumerate(available):
            ax = fig_overview.add_subplot(len(available), 1, i + 1)
            img = np.array(Image.open(save_dir / scenario / f"{scenario}_comparison.png"))
            ax.imshow(img)
            ax.axis("off")

        fig_overview.suptitle("RL Autonomous Driving — All Scenarios",
                            fontsize=18, fontweight="bold", color="white", y=1.0)
        plt.tight_layout()
        fig_overview.savefig(str(save_dir / "rl_overview.png"),
                           dpi=100, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"Saved overview to {save_dir}/rl_overview.png")

    print(f"\nAll results saved to {save_dir}/")


if __name__ == "__main__":
    main()
