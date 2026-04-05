"""
Train RL agents for autonomous driving in simulation.

Uses highway-env (lightweight driving simulator) with PPO/DQN from stable-baselines3.
Supports multiple driving scenarios:
    - Highway: lane changing and speed control on a multi-lane highway
    - Intersection: navigating through a T-intersection with cross traffic
    - Roundabout: entering and exiting a roundabout
    - Merge: merging onto a highway from an on-ramp
"""

import argparse
import gymnasium as gym
import highway_env
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# Environment configurations optimized for each scenario
ENV_CONFIGS = {
    "highway": {
        "env_id": "highway-fast-v0",
        "algo": "DQN",
        "config": {
            "observation": {"type": "Kinematics", "vehicles_count": 10},
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 30,
            "duration": 60,
            "reward_speed_range": [20, 30],
            "collision_reward": -1.0,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0.0,
        },
        "total_timesteps": 50_000,
        "description": "Lane changing and speed control on a 4-lane highway",
    },
    "intersection": {
        "env_id": "intersection-v1",
        "algo": "PPO",
        "config": {
            "observation": {"type": "Kinematics", "vehicles_count": 10,
                          "features": ["presence", "x", "y", "vx", "vy",
                                     "cos_h", "sin_h", "cos_d"]},
            "action": {"type": "ContinuousAction"},
            "duration": 13,
            "destination": "o1",
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "collision_reward": -5.0,
            "arrived_reward": 1.0,
            "high_speed_reward": 1.0,
        },
        "total_timesteps": 80_000,
        "description": "Navigating a T-intersection with cross traffic",
    },
    "roundabout": {
        "env_id": "roundabout-v0",
        "algo": "DQN",
        "config": {
            "observation": {"type": "Kinematics", "vehicles_count": 10},
            "action": {"type": "DiscreteMetaAction"},
            "duration": 20,
            "collision_reward": -1.0,
        },
        "total_timesteps": 50_000,
        "description": "Entering and exiting a roundabout",
    },
    "merge": {
        "env_id": "merge-v0",
        "algo": "DQN",
        "config": {
            "observation": {"type": "Kinematics", "vehicles_count": 10},
            "action": {"type": "DiscreteMetaAction"},
            "duration": 20,
            "collision_reward": -1.0,
        },
        "total_timesteps": 50_000,
        "description": "Merging onto a highway from an on-ramp",
    },
}


class TrainingLogger(BaseCallback):
    """Log training progress."""

    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"  Step {self.num_timesteps:>7d}  "
                      f"mean_reward={mean_reward:+.2f}  "
                      f"mean_ep_len={mean_len:.0f}")
        return True


def make_env(env_id, config, seed=0):
    """Create a configured highway-env environment."""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.unwrapped.configure(config)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train_scenario(scenario: str, output_dir: str, total_timesteps: int = None):
    """Train an RL agent for a specific driving scenario."""
    cfg = ENV_CONFIGS[scenario]
    env_id = cfg["env_id"]
    algo_name = cfg["algo"]
    env_config = cfg["config"]
    timesteps = total_timesteps or cfg["total_timesteps"]

    print(f"\n{'='*60}")
    print(f"  Training: {scenario.upper()}")
    print(f"  {cfg['description']}")
    print(f"  Algorithm: {algo_name}, Steps: {timesteps:,}")
    print(f"{'='*60}")

    # Create environments (single env for simplicity and compatibility)
    train_env = make_env(env_id, env_config, seed=0)()
    eval_env = make_env(env_id, env_config, seed=42)()

    # Create model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if algo_name == "DQN":
        model = DQN(
            "MlpPolicy", train_env,
            learning_rate=5e-4,
            buffer_size=15_000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            exploration_fraction=0.3,
            verbose=0,
        )
    else:  # PPO
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )
    log_callback = TrainingLogger(log_interval=5000)

    # Train
    model.learn(total_timesteps=timesteps, callback=[eval_callback, log_callback])

    # Save final model
    model_path = output_path / f"{scenario}_final"
    model.save(str(model_path))
    print(f"\n  Saved model to {model_path}")

    # Quick evaluation
    print(f"\n  Final evaluation (10 episodes):")
    rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    print(f"  Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for driving")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["highway", "intersection", "roundabout", "merge", "all"])
    parser.add_argument("--output_dir", type=str, default="outputs/rl_sim")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override training timesteps")
    args = parser.parse_args()

    scenarios = list(ENV_CONFIGS.keys()) if args.scenario == "all" else [args.scenario]

    print("=" * 60)
    print("  RL Training for Autonomous Driving Simulation")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print("=" * 60)

    for scenario in scenarios:
        train_scenario(scenario, f"{args.output_dir}/{scenario}", args.timesteps)

    print(f"\nAll training complete! Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
