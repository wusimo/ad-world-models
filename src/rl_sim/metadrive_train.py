"""
Train RL agents in MetaDrive — a 3D autonomous driving simulator.

Supports multiple driving scenarios:
    - Highway: multi-lane highway with traffic
    - Intersection: signalized intersection with turning
    - Roundabout: circular intersection navigation
    - Parking: reverse parking into a spot
    - City: procedurally generated urban roads

Uses PPO from stable-baselines3 for all scenarios.
"""

import argparse
import gymnasium
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


SCENARIOS = {
    "highway": {
        "env_cls": "metadrive.envs.top_down_env.TopDownSingleFrameMetaDriveEnv",
        "config": {
            "num_scenarios": 50,
            "start_seed": 0,
            "map": "SSS",  # straight highway
            "traffic_density": 0.15,
        },
        "policy": "CnnPolicy",
        "total_timesteps": 100_000,
        "description": "Multi-lane highway driving with surrounding traffic",
    },
    "intersection": {
        "env_cls": "metadrive.envs.MetaDriveEnv",
        "config": {
            "use_render": False,
            "image_observation": False,
            "num_scenarios": 50,
            "start_seed": 100,
            "map": "XTXT",  # intersection map
            "traffic_density": 0.15,
            "decision_repeat": 5,
            "vehicle_config": {"lidar": {"num_lasers": 72}},
        },
        "total_timesteps": 100_000,
        "description": "Urban intersection with turns and cross traffic",
    },
    "roundabout": {
        "env_cls": "metadrive.envs.MetaDriveEnv",
        "config": {
            "use_render": False,
            "image_observation": False,
            "num_scenarios": 50,
            "start_seed": 200,
            "map": "O",  # roundabout
            "traffic_density": 0.15,
            "decision_repeat": 5,
            "vehicle_config": {"lidar": {"num_lasers": 72}},
        },
        "total_timesteps": 100_000,
        "description": "Roundabout entry, circulation, and exit",
    },
    "city": {
        "env_cls": "metadrive.envs.MetaDriveEnv",
        "config": {
            "use_render": False,
            "image_observation": False,
            "num_scenarios": 100,
            "start_seed": 300,
            "map": 7,  # random 7-block city
            "traffic_density": 0.1,
            "decision_repeat": 5,
            "vehicle_config": {"lidar": {"num_lasers": 72}},
        },
        "total_timesteps": 150_000,
        "description": "Procedurally generated urban roads with diverse layouts",
    },
}


class ProgressLogger(BaseCallback):
    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0 and len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            print(f"  Step {self.num_timesteps:>7d}  "
                  f"reward={np.mean(rewards):+.2f} +/- {np.std(rewards):.2f}  "
                  f"ep_len={np.mean(lengths):.0f}")
        return True


class NormalizedImageWrapper(gymnasium.ObservationWrapper):
    """Convert float [0,1] image obs to uint8 [0,255] for SB3 CnnPolicy."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8,
        )

    def observation(self, obs):
        return (np.clip(obs, 0, 1) * 255).astype(np.uint8)

    def reset(self, **kwargs):
        kwargs.pop("options", None)
        kwargs.pop("seed", None)
        obs, info = self.env.reset()
        return self.observation(obs), info


def make_env(env_cls_path: str, config: dict, seed: int = 0, use_cnn: bool = False):
    """Import and create a MetaDrive environment."""
    module_path, cls_name = env_cls_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    EnvClass = getattr(module, cls_name)

    env = EnvClass(config=config)
    if use_cnn and len(env.observation_space.shape) == 3:
        env = NormalizedImageWrapper(env)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    return Monitor(env)


def train_scenario(scenario: str, output_dir: str, total_timesteps: int = None):
    cfg = SCENARIOS[scenario]
    timesteps = total_timesteps or cfg["total_timesteps"]

    print(f"\n{'='*60}")
    print(f"  Training: {scenario.upper()}")
    print(f"  {cfg['description']}")
    print(f"  Algorithm: PPO, Steps: {timesteps:,}")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    use_cnn = cfg.get("policy", "MlpPolicy") == "CnnPolicy"
    train_env = make_env(cfg["env_cls"], cfg["config"], seed=0, use_cnn=use_cnn)

    policy = cfg.get("policy", "MlpPolicy")
    model = PPO(
        policy, train_env,
        learning_rate=3e-4,
        n_steps=256 if policy == "CnnPolicy" else 512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cuda",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[ProgressLogger()],
    )

    model.save(str(output_path / f"{scenario}_final"))
    print(f"\n  Saved to {output_path / f'{scenario}_final'}")

    # Evaluate using the same env
    print(f"  Final evaluation (10 episodes):")
    rewards = []
    for ep in range(10):
        obs, _ = train_env.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = train_env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    print(f"  Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    train_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agents in MetaDrive")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--output_dir", type=str, default="outputs/metadrive")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    print("=" * 60)
    print("  MetaDrive RL Training — 3D Autonomous Driving Simulator")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print("=" * 60)

    for scenario in scenarios:
        train_scenario(scenario, f"{args.output_dir}/{scenario}", args.timesteps)

    print(f"\nAll training complete! Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
