"""
3D camera-based language-conditioned driving environment.

Wraps MetaDrive with RGB camera sensor for first-person view input,
plus language command from navigation system.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from src.vla_sim.env import COMMANDS, COMMAND_LIST


class LanguageDrivingEnv3D(gymnasium.Wrapper):
    """3D camera + language → action environment."""

    def __init__(self, base_env, command_bonus: float = 0.5):
        super().__init__(base_env)
        self.command_bonus = command_bonus
        self.current_command = "go_forward"

        # Image is (H, W, C, T) — we use only last frame
        img_shape = (180, 320, 3)  # H, W, C
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8),
            "command_id": spaces.Discrete(len(COMMAND_LIST)),
        })

    def _get_command(self, info):
        if info.get("navigation_left"):
            return "turn_left"
        elif info.get("navigation_right"):
            return "turn_right"
        return "go_forward"

    def _make_obs(self, raw_obs, info):
        img = raw_obs["image"]  # (H, W, C, T)
        if img.ndim == 4:
            img = img[..., -1]
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        self.current_command = self._get_command(info)
        cmd_id = COMMAND_LIST.index(self.current_command)
        return {"image": img_uint8, "command_id": cmd_id}

    def _command_bonus(self, action):
        steer, accel = action
        if self.current_command == "go_forward":
            return 0.1 * (1.0 - abs(steer))
        elif self.current_command == "turn_left":
            return 0.2 * max(0, steer)
        elif self.current_command == "turn_right":
            return 0.2 * max(0, -steer)
        return 0.0

    def reset(self, **kwargs):
        kwargs.pop("options", None)
        kwargs.pop("seed", None)
        raw_obs, info = self.env.reset()
        self.current_command = "go_forward"
        return self._make_obs(raw_obs, info), info

    def step(self, action):
        raw_obs, reward, term, trunc, info = self.env.step(action)
        bonus = self._command_bonus(action) * self.command_bonus
        obs = self._make_obs(raw_obs, info)
        info["command"] = self.current_command
        info["command_text"] = COMMANDS[self.current_command]
        return obs, reward + bonus, term, trunc, info

    @property
    def command_text(self):
        return COMMANDS[self.current_command]


def make_3d_lang_env(num_scenarios=20, map_type="SSS", traffic=0.15):
    """Create 3D RGB camera + language driving env."""
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    base = MetaDriveEnv(config={
        "use_render": False,
        "image_observation": True,
        "sensors": {"rgb_camera": (RGBCamera, 320, 180)},
        "vehicle_config": {"image_source": "rgb_camera"},
        "num_scenarios": num_scenarios,
        "map": map_type,
        "traffic_density": traffic,
        "norm_pixel": True,
        "stack_size": 1,
    })
    return LanguageDrivingEnv3D(base)
