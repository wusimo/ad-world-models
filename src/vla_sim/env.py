"""
Language-conditioned driving environment for VLA training.

Wraps MetaDrive to provide:
    - Visual observation: bird's-eye view image (200x200x3)
    - Language command: navigation instruction ("go forward", "turn left", etc.)
    - Action: continuous [steering, acceleration]
    - Reward: shaped for following language commands

The environment automatically generates language instructions based on
the navigation route and provides reward bonuses for following them.
"""

import gymnasium
import numpy as np
from gymnasium import spaces


# Language commands mapped to navigation states
COMMANDS = {
    "go_forward": "Drive forward and maintain your lane.",
    "turn_left": "Turn left at the upcoming intersection.",
    "turn_right": "Turn right at the upcoming intersection.",
    "change_lane_left": "Change to the left lane safely.",
    "change_lane_right": "Change to the right lane safely.",
    "slow_down": "Slow down, obstacle ahead.",
    "speed_up": "Speed up to match traffic flow.",
    "stop": "Come to a complete stop.",
}

COMMAND_LIST = list(COMMANDS.keys())


class LanguageDrivingEnv(gymnasium.Wrapper):
    """
    Language-conditioned driving environment for VLA training.

    Observation: dict with 'image' (200x200x3 uint8) and 'command' (str)
    Action: Box(-1, 1, (2,)) — [steering, acceleration]
    Reward: base driving reward + command-following bonus
    """

    def __init__(self, base_env, command_bonus: float = 0.5):
        super().__init__(base_env)
        self.command_bonus = command_bonus
        self.current_command = "go_forward"
        self.current_command_text = COMMANDS["go_forward"]
        self.command_steps = 0
        self.command_change_interval = 50  # change command every N steps

        # Override observation space to dict
        img_space = spaces.Box(0, 255, shape=(200, 200, 3), dtype=np.uint8)
        cmd_space = spaces.Discrete(len(COMMAND_LIST))
        self.observation_space = spaces.Dict({
            "image": img_space,
            "command_id": cmd_space,
        })

    def _get_command_from_nav(self, info):
        """Extract language command from MetaDrive navigation info."""
        if info.get("navigation_left"):
            return "turn_left"
        elif info.get("navigation_right"):
            return "turn_right"
        elif info.get("navigation_forward"):
            return "go_forward"
        return "go_forward"

    def _compute_command_reward(self, action, info):
        """Bonus reward for following the language command."""
        steer, accel = action
        bonus = 0.0

        if self.current_command == "go_forward":
            bonus = 0.1 * (1.0 - abs(steer))  # reward staying straight
        elif self.current_command == "turn_left":
            bonus = 0.2 * max(0, steer)  # reward turning left
        elif self.current_command == "turn_right":
            bonus = 0.2 * max(0, -steer)  # reward turning right
        elif self.current_command == "slow_down":
            bonus = 0.2 * max(0, -accel)  # reward decelerating
        elif self.current_command == "speed_up":
            bonus = 0.2 * max(0, accel)  # reward accelerating
        elif self.current_command == "change_lane_left":
            bonus = 0.15 * max(0, steer)
        elif self.current_command == "change_lane_right":
            bonus = 0.15 * max(0, -steer)

        return bonus * self.command_bonus

    def _make_obs(self, raw_obs, info):
        """Create dict observation with image and command."""
        img = (np.clip(raw_obs, 0, 1) * 255).astype(np.uint8)

        # Update command from navigation
        nav_cmd = self._get_command_from_nav(info)
        self.current_command = nav_cmd
        self.current_command_text = COMMANDS[nav_cmd]

        cmd_id = COMMAND_LIST.index(self.current_command)
        return {"image": img, "command_id": cmd_id}

    def reset(self, **kwargs):
        kwargs.pop("options", None)
        kwargs.pop("seed", None)
        raw_obs, info = self.env.reset()
        self.command_steps = 0
        self.current_command = "go_forward"
        self.current_command_text = COMMANDS["go_forward"]
        return self._make_obs(raw_obs, info), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.command_steps += 1

        # Add command-following bonus
        cmd_reward = self._compute_command_reward(action, info)
        total_reward = reward + cmd_reward

        obs = self._make_obs(raw_obs, info)

        # Add command info
        info["command"] = self.current_command
        info["command_text"] = self.current_command_text
        info["command_reward"] = cmd_reward

        return obs, total_reward, terminated, truncated, info

    @property
    def command_text(self):
        return self.current_command_text
