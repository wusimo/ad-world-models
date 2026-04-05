"""
VLA (Vision-Language-Action) model for simulated driving.

Architecture:
    Image (200x200x3) → CNN encoder → image features (512)
    Command text → embedding lookup → command features (64)
    [image_features | command_features] → MLP → Action (steer, accel)

This is a lightweight VLA that can be trained with:
    1. Imitation Learning (IL) from expert demonstrations
    2. Reinforcement Learning (RL) via PPO
    3. Combined IL + RL (pretrain with IL, fine-tune with RL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VLADrivingModel(nn.Module):
    """
    Vision-Language-Action model for driving.

    Takes bird's-eye view image + language command → driving actions.
    """

    def __init__(
        self,
        num_commands: int = 8,
        command_embed_dim: int = 64,
        image_feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Image encoder (lightweight CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),  # 200→49
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 49→23
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),  # 23→11
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, image_feature_dim),
            nn.ReLU(),
        )

        # Command encoder
        self.command_embedding = nn.Embedding(num_commands, command_embed_dim)

        # Fusion + action head
        fusion_dim = image_feature_dim + command_embed_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate outputs for mean and log_std (for RL)
        self.action_mean = nn.Linear(hidden_dim, 2)  # [steer, accel]
        self.action_log_std = nn.Parameter(torch.zeros(2))

        # Value head (for RL critic)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, image, command_id):
        """Encode image and command into fused features."""
        # image: (B, 3, H, W) float32 normalized to [0,1]
        img_feat = self.image_encoder(image)
        cmd_feat = self.command_embedding(command_id)
        return torch.cat([img_feat, cmd_feat], dim=-1)

    def forward(self, image, command_id):
        """Forward pass: returns action mean and value."""
        fused = self.encode(image, command_id)
        hidden = self.action_head(fused)
        action_mean = torch.tanh(self.action_mean(hidden))  # clamp to [-1, 1]
        value = self.value_head(fused)
        return action_mean, value

    def get_action(self, image, command_id, deterministic=False):
        """Sample or deterministic action for inference."""
        action_mean, value = self.forward(image, command_id)
        if deterministic:
            return action_mean, value
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, value, log_prob

    def evaluate_action(self, image, command_id, action):
        """Evaluate log_prob and entropy for PPO update."""
        action_mean, value = self.forward(image, command_id)
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy
