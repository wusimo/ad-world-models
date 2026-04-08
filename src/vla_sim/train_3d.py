"""
VLA model for 3D first-person camera input.

Architecture:
    RGB Camera (180x320x3) → CNN encoder → image features
    Language Command → embedding → command features
    Fused → MLP → action (steer, accel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLA3DModel(nn.Module):
    """VLA with 3D camera input (180x320 RGB)."""

    def __init__(
        self,
        num_commands: int = 8,
        command_embed_dim: int = 64,
        image_feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Image encoder for 180x320 RGB
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),  # 180→44, 320→79
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 44→21, 79→38
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),  # 21→10, 38→18
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # → (64, 4, 8)
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, image_feature_dim),
            nn.ReLU(),
        )

        self.command_embedding = nn.Embedding(num_commands, command_embed_dim)

        fusion_dim = image_feature_dim + command_embed_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_mean = nn.Linear(hidden_dim, 2)
        self.action_log_std = nn.Parameter(torch.zeros(2))

        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, image, command_id):
        img_feat = self.image_encoder(image)
        cmd_feat = self.command_embedding(command_id)
        return torch.cat([img_feat, cmd_feat], dim=-1)

    def forward(self, image, command_id):
        fused = self.encode(image, command_id)
        hidden = self.action_head(fused)
        action_mean = torch.tanh(self.action_mean(hidden))
        value = self.value_head(fused)
        return action_mean, value

    def get_action(self, image, command_id, deterministic=False):
        action_mean, value = self.forward(image, command_id)
        if deterministic:
            return action_mean, value
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = torch.clamp(dist.sample(), -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, value, log_prob

    def evaluate_action(self, image, command_id, action):
        action_mean, value = self.forward(image, command_id)
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy
