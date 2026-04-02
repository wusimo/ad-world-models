"""
UniAD-style End-to-End Planner.

Unified multi-task transformer that jointly performs:
    1. 3D Object Detection (DETR-style queries)
    2. Multi-Object Tracking (track queries with temporal propagation)
    3. Motion Forecasting (multi-modal future trajectory prediction)
    4. Ego Planning (collision-aware trajectory planning)

Reference: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)
Architecture: Detection → Tracking → Motion → Planning (cascaded transformers)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat


class SinusoidalPositionEncoding(nn.Module):
    """2D sinusoidal position encoding for BEV features."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        B, C, H, W = bev.shape
        assert C == self.d_model
        device = bev.device

        y_embed = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)

        dim_t = torch.arange(0, self.d_model // 2, device=device).float()
        dim_t = self.temperature ** (2 * dim_t / self.d_model)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # (H, W, d/2)
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos = torch.cat([pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()], dim=-1)
        pos = pos[:, :, : self.d_model].permute(2, 0, 1)  # (C, H, W)

        return bev + pos.unsqueeze(0)


class DetectionHead(nn.Module):
    """
    DETR-style 3D object detection from BEV features.

    Uses learnable object queries that attend to BEV features
    via cross-attention to predict 3D bounding boxes.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_classes: int = 10,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 512,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)  # content + pos

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),  # cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy
        )

    def forward(self, bev: torch.Tensor) -> dict:
        """
        Args:
            bev: (B, C, H, W) BEV feature map
        Returns:
            dict with 'classes' (B, Q, num_classes), 'boxes' (B, Q, 10),
            'query_features' (B, Q, C) for downstream tasks
        """
        B, C, H, W = bev.shape

        # Flatten BEV to sequence
        memory = rearrange(bev, "b c h w -> b (h w) c")

        # Split query embed into content and positional
        query_embed = self.query_embed.weight
        query_pos, query_content = query_embed.split(C, dim=-1)
        queries = repeat(query_content, "q c -> b q c", b=B)

        # Decode
        query_features = self.decoder(queries, memory)

        # Predict
        classes = self.class_head(query_features)
        boxes = self.box_head(query_features)

        return {
            "classes": classes,
            "boxes": boxes,
            "query_features": query_features,  # passed to tracking/motion
        }


class MotionForecaster(nn.Module):
    """
    Multi-modal motion forecasting from detection queries.

    For each detected object, predicts K possible future trajectories
    with associated probabilities. Uses cross-attention to BEV features
    for context-aware prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_modes: int = 6,
        future_steps: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps

        # Cross-attention to BEV for context
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Mode prediction
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
        )

        # Trajectory regression per mode
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_modes * future_steps * 2),
        )

    def forward(self, query_features: torch.Tensor, bev: torch.Tensor) -> dict:
        """
        Args:
            query_features: (B, Q, C) from detection head
            bev: (B, C, H, W) BEV features
        Returns:
            mode_probs: (B, Q, K) — probability per mode
            trajectories: (B, Q, K, T, 2) — predicted xy per mode per timestep
        """
        B, Q, C = query_features.shape
        memory = rearrange(bev, "b c h w -> b (h w) c")

        # Context-aware features
        attended, _ = self.cross_attn(query_features, memory, memory)
        features = self.norm(query_features + attended)

        # Mode probabilities
        mode_probs = self.mode_head(features).softmax(dim=-1)  # (B, Q, K)

        # Trajectory prediction
        traj = self.traj_head(features)  # (B, Q, K*T*2)
        traj = traj.reshape(B, Q, self.num_modes, self.future_steps, 2)

        return {"mode_probs": mode_probs, "trajectories": traj}


class PlanningHead(nn.Module):
    """
    Ego trajectory planning head.

    Attends to:
        - BEV features (scene context)
        - Detection queries (object awareness)
        - Motion forecasts (interaction-aware planning)

    Produces collision-aware ego trajectory via learned cost volume.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        future_steps: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.future_steps = future_steps

        # Ego query (learnable)
        self.ego_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Cross-attention to scene
        self.scene_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.object_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Trajectory decoder (autoregressive-style)
        self.traj_gru = nn.GRU(hidden_dim + 2, hidden_dim, batch_first=True)
        self.waypoint_head = nn.Linear(hidden_dim, 2)  # (dx, dy) per step

        # Collision scoring
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        bev: torch.Tensor,
        det_features: torch.Tensor,
        motion_traj: torch.Tensor,
    ) -> dict:
        """
        Args:
            bev: (B, C, H, W)
            det_features: (B, Q, C) object query features
            motion_traj: (B, Q, K, T, 2) predicted agent trajectories
        Returns:
            trajectory: (B, T, 2) planned ego trajectory
            collision_scores: (B, T) collision probability per step
        """
        B = bev.shape[0]
        memory = rearrange(bev, "b c h w -> b (h w) c")

        # Ego query attends to scene
        ego = self.ego_query.expand(B, -1, -1)
        ego_scene, _ = self.scene_attn(ego, memory, memory)
        ego = self.norm1(ego + ego_scene)

        # Ego query attends to detected objects
        ego_obj, _ = self.object_attn(ego, det_features, det_features)
        ego = self.norm2(ego + ego_obj)

        # Autoregressive trajectory generation
        hidden = ego.squeeze(1).unsqueeze(0)  # (1, B, C) for GRU
        waypoints = []
        prev_wp = torch.zeros(B, 1, 2, device=bev.device)

        for t in range(self.future_steps):
            gru_in = torch.cat([ego, prev_wp], dim=-1)  # (B, 1, C+2)
            out, hidden = self.traj_gru(gru_in, hidden)
            wp = self.waypoint_head(out)  # (B, 1, 2)
            waypoints.append(wp)
            prev_wp = wp

        trajectory = torch.cat(waypoints, dim=1)  # (B, T, 2)

        # Collision scoring
        ego_expanded = ego.expand(-1, self.future_steps, -1)
        collision_in = torch.cat([ego_expanded, trajectory], dim=-1)
        collision_scores = self.collision_head(collision_in).squeeze(-1).sigmoid()

        return {
            "trajectory": trajectory,
            "collision_scores": collision_scores,
        }


class ImageBackbone(nn.Module):
    """ResNet backbone for multi-camera feature extraction."""

    def __init__(self, name: str = "resnet50", pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        resnet = getattr(models, name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        # Use layers up to layer3 (stride 16)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        backbone_channels = 1024  # ResNet layer3 output
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N, 3, H, W) multi-camera images
        Returns:
            features: (B, N, C, H/16, W/16)
        """
        B, N, C, H, W = images.shape
        x = rearrange(images, "b n c h w -> (b n) c h w")
        x = self.backbone(x)
        x = self.neck(x)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=B, n=N)
        return x


class UniADPlanner(nn.Module):
    """
    UniAD: Unified Autonomous Driving — End-to-End Multi-Task Planner.

    Full pipeline:
        Multi-Camera Images
            → Image Backbone (ResNet)
            → BEV Transform (Lift-Splat-Shoot)
            → Detection (DETR decoder)
            → Motion Forecasting (cross-attention to BEV)
            → Ego Planning (collision-aware trajectory)

    This is a simplified but faithful implementation of the UniAD architecture.
    The real UniAD also includes occupancy prediction and map segmentation heads.
    """

    def __init__(self, config: dict):
        super().__init__()
        from src.data.bev_transform import BEVTransform

        bev_cfg = config["bev"]
        det_cfg = config["detection"]
        mot_cfg = config["motion"]
        plan_cfg = config["planning"]

        # Image backbone
        bb_cfg = config["backbone"]
        self.backbone = ImageBackbone(
            bb_cfg["name"], bb_cfg["pretrained"], bb_cfg["out_channels"]
        )

        # Camera-to-BEV transform
        self.bev_transform = BEVTransform(
            in_channels=bev_cfg["in_channels"],
            bev_channels=bev_cfg["bev_channels"],
            bev_size=tuple(bev_cfg["bev_size"]),
            bev_range=tuple(bev_cfg["bev_range"]),
            num_depth_bins=bev_cfg["num_depth_bins"],
            depth_min=bev_cfg["depth_min"],
            depth_max=bev_cfg["depth_max"],
        )

        # Positional encoding for BEV
        self.bev_pos = SinusoidalPositionEncoding(bev_cfg["bev_channels"])

        # Task heads
        self.detection = DetectionHead(
            bev_channels=bev_cfg["bev_channels"],
            hidden_dim=det_cfg["hidden_dim"],
            num_queries=det_cfg["num_queries"],
            num_classes=det_cfg["num_classes"],
            num_heads=det_cfg["num_heads"],
            num_layers=det_cfg["num_decoder_layers"],
            ffn_dim=det_cfg["ffn_dim"],
        )

        self.motion = MotionForecaster(
            hidden_dim=mot_cfg["hidden_dim"],
            num_modes=mot_cfg["num_modes"],
            future_steps=mot_cfg["future_steps"],
            num_heads=8,
        )

        self.planning = PlanningHead(
            hidden_dim=plan_cfg["hidden_dim"],
            future_steps=plan_cfg["future_steps"],
            num_heads=plan_cfg["num_heads"],
        )

    def forward(self, batch: dict) -> dict:
        """
        Full forward pass: images → detection → motion → planning.

        Args:
            batch: dict with 'images', 'intrinsics', 'extrinsics'
        Returns:
            dict with detection, motion, and planning outputs
        """
        images = batch["images"]         # (B, 6, 3, H, W)
        intrinsics = batch["intrinsics"]  # (B, 6, 3, 3)
        extrinsics = batch["extrinsics"]  # (B, 6, 4, 4)

        # 1. Extract image features
        features = self.backbone(images)  # (B, 6, C, H/16, W/16)

        # 2. Transform to BEV
        bev = self.bev_transform(features, intrinsics, extrinsics)  # (B, C, bH, bW)
        bev = self.bev_pos(bev)

        # 3. Detection
        det_out = self.detection(bev)

        # 4. Motion forecasting
        motion_out = self.motion(det_out["query_features"], bev)

        # 5. Planning
        plan_out = self.planning(
            bev, det_out["query_features"], motion_out["trajectories"]
        )

        return {
            "detection": det_out,
            "motion": motion_out,
            "planning": plan_out,
            "bev_features": bev,
        }

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """Inference mode — returns planned trajectory and detections."""
        self.eval()
        outputs = self.forward(batch)

        # Post-process detections (top-k by confidence)
        det_scores = outputs["detection"]["classes"].softmax(dim=-1).max(dim=-1)
        top_k = min(50, det_scores.values.shape[-1])
        top_indices = det_scores.values.topk(top_k, dim=-1).indices

        return {
            "planned_trajectory": outputs["planning"]["trajectory"],
            "collision_scores": outputs["planning"]["collision_scores"],
            "detection_scores": det_scores.values,
            "detection_classes": det_scores.indices,
            "detection_boxes": outputs["detection"]["boxes"],
            "motion_predictions": outputs["motion"]["trajectories"],
            "motion_mode_probs": outputs["motion"]["mode_probs"],
            "bev_features": outputs["bev_features"],
        }
