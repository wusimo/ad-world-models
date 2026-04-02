"""
Driving World Model (Vista/GenAD-style).

Learns a latent dynamics model of driving scenes:
    1. Encode BEV observations into latent space (VAE)
    2. Predict future latent states conditioned on actions (Temporal Transformer)
    3. Decode latent states back to BEV predictions
    4. Plan via Model Predictive Control (MPC) in latent space

References:
    - "Vista: A Generalizable Driving World Model"
    - "GenAD: Generalized Predictive Model for Autonomous Driving"
    - "World Models" (Ha & Schmidhuber, 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class BEVEncoder(nn.Module):
    """Encode BEV features into a compact latent representation."""

    def __init__(self, in_channels: int = 256, latent_dim: int = 64, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        layers = []
        ch = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(ch, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            ch = h_dim

        self.encoder = nn.Sequential(*layers)
        # After 3 stride-2 convs on 200x200: 25x25
        self.fc_mu = nn.Conv2d(ch, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(ch, latent_dim, 1)

    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev: (B, C, H, W) BEV feature map
        Returns:
            mu: (B, latent_dim, h, w)
            logvar: (B, latent_dim, h, w)
        """
        h = self.encoder(bev)
        return self.fc_mu(h), self.fc_logvar(h)


class BEVDecoder(nn.Module):
    """Decode latent representation back to BEV features."""

    def __init__(self, latent_dim: int = 64, out_channels: int = 256, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        ch = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(ch, h_dim, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            ch = h_dim

        layers.append(nn.Conv2d(ch, out_channels, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, h, w)
        Returns:
            bev_recon: (B, out_channels, H, W)
        """
        return self.decoder(z)


class ActionEncoder(nn.Module):
    """Encode driving actions into embeddings."""

    def __init__(self, action_dim: int = 3, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class TemporalTransformer(nn.Module):
    """
    Transformer that predicts future latent BEV states from current state + actions.

    Uses causal attention — each future step can only attend to past/current steps.
    Actions are injected via cross-attention at each layer.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        spatial_size: int = 25,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_steps: int = 12,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.seq_dim = latent_dim  # channel dim of latent

        # Flatten spatial dims and use latent_dim as feature dim
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)  # global pool per channel
        self.spatial_expand = nn.Linear(latent_dim, latent_dim * spatial_size * spatial_size)

        # Temporal position encoding
        self.time_embed = nn.Embedding(max_steps, latent_dim)

        # Action conditioning
        self.action_proj = nn.Linear(64, latent_dim)  # action_embed_dim → latent_dim

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        z_current: torch.Tensor,
        actions: torch.Tensor,
        action_embeds: torch.Tensor,
        num_future: int,
    ) -> torch.Tensor:
        """
        Predict future latent states autoregressively.

        Args:
            z_current: (B, latent_dim, h, w) — current latent state
            actions: not used directly (kept for interface)
            action_embeds: (B, T, action_embed_dim) — future action embeddings
            num_future: number of future steps to predict

        Returns:
            z_future: (B, T, latent_dim, h, w) — predicted future latent states
        """
        B = z_current.shape[0]
        h, w = z_current.shape[2], z_current.shape[3]
        device = z_current.device

        # Pool current state to sequence token
        z_pooled = self.spatial_pool(z_current).squeeze(-1).squeeze(-1)  # (B, latent_dim)

        # Build sequence: [z_0, a_0, z_1_pred, a_1, ...]
        # Start with current state token
        tokens = [z_pooled + self.time_embed(torch.zeros(B, dtype=torch.long, device=device))]

        predicted_states = []

        for t in range(num_future):
            # Add action embedding for this step
            a_t = self.action_proj(action_embeds[:, t])  # (B, latent_dim)
            tokens.append(a_t)

            # Create time embedding for next state
            time_idx = torch.full((B,), t + 1, dtype=torch.long, device=device)
            time_emb = self.time_embed(time_idx)

            # Stack all tokens so far
            seq = torch.stack(tokens, dim=1)  # (B, seq_len, latent_dim)

            # Causal attention
            mask = self._causal_mask(seq.shape[1], device)
            out = self.transformer(seq, mask=mask)

            # Take last token as predicted next state
            z_next = self.output_proj(out[:, -1]) + time_emb  # (B, latent_dim)
            tokens.append(z_next)

            predicted_states.append(z_next)

        # Stack predictions
        z_future_pooled = torch.stack(predicted_states, dim=1)  # (B, T, latent_dim)

        # Expand back to spatial dims
        z_future = self.spatial_expand(z_future_pooled)  # (B, T, latent_dim * h * w)
        z_future = z_future.reshape(B, num_future, self.latent_dim, h, w)

        return z_future


class LatentMPC(nn.Module):
    """
    Model Predictive Control in latent space.

    Samples candidate action sequences, rolls them out through the world model,
    evaluates costs (collision, progress, comfort), and selects the best trajectory.
    """

    def __init__(
        self,
        horizon: int = 6,
        num_samples: int = 64,
        action_dim: int = 3,
        temperature: float = 0.1,
        collision_weight: float = 10.0,
        progress_weight: float = 1.0,
        comfort_weight: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.temperature = temperature
        self.collision_weight = collision_weight
        self.progress_weight = progress_weight
        self.comfort_weight = comfort_weight

        # Learned cost networks
        self.collision_cost = nn.Sequential(
            nn.Linear(64, 32),  # latent_dim → 32
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.progress_cost = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _sample_actions(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample candidate action sequences."""
        # Sample from a prior distribution (Gaussian centered on "go straight")
        mean = torch.zeros(B, self.num_samples, self.horizon, self.action_dim, device=device)
        mean[:, :, :, 1] = 0.5  # Default: slight forward acceleration
        std = torch.ones_like(mean) * 0.3
        std[:, :, :, 0] = 0.15  # Less steering variation
        return mean + std * torch.randn_like(mean)

    def _compute_cost(
        self,
        z_futures: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost for each candidate trajectory.

        Args:
            z_futures: (B, K, T, latent_dim) — pooled latent predictions
            actions: (B, K, T, action_dim)
        Returns:
            costs: (B, K)
        """
        B, K, T, D = z_futures.shape

        z_flat = z_futures.reshape(B * K * T, D)

        # Collision cost (per step)
        col_cost = self.collision_cost(z_flat).reshape(B, K, T)
        col_cost = col_cost.sum(dim=-1) * self.collision_weight

        # Progress cost (reward forward motion)
        prog_cost = -self.progress_cost(z_flat).reshape(B, K, T)
        prog_cost = prog_cost.sum(dim=-1) * self.progress_weight

        # Comfort cost (penalize jerky actions)
        action_diff = (actions[:, :, 1:] - actions[:, :, :-1]).norm(dim=-1)
        comfort_cost = action_diff.sum(dim=-1) * self.comfort_weight

        return col_cost + prog_cost + comfort_cost

    def forward(
        self,
        z_current: torch.Tensor,
        world_model_rollout_fn,
        action_encoder: nn.Module,
    ) -> dict:
        """
        Plan via sampling-based MPC.

        Args:
            z_current: (B, latent_dim, h, w)
            world_model_rollout_fn: callable(z, action_embeds, T) → z_future
            action_encoder: encodes raw actions to embeddings
        Returns:
            best_actions: (B, T, action_dim)
            best_trajectory_latents: (B, T, latent_dim, h, w)
        """
        B = z_current.shape[0]
        device = z_current.device

        # Sample candidate action sequences
        candidate_actions = self._sample_actions(B, device)  # (B, K, T, action_dim)

        # Roll out each candidate through world model
        all_futures = []
        for k in range(self.num_samples):
            actions_k = candidate_actions[:, k]  # (B, T, action_dim)
            action_embeds_k = action_encoder(actions_k)  # (B, T, embed_dim)
            z_future_k = world_model_rollout_fn(
                z_current, actions_k, action_embeds_k, self.horizon
            )  # (B, T, latent_dim, h, w)
            all_futures.append(z_future_k)

        all_futures = torch.stack(all_futures, dim=1)  # (B, K, T, latent_dim, h, w)

        # Pool spatial dims for cost computation
        z_pooled = all_futures.mean(dim=(-2, -1))  # (B, K, T, latent_dim)

        # Compute costs
        costs = self._compute_cost(z_pooled, candidate_actions)  # (B, K)

        # Select best via softmin
        weights = F.softmax(-costs / self.temperature, dim=1)  # (B, K)
        best_idx = weights.argmax(dim=1)  # (B,)

        # Gather best actions and latents
        best_actions = candidate_actions[
            torch.arange(B, device=device), best_idx
        ]  # (B, T, action_dim)
        best_latents = all_futures[
            torch.arange(B, device=device), best_idx
        ]  # (B, T, latent_dim, h, w)

        return {
            "actions": best_actions,
            "trajectory_latents": best_latents,
            "all_costs": costs,
            "best_cost": costs[torch.arange(B, device=device), best_idx],
        }


class DrivingWorldModel(nn.Module):
    """
    Complete driving world model with latent dynamics and MPC planning.

    Architecture:
        BEV → VAE Encoder → Latent Space
            → Temporal Transformer (action-conditioned future prediction)
            → VAE Decoder → Predicted future BEV
            → MPC Planner → Optimal action sequence

    Training:
        1. Reconstruction: BEV → encode → decode → BEV (VAE loss)
        2. Prediction: z_t + a_t → z_{t+1} (next-state prediction)
        3. Planning: MPC cost learning (collision, progress objectives)
    """

    def __init__(self, config: dict):
        super().__init__()
        vae_cfg = config["vae"]
        temp_cfg = config["temporal"]
        act_cfg = config["action"]
        mpc_cfg = config["mpc"]

        # VAE
        self.encoder = BEVEncoder(
            in_channels=vae_cfg["in_channels"],
            latent_dim=vae_cfg["latent_dim"],
            hidden_dims=vae_cfg["hidden_dims"],
        )
        self.decoder = BEVDecoder(
            latent_dim=vae_cfg["latent_dim"],
            out_channels=vae_cfg["in_channels"],
            hidden_dims=list(reversed(vae_cfg["hidden_dims"])),
        )
        self.kl_weight = vae_cfg["kl_weight"]

        # Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=act_cfg["dim"],
            embed_dim=act_cfg["embed_dim"],
        )

        # Temporal dynamics
        self.temporal = TemporalTransformer(
            latent_dim=temp_cfg["latent_dim"],
            spatial_size=temp_cfg["bev_latent_size"][0],
            num_heads=temp_cfg["num_heads"],
            num_layers=temp_cfg["num_layers"],
            ffn_dim=temp_cfg["ffn_dim"],
            dropout=temp_cfg["dropout"],
            max_steps=temp_cfg["max_future_steps"],
        )

        # MPC planner
        self.mpc = LatentMPC(
            horizon=mpc_cfg["horizon"],
            num_samples=mpc_cfg["num_samples"],
            action_dim=act_cfg["dim"],
            temperature=mpc_cfg["temperature"],
            collision_weight=mpc_cfg["collision_cost_weight"],
            progress_weight=mpc_cfg["progress_cost_weight"],
            comfort_weight=mpc_cfg["comfort_cost_weight"],
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode BEV to latent space."""
        mu, logvar = self.encoder(bev)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to BEV."""
        return self.decoder(z)

    def predict_future(
        self,
        z_current: torch.Tensor,
        actions: torch.Tensor,
        action_embeds: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Predict future latent states given current state and actions."""
        return self.temporal(z_current, actions, action_embeds, num_steps)

    def forward(self, bev: torch.Tensor, future_bevs: torch.Tensor = None, actions: torch.Tensor = None) -> dict:
        """
        Full forward pass: encode → predict → decode.

        Args:
            bev: (B, C, H, W) current BEV
            future_bevs: (B, T, C, H, W) future BEV ground truth (for training)
            actions: (B, T, action_dim) ego actions (for prediction)
        Returns:
            dict with reconstructions, predictions, and losses
        """
        # Encode current BEV
        z, mu, logvar = self.encode(bev)

        # Reconstruction
        bev_recon = self.decode(z)
        recon_loss = F.mse_loss(bev_recon, bev)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        result = {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "bev_recon": bev_recon,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "vae_loss": recon_loss + self.kl_weight * kl_loss,
        }

        # Future prediction (if actions provided)
        if actions is not None:
            action_embeds = self.action_encoder(actions)
            T = actions.shape[1]
            z_future = self.predict_future(z, actions, action_embeds, T)

            # Decode future predictions
            B, T_pred = z_future.shape[:2]
            z_flat = rearrange(z_future, "b t c h w -> (b t) c h w")
            bev_future_pred = self.decode(z_flat)
            bev_future_pred = rearrange(bev_future_pred, "(b t) c h w -> b t c h w", b=B)

            result["z_future"] = z_future
            result["bev_future_pred"] = bev_future_pred

            # Future prediction loss (if GT provided)
            if future_bevs is not None:
                T_min = min(T_pred, future_bevs.shape[1])
                pred_loss = F.mse_loss(
                    bev_future_pred[:, :T_min], future_bevs[:, :T_min]
                )
                result["prediction_loss"] = pred_loss
                result["total_loss"] = result["vae_loss"] + pred_loss

        return result

    @torch.no_grad()
    def plan(self, bev: torch.Tensor) -> dict:
        """
        Plan optimal actions via MPC in latent space.

        Args:
            bev: (B, C, H, W) current BEV observation
        Returns:
            dict with optimal actions and predicted future states
        """
        self.eval()
        z, _, _ = self.encode(bev)

        # MPC planning
        mpc_result = self.mpc(z, self.predict_future, self.action_encoder)

        # Decode planned trajectory to BEV
        B, T = mpc_result["trajectory_latents"].shape[:2]
        z_flat = rearrange(mpc_result["trajectory_latents"], "b t c h w -> (b t) c h w")
        bev_planned = self.decode(z_flat)
        bev_planned = rearrange(bev_planned, "(b t) c h w -> b t c h w", b=B)

        return {
            "planned_actions": mpc_result["actions"],
            "planned_bev_sequence": bev_planned,
            "planning_cost": mpc_result["best_cost"],
        }

    @torch.no_grad()
    def imagine(self, bev: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        'Dream' future scenes given current BEV and action sequence.

        This is the core world model capability — predicting what will happen
        if the ego takes a specific sequence of actions.

        Args:
            bev: (B, C, H, W) current scene
            actions: (B, T, action_dim) action sequence
        Returns:
            bev_sequence: (B, T, C, H, W) predicted future BEV frames
        """
        self.eval()
        z, _, _ = self.encode(bev)
        action_embeds = self.action_encoder(actions)
        z_future = self.predict_future(z, actions, action_embeds, actions.shape[1])

        B, T = z_future.shape[:2]
        z_flat = rearrange(z_future, "b t c h w -> (b t) c h w")
        bev_future = self.decode(z_flat)
        return rearrange(bev_future, "(b t) c h w -> b t c h w", b=B)
