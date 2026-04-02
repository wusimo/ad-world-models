"""
Train the driving world model on precomputed BEV features.

Phase 1: VAE reconstruction (encode BEV → latent → decode BEV)
Phase 2: Temporal prediction on consecutive frame pairs

Run precompute_bev.py first to generate the BEV cache.
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.world_model.model import DrivingWorldModel


class BEVCacheDataset(Dataset):
    """Load precomputed BEV features from disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.index = torch.load(self.cache_dir / "index.pt", weights_only=False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data = torch.load(self.cache_dir / f"{idx:04d}.pt", weights_only=False)
        return {
            "bev": data["bev"].float(),
            "future_trajectory": data["future_trajectory"].float(),
        }


def train_vae(model, dataloader, device, num_epochs=100, lr=1e-3):
    """Phase 1: Train VAE reconstruction."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n = 0

        for batch in dataloader:
            bev = batch["bev"].to(device)
            outputs = model(bev)

            loss = outputs["vae_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * bev.shape[0]
            total_recon += outputs["recon_loss"].item() * bev.shape[0]
            total_kl += outputs["kl_loss"].item() * bev.shape[0]
            n += bev.shape[0]

        scheduler.step()
        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_kl = total_kl / n

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  loss={avg_loss:.5f}  "
                  f"recon={avg_recon:.5f}  kl={avg_kl:.5f}  lr={scheduler.get_last_lr()[0]:.6f}")

    return best_loss


def train_temporal(model, dataloader, device, num_epochs=50, lr=5e-4):
    """Phase 2: Train temporal prediction on consecutive BEV pairs."""
    # Freeze VAE encoder/decoder, only train temporal transformer
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # Build consecutive pairs
    dataset = dataloader.dataset
    pairs = []
    for i in range(len(dataset) - 1):
        d0 = dataset[i]
        d1 = dataset[i + 1]
        # Use trajectory delta as action proxy
        traj = d0["future_trajectory"]
        action = torch.zeros(1, 3)  # steer, accel, yaw_rate
        if traj.shape[0] > 0:
            action[0, 0] = traj[0, 1]  # lateral as steer proxy
            action[0, 1] = traj[0, 0]  # forward as accel proxy
        pairs.append((d0["bev"], d1["bev"], action))

    print(f"  Training temporal on {len(pairs)} consecutive pairs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        indices = torch.randperm(len(pairs))
        for i in range(0, len(pairs), 4):
            batch_idx = indices[i:i+4]
            bev_t = torch.stack([pairs[j][0] for j in batch_idx]).to(device)
            bev_t1 = torch.stack([pairs[j][1] for j in batch_idx]).to(device)
            actions = torch.stack([pairs[j][2] for j in batch_idx]).to(device)

            outputs = model(bev_t, future_bevs=bev_t1.unsqueeze(1), actions=actions)

            loss = outputs.get("prediction_loss", outputs["vae_loss"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / max(1, len(pairs) // 4)
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  temporal_loss={avg:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--cache_dir", default="data/nuscenes/bev_cache")
    parser.add_argument("--output", default="outputs/world_model/trained.pt")
    parser.add_argument("--vae_epochs", type=int, default=100)
    parser.add_argument("--temporal_epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = BEVCacheDataset(args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"Loaded {len(dataset)} cached BEV features")

    # Model
    model = DrivingWorldModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"World Model: {total_params:.1f}M params")

    # Phase 1: VAE
    print("\n=== Phase 1: VAE Reconstruction ===")
    train_vae(model, dataloader, device, num_epochs=args.vae_epochs)

    # Phase 2: Temporal
    print("\n=== Phase 2: Temporal Prediction ===")
    train_temporal(model, dataloader, device, num_epochs=args.temporal_epochs)

    # Save
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved trained model to {args.output}")


if __name__ == "__main__":
    main()
