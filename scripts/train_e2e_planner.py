"""
Train the UniAD-style E2E planner on nuScenes mini.

Primary loss: L1 regression on ego trajectory (6 future waypoints).
The gradient flows back through planning → motion → detection → BEV,
giving all heads useful signal even without per-task supervision.
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e2e_planner.model import UniADPlanner
from src.data.nuscenes_loader import NuScenesLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e2e_planner.yaml")
    parser.add_argument("--output", default="outputs/e2e_planner/trained.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_cfg = config["data"]
    dataset = NuScenesLoader(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split=data_cfg["split"],
        image_size=tuple(data_cfg["image_size"]),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=0, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Model
    model = UniADPlanner(config).to(device)

    # Freeze backbone (pretrained ResNet50)
    for param in model.backbone.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {total:.1f}M total, {trainable:.1f}M trainable (backbone frozen)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n=== Training E2E Planner ({args.epochs} epochs) ===")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n = 0

        for batch in dataloader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.float().to(device)

            outputs = model(batch)

            # Planning loss: L1 on trajectory
            pred_traj = outputs["planning"]["trajectory"]  # (B, 6, 2)
            gt_traj = batch["future_trajectory"]  # (B, 6, 2)
            plan_loss = F.l1_loss(pred_traj, gt_traj)

            # Collision regularization
            col_loss = outputs["planning"]["collision_scores"].mean()

            loss = plan_loss + 0.1 * col_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += plan_loss.item() * pred_traj.shape[0]
            n += pred_traj.shape[0]

        scheduler.step()
        avg_loss = total_loss / n

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Quick eval: check trajectory distance
            with torch.no_grad():
                model.eval()
                sample = dataset[0]
                b = dataset.collate_fn([sample])
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.float().to(device)
                out = model.predict(b)
                pred = out["planned_trajectory"][0].cpu().numpy()
                gt = b["future_trajectory"][0].cpu().numpy()
                import numpy as np
                ade = np.sqrt(((pred - gt)**2).sum(axis=1)).mean()

            print(f"  Epoch {epoch+1:3d}/{args.epochs}  L1={avg_loss:.4f}  "
                  f"ADE={ade:.2f}m  best={best_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\nBest planning L1: {best_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
