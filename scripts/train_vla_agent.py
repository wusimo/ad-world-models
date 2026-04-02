"""
Train the VLA agent's visual projector and trajectory decoder.

The LM backbone is frozen — we only train the visual projector
(BEV → LM token space) and trajectory decoder (LM hidden → waypoints).
Loss: L1 regression on ego trajectory.
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vla_agent.model import DriveVLAAgent
from src.data.nuscenes_loader import NuScenesLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vla_agent.yaml")
    parser.add_argument("--output", default="outputs/vla_agent/trained.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    # Model
    model = DriveVLAAgent(config).to(device)

    # Freeze everything except visual_projector and trajectory_decoder
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False
    for param in model.bev_transform.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"VLA Agent: trainable={trainable:.1f}M (visual_projector + trajectory_decoder)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n=== Training VLA Agent ({args.epochs} epochs) ===")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n = 0

        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.float().to(device)

            # Efficient training forward: skip CoT generation
            images = batch["images"]
            intrinsics = batch["intrinsics"]
            extrinsics = batch["extrinsics"]

            bev = model._extract_bev(images, intrinsics, extrinsics)
            visual_tokens = model.visual_projector(bev)

            # Single LM forward pass (no generation)
            prompt = "Analyze this driving scene and plan a safe trajectory."
            inputs = model.tokenizer(
                prompt, return_tensors="pt", padding=True,
                truncation=True, max_length=64,
            ).to(device)

            text_embeds = model.language_model.get_input_embeddings()(inputs["input_ids"])
            B = visual_tokens.shape[0]
            combined = torch.cat([visual_tokens, text_embeds.expand(B, -1, -1)], dim=1)

            vis_mask = torch.ones(B, visual_tokens.shape[1], device=device)
            attn_mask = torch.cat([vis_mask, inputs["attention_mask"].expand(B, -1)], dim=1)

            with torch.no_grad():
                lm_out = model.language_model(
                    inputs_embeds=combined,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                )

            last_hidden = lm_out.hidden_states[-1][:, -1, :]  # (B, lm_dim)
            trajectory = model.trajectory_decoder(last_hidden)  # (B, 6, 2)

            loss = F.l1_loss(trajectory, batch["future_trajectory"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * B
            n += B

        scheduler.step()
        avg_loss = total_loss / n

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  L1={avg_loss:.4f}  "
                  f"best={best_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\nBest trajectory L1: {best_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
