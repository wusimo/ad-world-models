"""
Precompute BEV features for all nuScenes samples.

Uses frozen ResNet50 backbone + LSS BEV transform to extract
(256, 200, 200) BEV features and cache them to disk for fast
world model training.
"""

import argparse
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.data.nuscenes_loader import NuScenesLoader
from src.e2e_planner.model import ImageBackbone
from src.data.bev_transform import BEVTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--cache_dir", default="data/nuscenes/bev_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_cfg = config["data"]
    dataset = NuScenesLoader(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split=data_cfg["split"],
        image_size=tuple(data_cfg["image_size"]),
    )

    device = torch.device(args.device)
    bev_cfg = config["bev"]

    # Build frozen BEV extractor
    backbone = ImageBackbone("resnet50", pretrained=True, out_channels=256).to(device)
    backbone.eval()
    bev_transform = BEVTransform(
        in_channels=bev_cfg["in_channels"],
        bev_channels=bev_cfg["bev_channels"],
        bev_size=tuple(bev_cfg["bev_size"]),
        bev_range=tuple(bev_cfg["bev_range"]),
    ).to(device)
    bev_transform.eval()

    print(f"Precomputing BEV features for {len(dataset)} samples...")
    metadata = []

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        batch = dataset.collate_fn([sample])

        images = batch["images"].float().to(device)
        intrinsics = batch["intrinsics"].float().to(device)
        extrinsics = batch["extrinsics"].float().to(device)

        with torch.no_grad():
            features = backbone(images)
            bev = bev_transform(features, intrinsics, extrinsics)

        # Save BEV and metadata
        token = batch["tokens"][0]
        torch.save({
            "bev": bev[0].cpu().half(),  # save as fp16 to save disk
            "future_trajectory": batch["future_trajectory"][0],
            "token": token,
        }, cache_dir / f"{idx:04d}.pt")

        metadata.append({"idx": idx, "token": token})

    # Save index
    torch.save(metadata, cache_dir / "index.pt")
    print(f"Saved {len(metadata)} BEV features to {cache_dir}")


if __name__ == "__main__":
    main()
