"""
Demo script for the UniAD-style End-to-End Planner.

Runs inference on nuScenes mini dataset and visualizes:
    - Ground truth 3D annotations in BEV (ego frame)
    - LiDAR point cloud BEV projection
    - Multi-modal motion predictions
    - Planned ego trajectory vs ground truth
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.e2e_planner.model import UniADPlanner
from src.data.nuscenes_loader import NuScenesLoader, CAMERA_NAMES, IMG_MEAN, IMG_STD
from src.visualization.bev_visualizer import BEVVisualizer


def main():
    parser = argparse.ArgumentParser(description="UniAD E2E Planner Demo")
    parser.add_argument("--config", type=str, default="configs/e2e_planner.yaml")
    parser.add_argument("--sample_idx", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/e2e_planner")
    parser.add_argument("--weights", type=str, default="outputs/e2e_planner/trained.pt",
                        help="Path to trained weights (skip if not found)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  UniAD-style End-to-End Planner Demo")
    print("  Architecture: Detection → Motion → Planning")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading nuScenes data...")
    data_cfg = config["data"]
    dataset = NuScenesLoader(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split=data_cfg["split"],
        image_size=tuple(data_cfg["image_size"]),
    )

    sample = dataset[args.sample_idx]
    batch = dataset.collate_fn([sample])

    # Keep CPU copies for visualization
    gt_annotations = batch["annotations"][0]
    gt_trajectory = batch["future_trajectory"][0].numpy()
    lidar_points = batch["lidar"][0].numpy()
    camera_images = batch["images"][0].numpy()

    # Move to device
    device = torch.device(args.device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.float().to(device)

    # Build model
    print("\n[2/4] Building UniAD planner...")
    model = UniADPlanner(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.1f}M")

    # Load trained weights
    weights_path = Path(args.weights)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"  Loaded trained weights from {weights_path}")
        trained = True
    else:
        print(f"  No trained weights found at {weights_path} — using random init")
        trained = False

    # Inference
    print("\n[3/4] Running inference...")
    with torch.no_grad():
        outputs = model.predict(batch)

    # Extract BEV features for visualization
    bev_features = outputs.get("bev_features")
    if bev_features is not None:
        bev_features = bev_features[0].cpu().numpy()

    traj = outputs["planned_trajectory"][0].cpu().numpy()
    col = outputs["collision_scores"][0].cpu().numpy()

    print(f"  GT annotations: {gt_annotations['num_objects']} objects")
    print(f"  GT trajectory total displacement: {np.linalg.norm(gt_trajectory[-1]):.1f}m")
    print(f"  Predicted trajectory ({len(traj)} steps):")
    for t, (x, y) in enumerate(traj):
        print(f"    t+{t+1}: ({x:+.2f}, {y:+.2f}) m")
    if not trained:
        print(f"  Note: Model is untrained — predictions are random. GT is shown for reference.")

    # Visualize
    print("\n[4/4] Generating visualizations...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = BEVVisualizer()

    # Camera views
    vis.visualize_camera_views(
        camera_images, CAMERA_NAMES, IMG_MEAN, IMG_STD,
        title="nuScenes 6-Camera Surround View",
        save_path=str(save_dir / "camera_views.png"),
    )

    # Full E2E visualization
    fig = vis.visualize_e2e(
        outputs,
        gt_annotations=gt_annotations,
        gt_trajectory=gt_trajectory,
        lidar_points=lidar_points,
        bev_features=bev_features,
        title=f"UniAD End-to-End Planner ({'Trained' if trained else 'Untrained'} — GT annotations shown)",
        save_path=str(save_dir / "e2e_planner_output.png"),
    )

    print(f"\n  Results saved to {save_dir}/")
    print(f"    - camera_views.png")
    print(f"    - e2e_planner_output.png")

    print("\n" + "=" * 60)
    print("  Architecture Summary")
    print("=" * 60)
    print("""
    Multi-Camera Images (6 views, 224x400)
        | ResNet-50 backbone
    Image Features (6 x 256 x 14 x 25)
        | Lift-Splat-Shoot BEV transform
    BEV Features (256 x 200 x 200)
        | DETR decoder (300 queries, 6 layers)
    3D Detections + Object Queries
        | Cross-attention motion forecaster
    Multi-Modal Motion Predictions (6 modes x 6 steps)
        | Collision-aware planning head
    Ego Trajectory (6 waypoints + collision scores)
    """)


if __name__ == "__main__":
    main()
