"""
Demo script for the Driving World Model (Vista/GenAD-style).

Demonstrates:
    1. BEV scene encoding to latent space (VAE)
    2. Future scene imagination given different action sequences
    3. MPC planning in latent space
    4. Decoded future BEV visualizations with action overlays
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.world_model.model import DrivingWorldModel
from src.data.nuscenes_loader import NuScenesLoader, CAMERA_NAMES, IMG_MEAN, IMG_STD
from src.data.bev_transform import BEVTransform
from src.visualization.bev_visualizer import BEVVisualizer
from src.e2e_planner.model import ImageBackbone


def main():
    parser = argparse.ArgumentParser(description="Driving World Model Demo")
    parser.add_argument("--config", type=str, default="configs/world_model.yaml")
    parser.add_argument("--sample_idx", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/world_model")
    parser.add_argument("--weights", type=str, default="outputs/world_model/trained.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  Driving World Model (Vista/GenAD-style)")
    print("  Imagine Future -> Plan in Imagination")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading nuScenes data...")
    data_cfg = config["data"]
    dataset = NuScenesLoader(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split=data_cfg["split"],
        image_size=tuple(data_cfg["image_size"]),
    )

    sample = dataset[args.sample_idx]
    batch = dataset.collate_fn([sample])

    device = torch.device(args.device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.float().to(device)

    # Build BEV extraction pipeline
    print("\n[2/5] Building BEV backbone...")
    bev_cfg = config["bev"]
    backbone = ImageBackbone("resnet50", pretrained=True, out_channels=256).to(device)
    bev_transform = BEVTransform(
        in_channels=bev_cfg["in_channels"],
        bev_channels=bev_cfg["bev_channels"],
        bev_size=tuple(bev_cfg["bev_size"]),
        bev_range=tuple(bev_cfg["bev_range"]),
    ).to(device)

    # Extract BEV features
    print("  Extracting BEV features from multi-camera images...")
    with torch.no_grad():
        img_features = backbone(batch["images"])
        bev = bev_transform(img_features, batch["intrinsics"], batch["extrinsics"])
    print(f"  BEV shape: {bev.shape}")

    # Keep CPU copy of current BEV for visualization
    current_bev_np = bev[0].cpu().numpy()

    # Build world model
    print("\n[3/5] Building world model...")
    model = DrivingWorldModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.1f}M")

    # Load trained weights
    weights_path = Path(args.weights)
    trained = False
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"  Loaded trained weights from {weights_path}")
        trained = True
    else:
        print(f"  No trained weights at {weights_path} — using random init")

    # VAE reconstruction test
    print("\n[4/5] Testing VAE reconstruction + future imagination...")
    with torch.no_grad():
        z, mu, logvar = model.encode(bev)
        bev_recon = model.decode(z)
        recon_error = (bev - bev_recon).abs().mean().item()
        print(f"  Reconstruction error (untrained): {recon_error:.4f}")
        print(f"  Latent shape: {z.shape}")

    # Imagine futures with different action sequences
    print("\n  Imagining futures with different driving actions...")
    action_scenarios = {
        "Go Straight": torch.tensor([[[0.0, 0.5, 0.0]]] * 6).permute(1, 0, 2).to(device),
        "Turn Left": torch.tensor([[[0.3, 0.3, 0.1]]] * 6).permute(1, 0, 2).to(device),
        "Turn Right": torch.tensor([[[-0.3, 0.3, -0.1]]] * 6).permute(1, 0, 2).to(device),
        "Brake": torch.tensor([[[0.0, -0.5, 0.0]]] * 6).permute(1, 0, 2).to(device),
    }

    imagined = {}
    action_arrays = {}
    for name, actions in action_scenarios.items():
        with torch.no_grad():
            bev_future = model.imagine(bev, actions)
        imagined[name] = bev_future[0].cpu().numpy()
        action_arrays[name] = actions[0].cpu().numpy()
        print(f"  {name}: imagined {bev_future.shape[1]} future frames")

    # MPC planning
    print("\n[5/5] Running MPC planning in latent space...")
    with torch.no_grad():
        plan_result = model.plan(bev)

    planned_actions = plan_result["planned_actions"][0].cpu().numpy()
    print(f"  Planning cost: {plan_result['planning_cost'][0]:.4f}")
    print(f"  Planned actions ({len(planned_actions)} steps):")
    for t, (steer, accel, yaw) in enumerate(planned_actions):
        print(f"    t+{t+1}: steer={steer:+.3f}, accel={accel:+.3f}, yaw_rate={yaw:+.3f}")

    # Visualize
    print("\nGenerating visualizations...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = BEVVisualizer()

    # Multi-scenario imagined futures
    vis.visualize_world_model(
        current_bev=current_bev_np,
        imagined_sequences=imagined,
        planned_actions=action_arrays,
        title=f"World Model: Imagined Futures Under Different Actions ({'Trained' if trained else 'Untrained'})",
        save_path=str(save_dir / "imagined_futures.png"),
    )

    # MPC planned sequence
    planned_bev = plan_result["planned_bev_sequence"][0].cpu().numpy()
    vis.visualize_mpc_planning(
        current_bev=current_bev_np,
        planned_bev_sequence=planned_bev,
        planned_actions=planned_actions,
        title=f"World Model: MPC-Planned Sequence ({'Trained' if trained else 'Untrained'})",
        save_path=str(save_dir / "mpc_planned.png"),
    )

    # Camera views
    camera_images = batch["images"][0].cpu().numpy()
    vis.visualize_camera_views(
        camera_images, CAMERA_NAMES, IMG_MEAN, IMG_STD,
        title="Input: nuScenes 6-Camera Surround View",
        save_path=str(save_dir / "camera_views.png"),
    )

    print(f"\n  Results saved to {save_dir}/")
    print(f"    - camera_views.png")
    print(f"    - imagined_futures.png")
    print(f"    - mpc_planned.png")

    print("\n" + "=" * 60)
    print("  Architecture Summary")
    print("=" * 60)
    print("""
    Current BEV Observation (256 x 200 x 200)
        | VAE Encoder (3x stride-2 conv)
    Latent State z (64 x 25 x 25)
        | + Action embeddings
    Temporal Transformer (4 layers, causal attention)
        | Autoregressive rollout
    Future Latent States z_t+1, z_t+2, ...
        | VAE Decoder (3x transposed conv)
    Predicted Future BEV Frames

    MPC Planning:
        Sample 64 action sequences
        -> Roll out each through world model
        -> Evaluate cost (collision + progress + comfort)
        -> Select lowest-cost trajectory
    """)


if __name__ == "__main__":
    main()
