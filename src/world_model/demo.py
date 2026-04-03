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
import matplotlib.pyplot as plt
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
    parser.add_argument("--vista_dir", type=str, default="outputs/vista",
                        help="Path to Vista outputs (from scripts/demo_vista.py)")
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
    bev_mean, bev_std = 0.0, 1.0
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            bev_mean = float(ckpt.get("bev_mean", 0.0))
            bev_std = float(ckpt.get("bev_std", 1.0))
            print(f"  BEV normalization: mean={bev_mean:.5f}, std={bev_std:.5f}")
        else:
            model.load_state_dict(ckpt)
        print(f"  Loaded trained weights from {weights_path}")
        trained = True
    else:
        print(f"  No trained weights at {weights_path} — using random init")

    # Normalize BEV for the world model (same as training)
    bev_normalized = (bev - bev_mean) / bev_std

    # VAE reconstruction test
    print("\n[4/5] Testing VAE reconstruction + future imagination...")
    with torch.no_grad():
        z, mu, logvar = model.encode(bev_normalized)
        bev_recon_norm = model.decode(z)
        # Denormalize for comparison
        bev_recon = bev_recon_norm * bev_std + bev_mean
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
            bev_future_norm = model.imagine(bev_normalized, actions)
            # Denormalize for visualization
            bev_future = bev_future_norm * bev_std + bev_mean
        imagined[name] = bev_future[0].cpu().numpy()
        action_arrays[name] = actions[0].cpu().numpy()
        print(f"  {name}: imagined {bev_future.shape[1]} future frames")

    # MPC planning
    print("\n[5/5] Running MPC planning in latent space...")
    with torch.no_grad():
        plan_result = model.plan(bev_normalized)

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
    planned_bev = (plan_result["planned_bev_sequence"][0] * bev_std + bev_mean).cpu().numpy()
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

    # Vista pretrained world model comparison (if available)
    vista_dir = Path(args.vista_dir)
    vista_virtual = vista_dir / "virtual" / "grids"
    vista_real = vista_dir / "real" / "grids"
    if vista_virtual.exists() and vista_real.exists():
        import glob
        virtual_grids = sorted(glob.glob(str(vista_virtual / "*.png")))
        real_grids = sorted(glob.glob(str(vista_real / "*.png")))

        if virtual_grids and real_grids:
            from PIL import Image as PILImage
            print("\n  Adding Vista pretrained world model comparison...")

            virtual_img = np.array(PILImage.open(virtual_grids[0]))
            real_img = np.array(PILImage.open(real_grids[0]))

            # Load individual frames for filmstrip
            vista_virtual_imgs = sorted(glob.glob(str(vista_dir / "virtual" / "images" / "*_0000.png")))
            vista_frames = []
            for i in range(min(6, 25)):
                fp = vista_dir / "virtual" / "images" / f"NUSCENES_000000_{i:04d}.png"
                if fp.exists():
                    vista_frames.append(np.array(PILImage.open(fp)))

            # Create combined figure: BEV model (top) vs Vista pixel model (bottom)
            fig_combined = plt.figure(figsize=(28, 18), facecolor="#1a1a1a")
            gs = fig_combined.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])

            # Row 1: Camera input
            ax_cam = fig_combined.add_subplot(gs[0])
            # Show front camera
            front_img = camera_images[0].transpose(1, 2, 0)
            front_img = front_img * IMG_STD + IMG_MEAN
            front_img = np.clip(front_img, 0, 1)
            ax_cam.imshow(front_img)
            ax_cam.set_title("Input: nuScenes Front Camera", fontsize=14,
                           color="white", fontweight="bold")
            ax_cam.axis("off")

            # Row 2: BEV world model (our implementation)
            ax_bev = fig_combined.add_subplot(gs[1])
            # Show current BEV + first imagined scenario side by side
            bev_display = np.linalg.norm(current_bev_np, axis=0)
            bev_vmin, bev_vmax = np.percentile(bev_display, [2, 98])
            bev_display = np.clip((bev_display - bev_vmin) / (bev_vmax - bev_vmin + 1e-8), 0, 1)

            first_scenario = list(imagined.values())[0]
            n_show = min(6, first_scenario.shape[0])
            combined_bev = [bev_display]
            for t in range(n_show):
                frame = np.linalg.norm(first_scenario[t], axis=0)
                fmin, fmax = np.percentile(frame, [2, 98])
                combined_bev.append(np.clip((frame - fmin) / (fmax - fmin + 1e-8), 0, 1))
            combined_bev = np.concatenate(combined_bev, axis=1)

            ax_bev.imshow(combined_bev, cmap="inferno", origin="lower", aspect="auto")
            ax_bev.set_title("BEV World Model (Our Implementation) — Abstract Feature Space",
                           fontsize=14, color="#ff9800", fontweight="bold")
            ax_bev.axis("off")
            # Add labels
            w = bev_display.shape[1]
            for i, label in enumerate(["t=0"] + [f"t+{t+1}" for t in range(n_show)]):
                ax_bev.text(w * i + w // 2, 5, label, fontsize=9, color="white",
                           ha="center", va="top", fontweight="bold",
                           path_effects=[__import__('matplotlib').patheffects.withStroke(linewidth=2, foreground="black")])

            # Row 3: Vista world model (pretrained, pixel-space)
            ax_vista = fig_combined.add_subplot(gs[2])
            if vista_frames and len(vista_frames) >= 6:
                # Show filmstrip of vista frames
                step_indices = [0, 4, 9, 14, 19, 24]
                frames_to_show = []
                for si in step_indices:
                    fp = vista_dir / "virtual" / "images" / f"NUSCENES_000000_{si:04d}.png"
                    if fp.exists():
                        frames_to_show.append(np.array(PILImage.open(fp)))
                if frames_to_show:
                    # Resize all to same height
                    target_h = min(f.shape[0] for f in frames_to_show)
                    resized = []
                    for f in frames_to_show:
                        if f.shape[0] != target_h:
                            ratio = target_h / f.shape[0]
                            new_w = int(f.shape[1] * ratio)
                            f = np.array(PILImage.fromarray(f).resize((new_w, target_h)))
                        resized.append(f)
                    filmstrip = np.concatenate(resized, axis=1)
                    ax_vista.imshow(filmstrip)
            else:
                ax_vista.imshow(virtual_img)

            ax_vista.set_title("Vista World Model (Pretrained, NeurIPS 2024) — Photorealistic Pixel Space",
                             fontsize=14, color="#00ff88", fontweight="bold")
            ax_vista.axis("off")

            fig_combined.suptitle("World Model Comparison: BEV Feature Space vs Pixel Space",
                                fontsize=18, fontweight="bold", color="white", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig_combined.savefig(str(save_dir / "world_model_comparison.png"),
                               dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            print(f"Saved to {save_dir}/world_model_comparison.png")
    else:
        print("\n  Vista outputs not found. Run Vista demo first for pixel-space comparison:")
        print(f"    python scripts/demo_vista.py --vista_dir /path/to/Vista")

    print(f"\n  Results saved to {save_dir}/")
    print(f"    - camera_views.png")
    print(f"    - imagined_futures.png")
    print(f"    - mpc_planned.png")
    if (save_dir / "world_model_comparison.png").exists():
        print(f"    - world_model_comparison.png (BEV vs Vista)")

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
