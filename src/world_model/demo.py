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
    camera_images = batch["images"][0].cpu().numpy()

    # Camera views
    vis.visualize_camera_views(
        camera_images, CAMERA_NAMES, IMG_MEAN, IMG_STD,
        title="Input: nuScenes 6-Camera Surround View",
        save_path=str(save_dir / "camera_views.png"),
    )

    # MPC planned sequence (BEV model)
    planned_bev = (plan_result["planned_bev_sequence"][0] * bev_std + bev_mean).cpu().numpy()
    vis.visualize_mpc_planning(
        current_bev=current_bev_np,
        planned_bev_sequence=planned_bev,
        planned_actions=planned_actions,
        title=f"World Model: MPC-Planned Sequence ({'Trained' if trained else 'Untrained'})",
        save_path=str(save_dir / "mpc_planned.png"),
    )

    # === Vista pretrained imagined futures (primary world model output) ===
    from PIL import Image as PILImage
    import glob
    import matplotlib.patheffects as pe

    vista_scenarios_dir = Path("outputs/vista_scenarios")
    scenario_map = {
        "Go Straight": "straight",
        "Turn Left": "left",
        "Turn Right": "right",
        "Free (No Action)": "free",
    }

    # Check if Vista scenario outputs exist
    has_vista_scenarios = all(
        (vista_scenarios_dir / d / "grid.png").exists()
        for d in ["straight", "left", "right"]
    )

    if has_vista_scenarios:
        print("\n  Building Vista imagined futures visualization...")

        # Key timesteps to show
        key_steps = [0, 4, 9, 14, 19, 24]
        scenarios_to_show = ["Go Straight", "Turn Left", "Turn Right"]

        n_rows = len(scenarios_to_show)
        n_cols = len(key_steps)

        fig, axes = plt.subplots(n_rows, n_cols + 1,
                                figsize=(3.5 * (n_cols + 1), 3.5 * n_rows),
                                facecolor="#1a1a1a")

        scenario_colors = {
            "Go Straight": "#00ff88",
            "Turn Left": "#ff6b6b",
            "Turn Right": "#4ecdc4",
        }

        for row, scenario_name in enumerate(scenarios_to_show):
            folder = scenario_map[scenario_name]
            color = scenario_colors[scenario_name]
            img_dir = vista_scenarios_dir / folder / "images"

            # Column 0: input frame + scenario label
            ax = axes[row, 0]
            frame0 = vista_scenarios_dir / folder / "images" / "frame_0000.png"
            if frame0.exists():
                ax.imshow(np.array(PILImage.open(frame0)))
            ax.axis("off")
            if row == 0:
                ax.set_title("Input (t=0)", fontsize=11, color="white", fontweight="bold")
            ax.text(0.02, 0.98, scenario_name, transform=ax.transAxes,
                   fontsize=11, color=color, fontweight="bold", va="top",
                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])

            # Columns 1-N: future frames
            for col, t in enumerate(key_steps):
                ax = axes[row, col + 1]
                frame_path = img_dir / f"frame_{t:04d}.png"
                if frame_path.exists():
                    ax.imshow(np.array(PILImage.open(frame_path)))
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"t+{t}" if t > 0 else "t=0",
                               fontsize=11, color="white", fontweight="bold")
                for sp in ax.spines.values():
                    sp.set_color(color)
                    sp.set_linewidth(2)
                    sp.set_visible(True)

        fig.suptitle("Vista World Model: Imagined Futures Under Different Actions (Pretrained)",
                    fontsize=16, fontweight="bold", color="white", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(str(save_dir / "imagined_futures.png"),
                   dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"  Saved to {save_dir}/imagined_futures.png")
    else:
        print("\n  Vista scenario outputs not found. Using BEV model for imagined futures.")
        print("  To generate Vista scenarios, run:")
        print("    for action in free straight left right; do")
        print("      python scripts/run_vista_single.py --action $action \\")
        print("        --output_dir outputs/vista_scenarios/$action --n_steps 5")
        print("    done")

        # Fallback: BEV model imagined futures
        vis.visualize_world_model(
            current_bev=current_bev_np,
            imagined_sequences=imagined,
            planned_actions=action_arrays,
            title=f"BEV World Model: Imagined Futures ({'Trained' if trained else 'Untrained'})",
            save_path=str(save_dir / "imagined_futures.png"),
        )

    # Vista comparison: BEV vs pixel-space world model
    if has_vista_scenarios:
        print("\n  Building world model comparison (BEV vs Vista)...")

        # Comparison figure: input camera → BEV features → Vista photorealistic
        fig_cmp = plt.figure(figsize=(28, 14), facecolor="#1a1a1a")
        gs = fig_cmp.add_gridspec(3, 7, height_ratios=[0.8, 1, 1])

        # Row 0: Input camera (centered, spanning middle columns)
        ax_cam = fig_cmp.add_subplot(gs[0, 2:5])
        front_img = camera_images[0].transpose(1, 2, 0)
        front_img = front_img * IMG_STD + IMG_MEAN
        front_img = np.clip(front_img, 0, 1)
        ax_cam.imshow(front_img)
        ax_cam.set_title("Input: nuScenes Front Camera", fontsize=13,
                        color="white", fontweight="bold")
        ax_cam.axis("off")
        # Hide unused axes in row 0
        for c in [0, 1, 5, 6]:
            fig_cmp.add_subplot(gs[0, c]).axis("off")

        # Row 1: BEV world model sequence
        bev_display = np.linalg.norm(current_bev_np, axis=0)
        bev_vmin, bev_vmax = np.percentile(bev_display, [2, 98])
        bev_display = np.clip((bev_display - bev_vmin) / (bev_vmax - bev_vmin + 1e-8), 0, 1)

        first_scenario = list(imagined.values())[0]
        bev_frames = [bev_display]
        for t in range(min(6, first_scenario.shape[0])):
            frame = np.linalg.norm(first_scenario[t], axis=0)
            fmin, fmax = np.percentile(frame, [2, 98])
            bev_frames.append(np.clip((frame - fmin) / (fmax - fmin + 1e-8), 0, 1))

        for col in range(min(7, len(bev_frames))):
            ax = fig_cmp.add_subplot(gs[1, col])
            ax.imshow(bev_frames[col], cmap="inferno", origin="lower")
            ax.axis("off")
            label = "t=0 (BEV)" if col == 0 else f"t+{col}"
            ax.set_title(label, fontsize=10, color="#ff9800", fontweight="bold")
        # Label for row
        fig_cmp.text(0.01, 0.5, "BEV Model\n(6.5M params)", fontsize=12,
                    color="#ff9800", fontweight="bold", va="center",
                    transform=fig_cmp.transFigure, rotation=90)

        # Row 2: Vista pixel-space sequence
        key_steps = [0, 4, 8, 12, 16, 20, 24]
        straight_dir = vista_scenarios_dir / "straight" / "images"
        for col, t in enumerate(key_steps[:7]):
            ax = fig_cmp.add_subplot(gs[2, col])
            fp = straight_dir / f"frame_{t:04d}.png"
            if fp.exists():
                ax.imshow(np.array(PILImage.open(fp)))
            ax.axis("off")
            label = "t=0 (Vista)" if t == 0 else f"t+{t}"
            ax.set_title(label, fontsize=10, color="#00ff88", fontweight="bold")
        fig_cmp.text(0.01, 0.18, "Vista Model\n(Pretrained)", fontsize=12,
                    color="#00ff88", fontweight="bold", va="center",
                    transform=fig_cmp.transFigure, rotation=90)

        fig_cmp.suptitle("World Model Comparison: BEV Feature Space vs Photorealistic Pixel Space",
                        fontsize=16, fontweight="bold", color="white", y=0.98)
        plt.tight_layout(rect=[0.03, 0, 1, 0.95])
        fig_cmp.savefig(str(save_dir / "world_model_comparison.png"),
                       dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"  Saved to {save_dir}/world_model_comparison.png")

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
