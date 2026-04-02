"""
Demo script for the DriveVLM-style VLA Agent.

Runs Chain-of-Thought reasoning on a driving scene:
    1. Scene description
    2. Critical object identification
    3. Behavior prediction
    4. Ego decision making
    5. Trajectory planning
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path
from textwrap import fill

from src.vla_agent.model import DriveVLAAgent
from src.data.nuscenes_loader import NuScenesLoader, CAMERA_NAMES, IMG_MEAN, IMG_STD
from src.visualization.bev_visualizer import BEVVisualizer


TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def main():
    parser = argparse.ArgumentParser(description="DriveVLM VLA Agent Demo")
    parser.add_argument("--config", type=str, default="configs/vla_agent.yaml")
    parser.add_argument("--sample_idx", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/vla_agent")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  DriveVLM-style Vision-Language-Action Agent")
    print("  Chain-of-Thought Reasoning for Driving")
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

    # Keep CPU copies
    gt_annotations = batch["annotations"][0]
    gt_trajectory = batch["future_trajectory"][0].numpy()
    camera_images = batch["images"][0].numpy()

    device = torch.device(args.device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.float().to(device)

    # Build model
    print("\n[2/4] Building VLA agent...")
    model = DriveVLAAgent(config).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total parameters: {total:.1f}M")
    print(f"  Trainable parameters: {trainable:.1f}M (LM frozen)")

    # Run Chain-of-Thought reasoning
    print("\n[3/4] Running Chain-of-Thought reasoning...")
    outputs = model.predict(batch)

    print("\n" + "=" * 60)
    print("  Chain-of-Thought Reasoning Output")
    print("=" * 60)

    for stage, response in outputs["reasoning"].items():
        print(f"\n--- {stage.upper().replace('_', ' ')} ---")
        text = response if isinstance(response, str) else response[0]
        # Trim repetitive/garbage output from untrained GPT-2
        text = text[:200].strip()
        print(f"  {text}")

    traj = outputs["trajectory"][0].cpu().numpy()
    print(f"\n--- PLANNED TRAJECTORY ---")
    for t, (x, y) in enumerate(traj):
        print(f"  t+{t+1}: ({x:+.2f}, {y:+.2f}) m")

    print("\n  Note: GPT-2 is a placeholder LM (not a driving VLM).")
    print("  In production, use InternVL, LLaMA-Drive, or similar.")

    # Visualize
    print("\n[4/4] Generating visualization...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = BEVVisualizer()

    # Camera views
    vis.visualize_camera_views(
        camera_images, CAMERA_NAMES, IMG_MEAN, IMG_STD,
        title="Input: nuScenes 6-Camera Surround View",
        save_path=str(save_dir / "camera_views.png"),
    )

    # Multi-panel figure: [Camera front | CoT reasoning text | BEV trajectory]
    fig = plt.figure(figsize=(28, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])

    # Panel 1: Front camera image
    ax1 = fig.add_subplot(gs[0, 0])
    img = camera_images[0].transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img, 0, 1)
    ax1.imshow(img)
    ax1.set_title("Input: CAM_FRONT", fontsize=13, color="white", fontweight="bold")
    ax1.axis("off")

    # Panel 2: Chain-of-Thought reasoning display
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1e1e2e")
    ax2.axis("off")
    ax2.set_title("Chain-of-Thought Reasoning (GPT-2 placeholder)", fontsize=13,
                  color="white", fontweight="bold")

    stage_colors = {
        "scene_description": "#64b5f6",
        "critical_objects": "#ff8a65",
        "behavior_prediction": "#81c784",
        "ego_decision": "#e57373",
        "trajectory_plan": "#00ff88",
    }
    stage_icons = {
        "scene_description": "1. SCENE",
        "critical_objects": "2. CRITICAL OBJECTS",
        "behavior_prediction": "3. PREDICTIONS",
        "ego_decision": "4. DECISION",
        "trajectory_plan": "5. TRAJECTORY",
    }

    y_pos = 0.95
    for stage, response in outputs["reasoning"].items():
        text = response if isinstance(response, str) else response[0]
        text = text[:120].strip().replace("\n", " ")
        if not text:
            text = "(empty — untrained model)"
        color = stage_colors.get(stage, "white")
        icon = stage_icons.get(stage, stage.upper())

        ax2.text(0.03, y_pos, icon, transform=ax2.transAxes,
                fontsize=10, color=color, fontweight="bold", va="top",
                fontfamily="monospace", path_effects=TEXT_OUTLINE)
        y_pos -= 0.04
        ax2.text(0.05, y_pos, fill(text, 60), transform=ax2.transAxes,
                fontsize=8, color="#cccccc", va="top", fontfamily="monospace",
                linespacing=1.3)
        y_pos -= 0.14

    # Arrow showing flow
    ax2.annotate("", xy=(0.01, 0.08), xytext=(0.01, 0.95),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=2),
                xycoords="axes fraction")

    # Panel 3: BEV trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    vis._setup_axes(ax3, "Planning: GT vs VLA Agent")

    # GT boxes
    if gt_annotations["num_objects"] > 0:
        vis.draw_gt_boxes(
            ax3,
            gt_annotations["centers"],
            gt_annotations["sizes"],
            gt_annotations["yaws"],
            gt_annotations["labels"],
            alpha=0.3,
        )

    # GT trajectory
    vis.draw_trajectory(ax3, gt_trajectory, color="white", label="Ground Truth",
                       linewidth=3.0, marker="s", markersize=7, zorder=85)

    # VLA planned trajectory
    vis.draw_trajectory(ax3, traj, color="#00ff88", label="VLA Agent (untrained)",
                       linewidth=2.5, markersize=6, zorder=95)

    # Auto-zoom
    all_pts = np.vstack([[0, 0], traj, gt_trajectory])
    pad = max(5.0, (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.25)
    ax3.set_ylim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax3.set_xlim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax3.legend(loc="upper right", fontsize=9, facecolor="#333", edgecolor="white",
              labelcolor="white", framealpha=0.8)

    fig.suptitle("DriveVLM — Vision-Language-Action Agent (Untrained)",
                fontsize=16, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(save_dir / "vla_agent_output.png"), dpi=150,
               bbox_inches="tight", facecolor="#1a1a1a")
    print(f"Saved to {save_dir}/vla_agent_output.png")

    # Save reasoning log
    with open(save_dir / "reasoning_log.txt", "w") as f:
        explanation = model.explain(batch)
        f.write(explanation)

    print(f"\n  Results saved to {save_dir}/")

    print("\n" + "=" * 60)
    print("  Architecture Summary")
    print("=" * 60)
    print("""
    Multi-Camera Images
        | ResNet-50 backbone
    Image Features
        | Lift-Splat-Shoot BEV transform
    BEV Features
        | Visual Projector (BEV -> LM token space)
    Visual Tokens (64 tokens)
        | Prepended to text prompt
    [visual_tokens] + [CoT prompt]
        | Language Model (GPT-2 / InternVL)
    Chain-of-Thought Reasoning:
      1. Scene Description
      2. Critical Object Identification
      3. Behavior Prediction
      4. Ego Decision
        | Last hidden state
    Trajectory Decoder -> 6 waypoints
    """)


if __name__ == "__main__":
    main()
