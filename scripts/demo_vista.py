"""
Demo script for Vista — a pretrained pixel-space driving world model.

Generates photorealistic future driving video from a single input image,
then creates a side-by-side comparison with ground truth.

Requirements:
    - Vista repo cloned to ../Vista (or set --vista_dir)
    - Vista weights at ../Vista/ckpts/vista.safetensors
    - nuScenes mini dataset at ./data/nuscenes
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path
from PIL import Image

TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def main():
    parser = argparse.ArgumentParser(description="Vista World Model Demo")
    parser.add_argument("--vista_dir", type=str, default="../Vista",
                        help="Path to cloned Vista repo")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="outputs/vista")
    args = parser.parse_args()

    vista_dir = Path(args.vista_dir).resolve()
    save_dir = Path(args.save_dir)

    # Check if Vista outputs already exist
    virtual_grid = save_dir / "virtual" / "grids" / f"NUSCENES_{args.sample_idx:06d}.png"
    real_grid = save_dir / "real" / "grids" / f"NUSCENES_{args.sample_idx:06d}.png"

    if not virtual_grid.exists() or not real_grid.exists():
        print("Vista outputs not found. Running Vista inference...")
        print(f"Run from the Vista directory:")
        print(f"  cd {vista_dir}")
        print(f"  python sample.py --dataset NUSCENES --action free \\")
        print(f"    --n_frames 25 --n_rounds 1 --low_vram \\")
        print(f"    --height 320 --width 576 --n_steps 5 \\")
        print(f"    --save {save_dir.resolve()}")
        return

    # Load grid images
    virtual_img = np.array(Image.open(virtual_grid))
    real_img = np.array(Image.open(real_grid))

    # Also load individual frames for a filmstrip
    virtual_frames = []
    real_frames = []
    for i in range(25):
        vf = save_dir / "virtual" / "images" / f"NUSCENES_{args.sample_idx:06d}_{i:04d}.png"
        rf = save_dir / "real" / "images" / f"NUSCENES_{args.sample_idx:06d}_{i:04d}.png"
        if vf.exists():
            virtual_frames.append(np.array(Image.open(vf)))
        if rf.exists():
            real_frames.append(np.array(Image.open(rf)))

    print(f"Loaded {len(virtual_frames)} virtual frames, {len(real_frames)} real frames")

    # Create comparison visualization
    fig = plt.figure(figsize=(24, 16), facecolor="#1a1a1a")

    # Top: Real (GT) grid
    ax1 = fig.add_subplot(211)
    ax1.imshow(real_img)
    ax1.set_title("Ground Truth (Real nuScenes Frames)", fontsize=16,
                  color="white", fontweight="bold", pad=10)
    ax1.axis("off")

    # Bottom: Virtual (Vista predicted) grid
    ax2 = fig.add_subplot(212)
    ax2.imshow(virtual_img)
    ax2.set_title("Vista World Model — Imagined Future (Photorealistic)", fontsize=16,
                  color="#00ff88", fontweight="bold", pad=10)
    ax2.axis("off")

    fig.suptitle("Vista: Pretrained Driving World Model (NeurIPS 2024)\n"
                 "Input: Single front camera image → Output: 25-frame future video",
                 fontsize=18, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    comparison_path = save_dir / "vista_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"Saved comparison to {comparison_path}")

    # Filmstrip: selected frames at key timesteps
    if virtual_frames and real_frames:
        key_steps = [0, 4, 9, 14, 19, 24]
        key_steps = [k for k in key_steps if k < len(virtual_frames) and k < len(real_frames)]

        fig2, axes = plt.subplots(2, len(key_steps), figsize=(4 * len(key_steps), 6),
                                  facecolor="#1a1a1a")

        for col, t in enumerate(key_steps):
            # Real
            axes[0, col].imshow(real_frames[t])
            axes[0, col].axis("off")
            label = "Input (t=0)" if t == 0 else f"GT t+{t}"
            axes[0, col].set_title(label, fontsize=11, color="white", fontweight="bold")
            if col == 0:
                axes[0, col].text(-0.1, 0.5, "Real", transform=axes[0, col].transAxes,
                                 fontsize=13, color="white", fontweight="bold",
                                 rotation=90, va="center", ha="right")

            # Virtual
            axes[1, col].imshow(virtual_frames[t])
            axes[1, col].axis("off")
            label = "Input (t=0)" if t == 0 else f"Pred t+{t}"
            axes[1, col].set_title(label, fontsize=11, color="#00ff88", fontweight="bold")
            if col == 0:
                axes[1, col].text(-0.1, 0.5, "Vista", transform=axes[1, col].transAxes,
                                 fontsize=13, color="#00ff88", fontweight="bold",
                                 rotation=90, va="center", ha="right")

        fig2.suptitle("Vista: Frame-by-Frame Comparison (Real vs Imagined)",
                     fontsize=14, fontweight="bold", color="white")
        plt.tight_layout(rect=[0.02, 0, 1, 0.93])
        filmstrip_path = save_dir / "vista_filmstrip.png"
        fig2.savefig(filmstrip_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"Saved filmstrip to {filmstrip_path}")

    print("\nVista generates photorealistic future driving scenes from a single image.")
    print("This is a pixel-space world model (NeurIPS 2024) — fundamentally different")
    print("from our BEV-space world model which predicts abstract feature maps.")


if __name__ == "__main__":
    main()
