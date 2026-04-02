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
from pathlib import Path

from src.vla_agent.model import DriveVLAAgent
from src.data.nuscenes_loader import NuScenesLoader
from src.visualization.bev_visualizer import BEVVisualizer


def main():
    parser = argparse.ArgumentParser(description="DriveVLM VLA Agent Demo")
    parser.add_argument("--config", type=str, default="configs/vla_agent.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
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

    device = torch.device(args.device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.float().to(device)

    # Build model
    print("\n[2/4] Building VLA agent...")
    model = DriveVLAAgent(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total parameters: {total_params:.1f}M")
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
        print(f"  {text[:500]}")

    traj = outputs["trajectory"][0].cpu().numpy()
    print(f"\n--- PLANNED TRAJECTORY ---")
    for t, (x, y) in enumerate(traj):
        print(f"  t+{t+1}: ({x:.2f}, {y:.2f}) m")

    # Visualize
    print("\n[4/4] Generating visualization...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = BEVVisualizer()
    fig, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(1, 1, figsize=(10, 10))
    vis._setup_axes(ax)
    vis.draw_planned_trajectory(ax, traj, label="VLA Agent Plan")
    ax.legend()
    ax.set_title("DriveVLM — VLA Agent Planned Trajectory", fontsize=14)
    fig.savefig(str(save_dir / "vla_agent_output.png"), dpi=150, bbox_inches="tight")

    # Save reasoning log
    with open(save_dir / "reasoning_log.txt", "w") as f:
        explanation = model.explain(batch)
        f.write(explanation)

    print(f"\n✓ Done! Results saved to {save_dir}/")

    print("\n" + "=" * 60)
    print("  Architecture Summary")
    print("=" * 60)
    print("""
    Multi-Camera Images
        ↓ ResNet-50 backbone
    Image Features
        ↓ Lift-Splat-Shoot BEV transform
    BEV Features
        ↓ Visual Projector (BEV → LM token space)
    Visual Tokens (64 tokens)
        ↓ Prepended to text prompt
    [visual_tokens] + [CoT prompt]
        ↓ Language Model (GPT-2 / InternVL)
    Chain-of-Thought Reasoning:
      1. Scene Description
      2. Critical Object Identification
      3. Behavior Prediction
      4. Ego Decision
        ↓ Last hidden state
    Trajectory Decoder → 6 waypoints
    """)


if __name__ == "__main__":
    main()
