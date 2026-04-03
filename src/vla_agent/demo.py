"""
Demo script for the DriveVLM-style VLA Agent.

Uses two components:
1. Qwen2.5-VL-3B (pretrained VLM) for Chain-of-Thought scene reasoning
2. Our trained trajectory decoder for waypoint planning

The VLM sees the actual camera image and produces real driving analysis.
The trajectory decoder uses BEV features to plan the ego trajectory.
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

# Chain-of-Thought prompts for driving scene analysis
COT_PROMPTS = {
    "scene_description": (
        "You are an autonomous driving AI. Describe this driving scene in 2 sentences: "
        "road type, lane layout, weather, and environment."
    ),
    "critical_objects": (
        "List the critical objects in this scene that affect driving "
        "(vehicles, pedestrians, obstacles, traffic signals). Be specific about their positions."
    ),
    "behavior_prediction": (
        "For the critical objects you identified, briefly predict what they will likely do next."
    ),
    "ego_decision": (
        "Based on the scene, what should the ego vehicle do? "
        "(maintain speed / accelerate / brake / turn / lane change). Give a 1-sentence decision."
    ),
}


def run_vlm_reasoning(img_path: str, model, processor, device: str = "cuda"):
    """Run Chain-of-Thought reasoning using Qwen2.5-VL on a driving image."""
    from qwen_vl_utils import process_vision_info

    reasoning = {}
    context = ""

    for stage, prompt in COT_PROMPTS.items():
        full_prompt = context + "\n\n" + prompt if context else prompt

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": full_prompt},
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)

        response = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        reasoning[stage] = response
        context += f"\n{stage}: {response}"

    return reasoning


def main():
    parser = argparse.ArgumentParser(description="DriveVLM VLA Agent Demo")
    parser.add_argument("--config", type=str, default="configs/vla_agent.yaml")
    parser.add_argument("--sample_idx", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/vla_agent")
    parser.add_argument("--weights", type=str, default="outputs/vla_agent/trained.pt")
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace VLM model ID for CoT reasoning")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  DriveVLM-style Vision-Language-Action Agent")
    print("  Qwen2.5-VL (pretrained) + Trained Trajectory Decoder")
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

    # Get front camera image path for VLM
    nusc_sample = dataset.samples[args.sample_idx]
    cam_data = dataset.nusc.get("sample_data", nusc_sample["data"]["CAM_FRONT"])
    front_cam_path = str(dataset.dataroot / cam_data["filename"])

    # Keep CPU copies
    gt_annotations = batch["annotations"][0]
    gt_trajectory = batch["future_trajectory"][0].numpy()
    camera_images = batch["images"][0].numpy()

    device = torch.device(args.device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.float().to(device)

    # Load pretrained VLM for Chain-of-Thought reasoning
    print(f"\n[2/5] Loading VLM: {args.vlm}...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vlm, torch_dtype=torch.bfloat16, device_map="auto",
        )
        vlm_processor = AutoProcessor.from_pretrained(args.vlm)
        vlm_available = True
        print(f"  VLM loaded ({sum(p.numel() for p in vlm.parameters())/1e9:.1f}B params)")
    except Exception as e:
        print(f"  Could not load VLM: {e}")
        print("  Falling back to GPT-2 placeholder")
        vlm_available = False

    # Load trajectory model
    print(f"\n[3/5] Loading trajectory decoder...")
    traj_model = DriveVLAAgent(config).to(device)
    weights_path = Path(args.weights)
    trained = False
    if weights_path.exists():
        traj_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        trained = True
        print(f"  Loaded trained weights from {weights_path}")
    else:
        print(f"  No trained weights at {weights_path}")

    # Run Chain-of-Thought reasoning with real VLM
    print(f"\n[4/5] Running Chain-of-Thought reasoning...")
    if vlm_available:
        reasoning = run_vlm_reasoning(front_cam_path, vlm, vlm_processor, str(device))
        vlm_name = args.vlm.split("/")[-1]

        # Free VLM memory for trajectory model
        del vlm
        torch.cuda.empty_cache()
    else:
        # Fallback to old GPT-2 path
        outputs = traj_model.predict(batch)
        reasoning = outputs["reasoning"]
        vlm_name = "GPT-2 (placeholder)"

    print("\n" + "=" * 60)
    print("  Chain-of-Thought Reasoning Output")
    print("=" * 60)
    for stage, response in reasoning.items():
        print(f"\n--- {stage.upper().replace('_', ' ')} ---")
        text = response if isinstance(response, str) else response[0]
        print(f"  {text[:300]}")

    # Get trajectory from trained decoder
    print(f"\n[5/5] Computing trajectory...")
    traj_model.eval()
    with torch.no_grad():
        bev = traj_model._extract_bev(batch["images"], batch["intrinsics"], batch["extrinsics"])
        visual_tokens = traj_model.visual_projector(bev)

        inputs = traj_model.tokenizer(
            "Plan a safe trajectory.", return_tensors="pt",
            padding=True, truncation=True, max_length=64,
        ).to(device)
        text_embeds = traj_model.language_model.get_input_embeddings()(inputs["input_ids"])
        B = visual_tokens.shape[0]
        combined = torch.cat([visual_tokens, text_embeds.expand(B, -1, -1)], dim=1)
        vis_mask = torch.ones(B, visual_tokens.shape[1], device=device)
        attn_mask = torch.cat([vis_mask, inputs["attention_mask"].expand(B, -1)], dim=1)

        lm_out = traj_model.language_model(
            inputs_embeds=combined, attention_mask=attn_mask, output_hidden_states=True,
        )
        last_hidden = lm_out.hidden_states[-1][:, -1, :]
        traj = traj_model.trajectory_decoder(last_hidden)[0].cpu().numpy()

    print(f"  Planned trajectory ({len(traj)} steps):")
    for t, (x, y) in enumerate(traj):
        print(f"    t+{t+1}: ({x:+.2f}, {y:+.2f}) m")

    # Visualize
    print("\nGenerating visualization...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis = BEVVisualizer()

    # Camera views
    vis.visualize_camera_views(
        camera_images, CAMERA_NAMES, IMG_MEAN, IMG_STD,
        title="Input: nuScenes 6-Camera Surround View",
        save_path=str(save_dir / "camera_views.png"),
    )

    # Multi-panel figure
    fig = plt.figure(figsize=(30, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1])

    # Panel 1: Front camera image
    ax1 = fig.add_subplot(gs[0, 0])
    img = camera_images[0].transpose(1, 2, 0)
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img, 0, 1)
    ax1.imshow(img)
    ax1.set_title("Input: CAM_FRONT", fontsize=13, color="white", fontweight="bold")
    ax1.axis("off")

    # Panel 2: Chain-of-Thought reasoning
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1e1e2e")
    ax2.axis("off")
    ax2.set_title(f"Chain-of-Thought Reasoning ({vlm_name})", fontsize=13,
                  color="white", fontweight="bold")

    stage_colors = {
        "scene_description": "#64b5f6",
        "critical_objects": "#ff8a65",
        "behavior_prediction": "#81c784",
        "ego_decision": "#e57373",
    }
    stage_icons = {
        "scene_description": "1. SCENE",
        "critical_objects": "2. CRITICAL OBJECTS",
        "behavior_prediction": "3. PREDICTIONS",
        "ego_decision": "4. DECISION",
    }

    y_pos = 0.95
    for stage, response in reasoning.items():
        text = response if isinstance(response, str) else response[0]
        text = text[:200].strip().replace("\n", " ")
        if not text:
            text = "(no output)"
        color = stage_colors.get(stage, "white")
        icon = stage_icons.get(stage, stage.upper())

        ax2.text(0.03, y_pos, icon, transform=ax2.transAxes,
                fontsize=10, color=color, fontweight="bold", va="top",
                fontfamily="monospace", path_effects=TEXT_OUTLINE)
        y_pos -= 0.04
        ax2.text(0.05, y_pos, fill(text, 55), transform=ax2.transAxes,
                fontsize=8, color="#cccccc", va="top", fontfamily="monospace",
                linespacing=1.4)
        y_pos -= 0.22

    # Arrow showing flow
    ax2.annotate("", xy=(0.01, 0.08), xytext=(0.01, 0.95),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=2),
                xycoords="axes fraction")

    # Panel 3: BEV trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    vis._setup_axes(ax3, "Planning: GT vs VLA Agent")

    if gt_annotations["num_objects"] > 0:
        vis.draw_gt_boxes(
            ax3, gt_annotations["centers"], gt_annotations["sizes"],
            gt_annotations["yaws"], gt_annotations["labels"], alpha=0.3,
        )

    vis.draw_trajectory(ax3, gt_trajectory, color="white", label="Ground Truth",
                       linewidth=3.0, marker="s", markersize=7, zorder=85)
    vis.draw_trajectory(ax3, traj, color="#00ff88",
                       label=f"VLA Agent ({'trained' if trained else 'untrained'})",
                       linewidth=2.5, markersize=6, zorder=95)

    # Auto-zoom
    all_pts = np.vstack([[0, 0], traj, gt_trajectory])
    pad = max(5.0, (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.25)
    ax3.set_ylim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax3.set_xlim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax3.legend(loc="upper right", fontsize=9, facecolor="#333", edgecolor="white",
              labelcolor="white", framealpha=0.8)

    status = "Pretrained VLM" if vlm_available else "Trained"
    fig.suptitle(f"DriveVLM — Vision-Language-Action Agent ({status})",
                fontsize=16, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(save_dir / "vla_agent_output.png"), dpi=150,
               bbox_inches="tight", facecolor="#1a1a1a")
    print(f"Saved to {save_dir}/vla_agent_output.png")

    # Save reasoning log
    with open(save_dir / "reasoning_log.txt", "w") as f:
        f.write(f"VLM: {vlm_name}\n")
        f.write(f"Image: {front_cam_path}\n\n")
        for stage, response in reasoning.items():
            f.write(f"=== {stage.upper()} ===\n{response}\n\n")
        f.write(f"=== TRAJECTORY ===\n")
        for t, (x, y) in enumerate(traj):
            f.write(f"t+{t+1}: ({x:+.2f}, {y:+.2f}) m\n")

    print(f"\n  Results saved to {save_dir}/")

    print("\n" + "=" * 60)
    print("  Architecture Summary")
    print("=" * 60)
    print(f"""
    Camera Image (1600x900)
        | Qwen2.5-VL-3B (pretrained VLM, {3}B params)
    Chain-of-Thought Reasoning:
      1. Scene Description
      2. Critical Object Identification
      3. Behavior Prediction
      4. Ego Decision

    Multi-Camera Images (6 x 224 x 400)
        | ResNet-50 + Lift-Splat-Shoot
    BEV Features (256 x 200 x 200)
        | Visual Projector + LM hidden state
    Trajectory Decoder -> 6 waypoints (x, y)
    """)


if __name__ == "__main__":
    main()
