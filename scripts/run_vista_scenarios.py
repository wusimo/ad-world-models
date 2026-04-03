"""
Run Vista pretrained world model for multiple driving action scenarios.

Generates photorealistic future videos conditioned on different trajectories:
    - Go Straight: forward trajectory
    - Turn Left: leftward curved trajectory
    - Turn Right: rightward curved trajectory

This script must be run from the Vista repo directory.
Usage:
    cd /path/to/Vista
    python /path/to/ad-world-models/scripts/run_vista_scenarios.py
"""

import argparse
import json
import os
import sys
import math
import torch
import numpy as np
from pathlib import Path


def generate_trajectory(action: str, num_points: int = 23, step_dist: float = 1.5):
    """
    Generate a synthetic ego trajectory for Vista conditioning.

    Vista expects trajectory as (N, 2) tensor with N=num_features/2 future (x,y) waypoints
    in ego frame (x=right, y=forward in Vista's convention based on nuScenes camera frame).

    Returns: list of [x, y] points
    """
    traj = []
    x, y = 0.0, 0.0
    heading = 0.0  # radians, 0 = straight ahead

    if action == "straight":
        turn_rate = 0.0
    elif action == "left":
        turn_rate = 0.04  # radians per step
    elif action == "right":
        turn_rate = -0.04
    else:
        turn_rate = 0.0

    for i in range(num_points):
        heading += turn_rate
        x += step_dist * math.sin(heading)
        y += step_dist * math.cos(heading)
        traj.append([x, y])

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vista_dir", type=str, default=".",
                        help="Path to Vista repo root")
    parser.add_argument("--output_dir", type=str,
                        default="../ad-world-models/outputs/vista_scenarios")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=5,
                        help="Diffusion steps (lower=faster, higher=better quality)")
    args = parser.parse_args()

    vista_dir = Path(args.vista_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Add Vista to path
    sys.path.insert(0, str(vista_dir))

    # Check Vista is accessible
    assert (vista_dir / "sample.py").exists(), f"Vista not found at {vista_dir}"
    assert (vista_dir / "ckpts" / "vista.safetensors").exists(), "Vista weights not found"

    os.chdir(str(vista_dir))

    import init_proj_path
    from sample_utils import (
        set_lowvram_mode, init_model, init_sampling,
        init_embedder_options, do_sample, save_img_seq_to_video
    )
    from sample import VERSION2SPECS, DATASET2SOURCES, load_img, get_sample
    from pytorch_lightning import seed_everything
    from PIL import Image

    # Init model
    print("Loading Vista model...")
    set_lowvram_mode(True)
    version_dict = VERSION2SPECS["vwm"]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    # Load a nuScenes sample for the input frame
    print(f"Loading nuScenes sample {args.sample_idx}...")
    frame_list, _, _, _ = get_sample(args.sample_idx, "NUSCENES", 25, "free")

    # Load condition image
    cond_img = load_img(frame_list[0], 320, 576)

    # Scenarios to generate
    scenarios = {
        "go_straight": "straight",
        "turn_left": "left",
        "turn_right": "right",
    }

    for scenario_name, action in scenarios.items():
        print(f"\n{'='*50}")
        print(f"  Generating: {scenario_name}")
        print(f"{'='*50}")

        seed_everything(42)

        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Generate trajectory
        traj = generate_trajectory(action, num_points=23)

        value_dict = init_embedder_options(unique_keys)
        value_dict["cond_frames_without_noise"] = cond_img[None]
        value_dict["cond_aug"] = 0.0
        value_dict["cond_frames"] = cond_img[None]

        # Set trajectory (Vista expects first 4 waypoints flattened to 8 values)
        traj_tensor = torch.tensor(traj[:4]).flatten().float()  # (8,)
        value_dict["trajectory"] = traj_tensor

        sampler = init_sampling(
            guider="VanillaCFG",
            steps=args.n_steps,
            cfg_scale=2.5,
            num_frames=25,
        )

        uc_keys = ["cond_frames", "cond_frames_without_noise",
                    "command", "trajectory", "speed", "angle", "goal"]

        # Load all frames (needed by do_sample for shape)
        img_seq = [load_img(fp, 320, 576) for fp in frame_list]
        images = torch.stack(img_seq)

        try:
            out = do_sample(
                images,
                model,
                sampler,
                value_dict,
                num_rounds=1,
                num_frames=25,
                force_uc_zero_embeddings=uc_keys,
                initial_cond_indices=[0],
            )

            if isinstance(out, (tuple, list)):
                samples, samples_z, inputs = out

                # Save individual frames
                img_dir = scenario_dir / "images"
                img_dir.mkdir(exist_ok=True)

                for i in range(samples.shape[0]):
                    frame = samples[i].cpu()
                    frame = torch.clamp((frame + 1.0) / 2.0, 0.0, 1.0)
                    frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(frame).save(img_dir / f"frame_{i:04d}.png")

                # Save video
                save_img_seq_to_video(
                    samples, str(scenario_dir / f"{scenario_name}.mp4"), fps=10
                )

                # Save grid
                from torchvision.utils import make_grid
                grid = make_grid(
                    torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0),
                    nrow=5, padding=2
                )
                grid_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(grid_img).save(scenario_dir / "grid.png")

                print(f"  Saved {samples.shape[0]} frames to {scenario_dir}")

        except Exception as e:
            print(f"  Error generating {scenario_name}: {e}")
            # Save error info
            with open(scenario_dir / "error.txt", "w") as f:
                f.write(str(e))
            continue

    # Also generate "free" (no action conditioning) for comparison
    print(f"\n{'='*50}")
    print(f"  Generating: free (no action conditioning)")
    print(f"{'='*50}")

    seed_everything(42)
    free_dir = output_dir / "free"
    free_dir.mkdir(parents=True, exist_ok=True)

    value_dict = init_embedder_options(unique_keys)
    value_dict["cond_frames_without_noise"] = cond_img[None]
    value_dict["cond_aug"] = 0.0
    value_dict["cond_frames"] = cond_img[None]

    sampler = init_sampling(guider="VanillaCFG", steps=args.n_steps,
                           cfg_scale=2.5, num_frames=25)

    try:
        out = do_sample(
            images, model, sampler, value_dict,
            num_rounds=1, num_frames=25,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[0],
        )
        if isinstance(out, (tuple, list)):
            samples, _, _ = out
            img_dir = free_dir / "images"
            img_dir.mkdir(exist_ok=True)
            for i in range(samples.shape[0]):
                frame = samples[i].cpu()
                frame = torch.clamp((frame + 1.0) / 2.0, 0.0, 1.0)
                frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(frame).save(img_dir / f"frame_{i:04d}.png")

            from torchvision.utils import make_grid
            grid = make_grid(torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0), nrow=5, padding=2)
            grid_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(grid_img).save(free_dir / "grid.png")
            print(f"  Saved {samples.shape[0]} frames to {free_dir}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\nAll scenarios saved to {output_dir}")


if __name__ == "__main__":
    main()
