"""
Run Vista for a single scenario. Called by run_vista_scenarios.sh.

This runs as a separate process per scenario to avoid GPU OOM from accumulated memory.
"""

import argparse
import json
import os
import sys
import math
import torch
import numpy as np
from pathlib import Path


def generate_trajectory(action: str, num_points: int = 4, step_dist: float = 2.0):
    """Generate trajectory waypoints for Vista conditioning."""
    traj = []
    x, y = 0.0, 0.0
    heading = 0.0

    if action == "straight":
        turn_rate = 0.0
    elif action == "left":
        turn_rate = 0.08
    elif action == "right":
        turn_rate = -0.08
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
    parser.add_argument("--action", type=str, required=True,
                        choices=["free", "straight", "left", "right"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=5)
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Must run from Vista directory for imports
    vista_dir = Path(__file__).resolve().parent.parent.parent / "Vista"
    if not vista_dir.exists():
        # Try sibling directory
        vista_dir = Path(__file__).resolve().parent.parent / ".." / "Vista"
        vista_dir = vista_dir.resolve()
    assert vista_dir.exists(), f"Vista not found at {vista_dir}. Clone it alongside ad-world-models."
    os.chdir(str(vista_dir))
    sys.path.insert(0, str(vista_dir))

    import init_proj_path
    from sample_utils import (
        set_lowvram_mode, init_model, init_sampling,
        init_embedder_options, do_sample
    )
    from sample import VERSION2SPECS, DATASET2SOURCES, load_img, get_sample
    from pytorch_lightning import seed_everything
    from PIL import Image

    set_lowvram_mode(True)
    seed_everything(42)

    # Load model
    print(f"Loading Vista for action={args.action}...")
    version_dict = VERSION2SPECS["vwm"]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    # Load sample
    frame_list, _, _, _ = get_sample(args.sample_idx, "NUSCENES", 25, "free")
    cond_img = load_img(frame_list[0], 320, 576)

    img_seq = [load_img(fp, 320, 576) for fp in frame_list]
    images = torch.stack(img_seq)

    value_dict = init_embedder_options(unique_keys)
    value_dict["cond_frames_without_noise"] = cond_img[None]
    value_dict["cond_aug"] = 0.0
    value_dict["cond_frames"] = cond_img[None]

    # Add trajectory conditioning if not free
    if args.action != "free":
        traj = generate_trajectory(args.action, num_points=4)
        traj_tensor = torch.tensor(traj).flatten().float()  # (8,)
        value_dict["trajectory"] = traj_tensor

    sampler = init_sampling(
        guider="VanillaCFG", steps=args.n_steps,
        cfg_scale=2.5, num_frames=25,
    )

    uc_keys = ["cond_frames", "cond_frames_without_noise",
               "command", "trajectory", "speed", "angle", "goal"]

    out = do_sample(
        images, model, sampler, value_dict,
        num_rounds=1, num_frames=25,
        force_uc_zero_embeddings=uc_keys,
        initial_cond_indices=[0],
    )

    if isinstance(out, (tuple, list)):
        samples, _, _ = out

        # Save frames
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for i in range(samples.shape[0]):
            frame = samples[i].cpu()
            frame = torch.clamp((frame + 1.0) / 2.0, 0.0, 1.0)
            frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(img_dir / f"frame_{i:04d}.png")

        # Save grid
        from torchvision.utils import make_grid
        grid = make_grid(
            torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0),
            nrow=5, padding=2,
        )
        grid_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_img).save(output_dir / "grid.png")

        print(f"Saved {samples.shape[0]} frames to {output_dir}")


if __name__ == "__main__":
    main()
