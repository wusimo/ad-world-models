"""
Generate video of VLA 3D agent driving.

Each frame shows:
    - 3D first-person camera view (large, what the VLA sees)
    - Top-down BEV inset (bottom-right corner)
    - Steering wheel + throttle gauge overlay (bottom-left)
    - Language command text (top)
    - Cumulative reward + step counter (top-right)
"""

import argparse
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.vla_sim.train_3d import VLA3DModel
from src.vla_sim.env_3d import make_3d_lang_env
from src.vla_sim.env import COMMANDS, COMMAND_LIST


def draw_steering_circle(draw: ImageDraw, center, radius, steer: float):
    """Draw a steering wheel indicator."""
    cx, cy = center
    # Outer rim
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline="white", width=3)
    # Inner hub
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill="#444", outline="white")

    # Steering bars rotated by angle
    angle = -steer * np.pi / 4  # ±45° max visual
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    r = radius - 8
    color = "#00ff88" if abs(steer) < 0.2 else ("#ffa726" if abs(steer) < 0.5 else "#ff5252")

    # Horizontal bar
    x1, y1 = cx - r * cos_a, cy + r * sin_a
    x2, y2 = cx + r * cos_a, cy - r * sin_a
    draw.line([(x1, y1), (x2, y2)], fill=color, width=4)

    # Top bar
    x3, y3 = cx + 8 * sin_a, cy + 8 * cos_a
    x4, y4 = cx + r * sin_a, cy + r * cos_a
    draw.line([(x3, y3), (x4, y4)], fill=color, width=4)


def draw_throttle_bar(draw: ImageDraw, x, y, width, height, accel: float):
    """Draw a vertical bar for throttle/brake."""
    # Background
    draw.rectangle([x, y, x + width, y + height], outline="white", width=2)
    # Mid line
    mid_y = y + height // 2
    draw.line([(x + 4, mid_y), (x + width - 4, mid_y)], fill="white", width=1)

    # Bar value
    bar_h = max(1, int(abs(accel) * (height // 2 - 4)))
    if accel >= 0.01:
        draw.rectangle([x + 4, mid_y - bar_h, x + width - 4, mid_y - 1],
                      fill="#00ff88")
    elif accel <= -0.01:
        draw.rectangle([x + 4, mid_y + 1, x + width - 4, mid_y + bar_h],
                      fill="#ff5252")


def composite_frame(
    cam_img: np.ndarray,
    td_img: np.ndarray,
    command_text: str,
    steer: float,
    accel: float,
    cum_reward: float,
    step: int,
    output_size=(960, 540),
):
    """Build a single composite video frame."""
    out_w, out_h = output_size

    # Resize camera view to fill the frame
    cam_pil = Image.fromarray(cam_img).resize((out_w, out_h), Image.LANCZOS)

    # Add a slight darkening overlay at top/bottom for text legibility
    overlay = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, 0, out_w, 60], fill=(0, 0, 0, 140))  # top bar
    odraw.rectangle([0, out_h - 130, out_w, out_h], fill=(0, 0, 0, 140))  # bottom bar
    cam_pil = Image.alpha_composite(cam_pil.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(cam_pil)

    # Try to load a nice font, fallback to default
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_lg = ImageFont.load_default()
        font_md = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    # Top: language command
    draw.text((20, 18), f'\u201C{command_text}\u201D',
             fill="#ffeb3b", font=font_md)

    # Top-right: step + reward
    info_text = f"step {step}   reward: {cum_reward:+.1f}"
    draw.text((out_w - 280, 20), info_text, fill="white", font=font_md)

    # Bottom-right: top-down inset
    inset_size = 110
    td_pil = Image.fromarray(td_img).resize((inset_size, inset_size), Image.LANCZOS)
    inset_x = out_w - inset_size - 15
    inset_y = out_h - inset_size - 15
    cam_pil.paste(td_pil, (inset_x, inset_y))
    # Inset border
    draw = ImageDraw.Draw(cam_pil)
    draw.rectangle([inset_x - 2, inset_y - 2, inset_x + inset_size + 1, inset_y + inset_size + 1],
                  outline="#64b5f6", width=2)
    draw.text((inset_x, inset_y - 18), "Top-Down BEV", fill="#64b5f6", font=font_sm)

    # Bottom-left: steering wheel + throttle
    wheel_cx, wheel_cy = 75, out_h - 60
    draw_steering_circle(draw, (wheel_cx, wheel_cy), 42, steer)
    draw.text((wheel_cx - 32, out_h - 18), f"steer={steer:+.2f}",
             fill="white", font=font_sm)

    throttle_x, throttle_y = 145, out_h - 105
    draw_throttle_bar(draw, throttle_x, throttle_y, 28, 90, accel)
    draw.text((throttle_x - 8, out_h - 18), f"accel={accel:+.2f}",
             fill="white", font=font_sm)

    # VLA label
    draw.text((205, out_h - 100), "VLA Agent",
             fill="#00ff88", font=font_md)
    draw.text((205, out_h - 75), "3D Camera + Language",
             fill="#aaa", font=font_sm)
    draw.text((205, out_h - 55), "→ Steer + Throttle",
             fill="#aaa", font=font_sm)

    return np.array(cam_pil)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/vla_sim/vla_3d_driving.pt")
    parser.add_argument("--save_dir", default="outputs/vla_sim")
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VLA 3D Driving Video Generator")
    print("=" * 60)

    # Env + model
    print("\n[1] Loading environment and model...")
    env = make_3d_lang_env(num_scenarios=20, map_type="SSS", traffic=0.15)
    model = VLA3DModel().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded VLA from {args.weights}")

    # Generate video
    video_path = save_dir / "vla_3d_drive.mp4"
    gif_path = save_dir / "vla_3d_drive.gif"

    print(f"\n[2] Recording {args.num_episodes} episodes ({args.max_steps} steps each)...")
    all_frames = []
    cum_reward = 0.0
    global_step = 0

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        for step in range(args.max_steps):
            cam_img = obs["image"]

            # Top-down render
            try:
                td = env.unwrapped.render(
                    mode="topdown", screen_size=(220, 220),
                    target_vehicle_heading_up=True,
                )
                td_img = td if td is not None else np.zeros((220, 220, 3), dtype=np.uint8)
                if td_img.dtype != np.uint8:
                    td_img = np.clip(td_img, 0, 255).astype(np.uint8)
            except Exception:
                td_img = np.zeros((220, 220, 3), dtype=np.uint8)

            # VLA action
            img_t = torch.from_numpy(cam_img).float().to(device) / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)
            cmd_t = torch.tensor([obs["command_id"]], device=device)
            with torch.no_grad():
                action, _ = model(img_t, cmd_t)
            action_np = action[0].cpu().numpy()
            steer, accel = float(action_np[0]), float(action_np[1])

            # Composite frame
            frame = composite_frame(
                cam_img=cam_img,
                td_img=td_img,
                command_text=env.command_text,
                steer=steer,
                accel=accel,
                cum_reward=cum_reward,
                step=global_step,
            )
            all_frames.append(frame)

            # Step
            obs, reward, term, trunc, info = env.step(action_np)
            ep_reward += reward
            cum_reward += reward
            global_step += 1

            if term or trunc:
                # Add a few "episode end" frames
                for _ in range(args.fps // 2):
                    all_frames.append(frame)
                break

        print(f"  Episode {ep+1}: {step+1} steps, reward = {ep_reward:.1f}")

    env.close()

    # Save as MP4
    print(f"\n[3] Writing video ({len(all_frames)} frames @ {args.fps} fps)...")
    print(f"  → {video_path}")
    with imageio.get_writer(str(video_path), fps=args.fps, codec="libx264",
                            quality=8, pixelformat="yuv420p") as writer:
        for f in all_frames:
            writer.append_data(f)

    # Save as GIF (lower fps to keep file small)
    print(f"  → {gif_path} (lower fps)")
    gif_fps = max(8, args.fps // 2)
    gif_frames = all_frames[::2]  # subsample
    imageio.mimsave(str(gif_path), gif_frames, fps=gif_fps, loop=0)

    # File sizes
    mp4_mb = video_path.stat().st_size / 1e6
    gif_mb = gif_path.stat().st_size / 1e6
    print(f"\n✓ Saved {len(all_frames)} frames")
    print(f"  MP4: {mp4_mb:.1f} MB ({video_path})")
    print(f"  GIF: {gif_mb:.1f} MB ({gif_path})")
    print(f"\nTotal cumulative reward: {cum_reward:+.1f}")


if __name__ == "__main__":
    main()
