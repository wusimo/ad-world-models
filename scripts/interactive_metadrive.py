"""
Interactive MetaDrive GUI demo.

Opens a 3D Panda3D window where you can either:
    1. Drive manually with keyboard (W/A/S/D)
    2. Watch a trained VLA agent drive
    3. Watch the IDM expert drive

Requirements:
    - A display server (X11/Wayland on Linux, native on Mac/Windows)
    - For SSH: enable X11 forwarding with `ssh -X` or use VNC
    - For headless servers: this won't work — use the video generator instead

Controls:
    W: accelerate
    S: brake
    A: steer left
    D: steer right
    R: reset episode
    ESC: quit
"""

import argparse
import os
import sys
import torch
import numpy as np

from src.vla_sim.train_3d import VLA3DModel
from src.vla_sim.env import COMMAND_LIST


def check_display():
    """Verify a display is available."""
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("ERROR: No display server detected.")
            print("  Set DISPLAY env var or run with X11 forwarding (ssh -X).")
            print("  For headless usage, use the video generator instead:")
            print("    python -m src.vla_sim.video")
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "vla", "idm"], default="vla",
                        help="Control mode: manual keyboard, trained VLA agent, or IDM expert")
    parser.add_argument("--weights", default="outputs/vla_sim/vla_3d_driving.pt",
                        help="Path to VLA weights (for --mode vla)")
    parser.add_argument("--map", default="SSS",
                        choices=["SSS", "SCrCSC", "CSRCSX", "O", "XOX", "3", "5"])
    parser.add_argument("--traffic", type=float, default=0.15)
    parser.add_argument("--num_scenarios", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not check_display():
        sys.exit(1)

    print("=" * 60)
    print("  Interactive MetaDrive GUI")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Map: {args.map}, Traffic: {args.traffic}")
    print("=" * 60)

    # Build env config
    from metadrive.envs.metadrive_env import MetaDriveEnv

    # Map type can be string or int
    map_val = args.map
    try:
        map_val = int(map_val)
    except ValueError:
        pass

    config = {
        "use_render": True,                  # GUI ON
        "manual_control": args.mode == "manual",
        "num_scenarios": args.num_scenarios,
        "map": map_val,
        "traffic_density": args.traffic,
        "window_size": (1280, 720),
        "show_interface": True,
        "show_fps": True,
    }

    # If using VLA, also enable RGB camera input
    if args.mode == "vla":
        from metadrive.component.sensors.rgb_camera import RGBCamera
        config.update({
            "image_observation": True,
            "sensors": {"rgb_camera": (RGBCamera, 320, 180)},
            "vehicle_config": {"image_source": "rgb_camera"},
            "norm_pixel": True,
            "stack_size": 1,
        })

    env = MetaDriveEnv(config=config)
    obs, info = env.reset()

    print("\nControls:")
    if args.mode == "manual":
        print("  W/A/S/D : drive")
        print("  R       : reset")
        print("  ESC     : quit")
    else:
        print("  R       : reset episode")
        print("  ESC     : quit")
    print()

    # Load VLA model if needed
    model = None
    if args.mode == "vla":
        from pathlib import Path
        if not Path(args.weights).exists():
            print(f"ERROR: VLA weights not found at {args.weights}")
            print("  Train first: python scripts/train_vla_3d.py")
            env.close()
            sys.exit(1)
        device = torch.device(args.device)
        model = VLA3DModel().to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded VLA from {args.weights}")

    # Run loop
    print("\nRunning... (close window to quit)")
    total_reward = 0.0
    step = 0
    try:
        while True:
            if args.mode == "manual":
                # Manual control: action comes from keyboard via env
                action = [0.0, 0.0]  # actual control is set internally
            elif args.mode == "vla":
                # VLA inference
                cam = obs["image"] if isinstance(obs, dict) else obs
                if cam.ndim == 4:
                    cam = cam[..., -1]
                cam_uint8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)

                # Determine command from navigation info
                if info.get("navigation_left"):
                    cmd_id = COMMAND_LIST.index("turn_left")
                elif info.get("navigation_right"):
                    cmd_id = COMMAND_LIST.index("turn_right")
                else:
                    cmd_id = COMMAND_LIST.index("go_forward")

                img_t = torch.from_numpy(cam_uint8).float().to(device) / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                cmd_t = torch.tensor([cmd_id], device=device)
                with torch.no_grad():
                    a, _ = model(img_t, cmd_t)
                action = a[0].cpu().numpy().tolist()
            else:  # IDM
                # Use a simple lane-keeping policy
                try:
                    vehicle = env.vehicle
                    lane = vehicle.lane
                    lateral = lane.local_coordinates(vehicle.position)[1]
                    heading_diff = vehicle.heading_theta - lane.heading_theta_at(
                        lane.local_coordinates(vehicle.position)[0]
                    )
                    steer = float(np.clip(-0.3 * lateral - 0.5 * heading_diff, -1, 1))
                    speed_err = (30.0 - vehicle.speed_km_h) / 10.0
                    accel = float(np.clip(speed_err, -0.5, 0.6))
                    action = [steer, accel]
                except Exception:
                    action = [0.0, 0.3]

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 50 == 0:
                print(f"  step {step:>5d}  reward={total_reward:+.1f}")

            if term or trunc:
                print(f"\n  Episode end at step {step}, total reward = {total_reward:+.1f}")
                obs, info = env.reset()
                total_reward = 0.0
                step = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        env.close()
        print("Closed.")


if __name__ == "__main__":
    main()
