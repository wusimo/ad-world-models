"""
Rich 3D demo for the VLA driving agent.

Shows:
    1. First-person 3D camera view (what the VLA "sees" through the windshield)
    2. Top-down BEV (the actual model input)
    3. Language command
    4. Action visualization (steering wheel + throttle/brake gauge)
    5. Action history over time
    6. Cumulative reward curve
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Wedge, Circle, Rectangle, FancyBboxPatch
from pathlib import Path
from PIL import Image as PILImage

from src.vla_sim.train_3d import VLA3DModel
from src.vla_sim.env_3d import make_3d_lang_env
from src.vla_sim.env import COMMANDS, COMMAND_LIST


TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


def make_3d_env(num_scenarios=20, map_type="SSS", traffic=0.15):
    """Create MetaDrive env with both 3D camera AND topdown observation."""
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    env = MetaDriveEnv(config={
        "use_render": False,
        "image_observation": True,
        "sensors": {"rgb_camera": (RGBCamera, 320, 180)},
        "vehicle_config": {"image_source": "rgb_camera"},
        "num_scenarios": num_scenarios,
        "map": map_type,
        "traffic_density": traffic,
        "norm_pixel": True,
        "stack_size": 1,
    })
    return env


def get_topdown_view(env, size=200):
    """Get top-down rendered view (matches what VLA was trained on)."""
    try:
        # Use TopDown observation rendering
        from metadrive.obs.top_down_obs import TopDownObservation
        # Render via env's render method
        frame = env.render(mode="topdown", screen_size=(size, size))
        if frame is not None:
            return frame
    except Exception:
        pass

    # Fallback: use the engine's draw method
    try:
        from metadrive.utils.draw_top_down_map import draw_top_down_map
        m = draw_top_down_map(env.current_map)
        return np.array(m)
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def draw_steering_wheel(ax, steering: float):
    """Draw a steering wheel showing the steer angle."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Wheel rim
    rim = Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=4)
    ax.add_patch(rim)

    # Inner hub
    hub = Circle((0, 0), 0.15, facecolor="#444", edgecolor="white", linewidth=1)
    ax.add_patch(hub)

    # Steering bar (rotated by steering angle)
    angle_rad = -steering * np.pi / 4  # max ±45 deg visual
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Horizontal bar
    bar_color = "#00ff88" if abs(steering) < 0.2 else ("#ff8a65" if abs(steering) < 0.5 else "#ff6b6b")
    x1, y1 = -0.85 * cos_a, -0.85 * sin_a
    x2, y2 = 0.85 * cos_a, 0.85 * sin_a
    ax.plot([x1, x2], [y1, y2], color=bar_color, linewidth=6)

    # Vertical (top) bar
    x3, y3 = 0.15 * sin_a, -0.15 * cos_a
    x4, y4 = -0.85 * sin_a, 0.85 * cos_a
    ax.plot([x3, x4], [y3, y4], color=bar_color, linewidth=6)

    ax.text(0, -1.35, f"steer={steering:+.2f}", fontsize=9, color="white",
           ha="center", fontweight="bold", path_effects=TEXT_OUTLINE)


def draw_throttle_gauge(ax, accel: float):
    """Draw vertical bar gauge for throttle/brake."""
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    # Outline
    rect = Rectangle((-0.3, -1), 0.6, 2, fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.axhline(0, color="white", linewidth=1, alpha=0.5, xmin=0.3, xmax=0.7)

    # Bar
    if accel >= 0:
        # Throttle (green, up from 0)
        bar = Rectangle((-0.25, 0), 0.5, accel, facecolor="#00ff88", alpha=0.85)
        label = f"throttle={accel:+.2f}"
    else:
        # Brake (red, down from 0)
        bar = Rectangle((-0.25, accel), 0.5, -accel, facecolor="#ff6b6b", alpha=0.85)
        label = f"brake={-accel:+.2f}"
    ax.add_patch(bar)

    ax.text(0, -1.35, label, fontsize=9, color="white", ha="center",
           fontweight="bold", path_effects=TEXT_OUTLINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="outputs/vla_sim/vla_3d_driving.pt")
    parser.add_argument("--save_dir", type=str, default="outputs/vla_sim")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VLA 3D Driving Demo")
    print("=" * 60)

    # Create 3D camera env (matches the training)
    print("\n[1] Creating 3D MetaDrive language env...")
    env = make_3d_lang_env(num_scenarios=20, map_type="SSS", traffic=0.15)
    obs, info = env.reset()
    print(f"  3D camera obs: {obs['image'].shape}")

    # Load VLA 3D model
    print("\n[2] Loading VLA 3D model...")
    model = VLA3DModel().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded from {args.weights}")

    # Run rollout
    print("\n[3] Running VLA episode...")
    cam_views, td_views, commands, actions, rewards = [], [], [], [], []

    for step in range(args.max_steps):
        # 3D first-person camera view (the model input)
        cam_img = obs["image"]  # already (H, W, C) uint8

        # Top-down view from same env (for visualization context)
        try:
            td_render = env.unwrapped.render(
                mode="topdown", screen_size=(300, 300),
                target_vehicle_heading_up=True,
            )
            td_img = td_render if td_render is not None else np.zeros((300, 300, 3), dtype=np.uint8)
            if td_img.dtype != np.uint8:
                td_img = (np.clip(td_img, 0, 255)).astype(np.uint8)
        except Exception:
            td_img = np.zeros((300, 300, 3), dtype=np.uint8)

        # Run VLA inference (3D camera + command → action)
        img_t = torch.from_numpy(cam_img).float().to(device) / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        cmd_t = torch.tensor([obs["command_id"]], device=device)
        with torch.no_grad():
            action, _ = model(img_t, cmd_t)
        action_np = action[0].cpu().numpy()

        cam_views.append(cam_img)
        td_views.append(td_img)
        commands.append(env.command_text)
        actions.append(action_np)

        # Step env
        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(reward)

        if terminated or truncated:
            break

    print(f"  Episode complete: {len(rewards)} steps, total reward = {sum(rewards):.1f}")

    env.close()

    # === Build the rich visualization ===
    print("\n[4] Building visualization...")

    # Select 6 key timesteps spread across the episode
    n_show = 6
    indices = np.linspace(0, len(cam_views) - 1, n_show, dtype=int)

    fig = plt.figure(figsize=(4.5 * n_show, 14), facecolor="#1a1a1a")
    gs = fig.add_gridspec(5, n_show, height_ratios=[1.4, 1, 0.4, 0.5, 0.6])

    for col, t in enumerate(indices):
        # Row 1: 3D first-person camera — THIS IS WHAT THE VLA SEES
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(cam_views[t])
        ax.axis("off")
        if col == 0:
            ax.set_title("3D Camera View\n(VLA model input)", fontsize=11,
                        color="#00ff88", fontweight="bold")
        else:
            ax.set_title(f"step {t}", fontsize=9, color="#888")
        for sp in ax.spines.values():
            sp.set_color("#00ff88")
            sp.set_linewidth(2)
            sp.set_visible(True)

        # Row 2: Top-down view (context — what's happening in the world)
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(td_views[t])
        ax.axis("off")
        if col == 0:
            ax.set_title("Top-Down Context\n(world state)", fontsize=11,
                        color="#64b5f6", fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color("#64b5f6")
            sp.set_linewidth(2)
            sp.set_visible(True)

        # Row 3: Language command
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        cmd_text = commands[t].split(".")[0] if commands[t] else ""
        ax.text(0.5, 0.5, f'"{cmd_text}"', transform=ax.transAxes,
               fontsize=10, color="#ffeb3b", ha="center", va="center",
               fontstyle="italic", path_effects=TEXT_OUTLINE)
        if col == 0:
            ax.text(-0.05, 0.5, "Language\nCommand", transform=ax.transAxes,
                   fontsize=10, color="#ffeb3b", fontweight="bold",
                   ha="right", va="center")

        # Row 4: Action gauges (steering wheel + throttle)
        steer, accel = actions[t]
        # Split row 4 into two halves: steering wheel and throttle gauge
        sub_gs = gs[3, col].subgridspec(1, 2)
        ax_steer = fig.add_subplot(sub_gs[0, 0])
        ax_throttle = fig.add_subplot(sub_gs[0, 1])
        ax_steer.set_facecolor("#1a1a1a")
        ax_throttle.set_facecolor("#1a1a1a")
        draw_steering_wheel(ax_steer, steer)
        draw_throttle_gauge(ax_throttle, accel)
        if col == 0:
            ax_steer.text(-0.4, 0.5, "VLA\nAction", transform=ax_steer.transAxes,
                         fontsize=10, color="white", fontweight="bold",
                         ha="right", va="center")

    # Row 5: Cumulative reward curve + action history
    ax_curves = fig.add_subplot(gs[4, :])
    ax_curves.set_facecolor("#2d2d2d")
    cum_r = np.cumsum(rewards)
    ax_curves.plot(cum_r, color="#00ff88", lw=2, label=f"Cumulative Reward (total={cum_r[-1]:.1f})")
    ax_curves.fill_between(range(len(cum_r)), cum_r, alpha=0.15, color="#00ff88")

    # Action history overlay (secondary axis)
    ax2 = ax_curves.twinx()
    steers = [a[0] for a in actions]
    accels = [a[1] for a in actions]
    ax2.plot(steers, color="#ff8a65", lw=1, alpha=0.7, label="Steering")
    ax2.plot(accels, color="#64b5f6", lw=1, alpha=0.7, label="Throttle")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Action Value", color="white")
    ax2.tick_params(colors="white")

    ax_curves.set_xlabel("Step", color="white")
    ax_curves.set_ylabel("Cumulative Reward", color="#00ff88")
    ax_curves.tick_params(axis="y", colors="#00ff88")
    ax_curves.tick_params(axis="x", colors="white")
    for sp in ax_curves.spines.values():
        sp.set_color("white")
    ax_curves.grid(True, alpha=0.2, color="white")

    # Combined legend
    lines1, labels1 = ax_curves.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_curves.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                    facecolor="#333", edgecolor="white", labelcolor="white", loc="upper left")

    fig.suptitle("VLA Driving Demo: 3D Camera + Language Command → Action",
                fontsize=16, fontweight="bold", color="white", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = save_dir / "vla_sim_3d_demo.png"
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
