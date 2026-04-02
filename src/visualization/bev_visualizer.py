"""
BEV (Bird's Eye View) visualization for autonomous driving.

Renders BEV feature maps, GT/predicted detection boxes, motion predictions,
and planned trajectories overlaid on a top-down view.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import patheffects
import torch


# nuScenes class colors (RGB)
CLASS_COLORS = {
    "car": "#1f77b4",
    "truck": "#2ca02c",
    "bus": "#d62728",
    "trailer": "#ff7f0e",
    "construction_vehicle": "#8c564b",
    "pedestrian": "#e377c2",
    "motorcycle": "#9467bd",
    "bicycle": "#bcbd22",
    "traffic_cone": "#ff9896",
    "barrier": "#7f7f7f",
}

CLASS_NAMES = list(CLASS_COLORS.keys())

# Outline effect for text readability
TEXT_OUTLINE = [patheffects.withStroke(linewidth=2, foreground="black")]


class BEVVisualizer:
    """Visualize autonomous driving outputs in BEV (top-down) view."""

    def __init__(
        self,
        bev_range: tuple = (-50, -50, 50, 50),
        figsize: tuple = (12, 12),
    ):
        self.bev_range = bev_range
        self.figsize = figsize

    def _setup_axes(self, ax: plt.Axes, title: str = ""):
        """Configure BEV plot axes with road-like background."""
        ax.set_facecolor("#2d2d2d")
        ax.set_xlim(self.bev_range[1], self.bev_range[3])
        ax.set_ylim(self.bev_range[0], self.bev_range[2])
        ax.set_aspect("equal")
        ax.set_xlabel("Lateral (m)", fontsize=11, color="white")
        ax.set_ylabel("Longitudinal (m)", fontsize=11, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        # Grid — road-like
        ax.grid(True, alpha=0.15, color="white", linewidth=0.5)

        # Draw dashed lane lines
        for offset in [-3.5, 0, 3.5]:
            ax.axvline(offset, color="white", linewidth=0.8, alpha=0.3, linestyle="--")

        # Draw ego vehicle as a detailed shape
        ego_body = patches.FancyBboxPatch(
            (-1.0, -2.3), 2.0, 4.6, boxstyle="round,pad=0.15",
            facecolor="#2196F3", edgecolor="white", alpha=0.9, linewidth=1.5, zorder=100,
        )
        ax.add_patch(ego_body)
        # Heading arrow
        ax.annotate("", xy=(0, 3.5), xytext=(0, 1.5),
                     arrowprops=dict(arrowstyle="->", color="white", lw=2), zorder=101)
        ax.text(0, -0.3, "EGO", color="white", fontsize=7, ha="center", va="center",
                fontweight="bold", zorder=102, path_effects=TEXT_OUTLINE)

        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)

    def draw_gt_boxes(
        self,
        ax: plt.Axes,
        centers: np.ndarray,
        sizes: np.ndarray,
        yaws: np.ndarray,
        labels: list[str],
        velocities: np.ndarray = None,
        alpha: float = 0.7,
    ):
        """
        Draw ground truth 3D boxes in BEV.

        In ego frame: x=forward, y=left. BEV plot: x-axis=lateral(y), y-axis=longitudinal(x).
        """
        drawn_labels = set()
        for i in range(len(centers)):
            x_ego, y_ego, z_ego = centers[i]
            # nuScenes size: (width, length, height)
            w, l, h = sizes[i]
            yaw = yaws[i]
            label = labels[i]

            color = CLASS_COLORS.get(label, "#aaaaaa")

            # BEV: plot y_ego on x-axis, x_ego on y-axis
            corners = np.array([
                [-w/2, -l/2], [w/2, -l/2], [w/2, l/2], [-w/2, l/2]
            ])
            # Rotate
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            corners = (R @ corners.T).T
            # Translate (swap x/y for BEV axes)
            corners[:, 0] += y_ego
            corners[:, 1] += x_ego

            poly = plt.Polygon(corners, closed=True, facecolor=color,
                              edgecolor="white", alpha=alpha, linewidth=1.0, zorder=50)
            ax.add_patch(poly)

            # Label (only first occurrence per class for legend clarity)
            show_label = label not in drawn_labels
            if show_label:
                drawn_labels.add(label)
                # Invisible plot for legend
                ax.plot([], [], "s", color=color, label=label, markersize=8)

            # Velocity arrow
            if velocities is not None and i < len(velocities):
                vx, vy = velocities[i]
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 0.5:
                    ax.arrow(y_ego, x_ego, vy * 1.5, vx * 1.5,
                            head_width=0.6, head_length=0.4,
                            fc=color, ec="white", linewidth=0.5, alpha=0.8, zorder=51)

    def draw_predicted_boxes(
        self,
        ax: plt.Axes,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        score_threshold: float = 0.3,
    ):
        """
        Draw model-predicted detection boxes (untrained model output).
        boxes: (N, 10) — cx, cy, cz, w, l, h, sin, cos, vx, vy
        """
        for i in range(len(boxes)):
            if scores[i] < score_threshold:
                continue
            cx, cy, cz, w, l, h = boxes[i, :6]
            yaw = np.arctan2(boxes[i, 6], boxes[i, 7])

            cls_idx = int(classes[i]) % len(CLASS_NAMES)
            color = list(CLASS_COLORS.values())[cls_idx]

            corners = np.array([[-w/2, -l/2], [w/2, -l/2], [w/2, l/2], [-w/2, l/2]])
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            corners = (R @ corners.T).T
            corners[:, 0] += cy
            corners[:, 1] += cx

            poly = plt.Polygon(corners, closed=True, facecolor="none",
                              edgecolor=color, alpha=0.6, linewidth=1.5,
                              linestyle="--", zorder=55)
            ax.add_patch(poly)

    def draw_motion_predictions(
        self,
        ax: plt.Axes,
        trajectories: np.ndarray,
        mode_probs: np.ndarray,
        origins: np.ndarray,
        top_k: int = 10,
    ):
        """
        Draw predicted motion trajectories.
        trajectories: (N, K, T, 2) in ego frame
        origins: (N, 2) starting positions (y_ego, x_ego in BEV)
        """
        for i in range(min(top_k, len(origins))):
            oy, ox = origins[i]
            for k in range(trajectories.shape[1]):
                prob = mode_probs[i, k]
                if prob < 0.05:
                    continue
                traj = trajectories[i, k]  # (T, 2)
                points_x = oy + traj[:, 1]
                points_y = ox + traj[:, 0]
                ax.plot(points_x, points_y, "-", alpha=min(1.0, prob * 3),
                       color="cyan", linewidth=1.5 * prob + 0.5, zorder=60)
                ax.plot(points_x[-1], points_y[-1], "o", color="cyan",
                       markersize=3, alpha=min(1.0, prob * 3), zorder=61)

    def draw_trajectory(
        self,
        ax: plt.Axes,
        trajectory: np.ndarray,
        color: str = "lime",
        label: str = "Planned",
        linewidth: float = 3.0,
        marker: str = "o",
        markersize: int = 8,
        zorder: int = 90,
        show_timesteps: bool = True,
    ):
        """
        Draw a trajectory in BEV.
        trajectory: (T, 2) — (x_forward, y_left) in ego frame
        """
        # Add origin
        full_traj = np.vstack([[0, 0], trajectory])
        # BEV: x-axis = lateral (y_ego), y-axis = longitudinal (x_ego)
        bev_x = full_traj[:, 1]
        bev_y = full_traj[:, 0]

        ax.plot(bev_x, bev_y, "-" + marker, color=color, linewidth=linewidth,
               markersize=markersize, label=label, zorder=zorder,
               markeredgecolor="white", markeredgewidth=1.0)

        if show_timesteps:
            for t in range(1, len(full_traj)):
                ax.text(bev_x[t], bev_y[t] + 1.2, f"t+{t}",
                       fontsize=7, color=color, ha="center", va="bottom",
                       fontweight="bold", zorder=zorder + 1,
                       path_effects=TEXT_OUTLINE)

    def draw_trajectory_with_collision(
        self,
        ax: plt.Axes,
        trajectory: np.ndarray,
        collision_scores: np.ndarray,
        label: str = "Planned (collision risk)",
        zorder: int = 90,
    ):
        """Draw trajectory colored by collision risk (green=safe, red=danger)."""
        full_traj = np.vstack([[0, 0], trajectory])
        bev_x = full_traj[:, 1]
        bev_y = full_traj[:, 0]

        points = np.column_stack([bev_x, bev_y])
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

        # Pad collision scores to match segments
        padded_scores = np.concatenate([[0], collision_scores])
        seg_scores = (padded_scores[:-1] + padded_scores[1:]) / 2

        cmap = plt.cm.RdYlGn_r
        colors = cmap(seg_scores)
        lc = LineCollection(segments, colors=colors, linewidths=4, zorder=zorder)
        ax.add_collection(lc)

        ax.plot(bev_x, bev_y, "o", color="white", markersize=7,
               markeredgecolor="gray", markeredgewidth=1, zorder=zorder + 1)
        ax.plot([], [], "-", color="orange", linewidth=3, label=label)

    def draw_bev_heatmap(self, ax: plt.Axes, bev: np.ndarray, alpha: float = 0.4):
        """Draw BEV feature map as background heatmap."""
        if bev.ndim == 3:
            bev = np.linalg.norm(bev, axis=0)  # L2 norm across channels
        # Normalize
        bev = (bev - bev.min()) / (bev.max() - bev.min() + 1e-8)
        extent = [self.bev_range[1], self.bev_range[3],
                  self.bev_range[0], self.bev_range[2]]
        ax.imshow(bev, extent=extent, origin="lower",
                 cmap="inferno", alpha=alpha, aspect="equal", zorder=1)

    def draw_lidar_bev(self, ax: plt.Axes, lidar_points: np.ndarray, alpha: float = 0.3):
        """Draw LiDAR point cloud as BEV scatter."""
        # lidar_points: (N, 5) — x, y, z, intensity, ring
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]
        # BEV: lateral=y, longitudinal=x
        mask = (np.abs(x) < 50) & (np.abs(y) < 50)
        ax.scatter(y[mask], x[mask], c=z[mask], cmap="coolwarm", s=0.3,
                  alpha=alpha, zorder=2, vmin=-2, vmax=2)

    def visualize_e2e(
        self,
        outputs: dict,
        gt_annotations: dict = None,
        gt_trajectory: np.ndarray = None,
        lidar_points: np.ndarray = None,
        bev_features: np.ndarray = None,
        trained: bool = False,
        title: str = "UniAD End-to-End Planner",
        save_path: str = None,
    ) -> plt.Figure:
        """Full visualization of E2E planner outputs with GT overlay."""
        fig = plt.figure(figsize=(30, 12), facecolor="#1a1a1a")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])

        # --- Panel 1: Scene Context (GT + LiDAR) ---
        ax1 = fig.add_subplot(gs[0, 0])
        self._setup_axes(ax1, "Scene: GT Annotations")

        if lidar_points is not None:
            self.draw_lidar_bev(ax1, lidar_points, alpha=0.4)

        if gt_annotations is not None and gt_annotations["num_objects"] > 0:
            self.draw_gt_boxes(
                ax1,
                gt_annotations["centers"],
                gt_annotations["sizes"],
                gt_annotations["yaws"],
                gt_annotations["labels"],
                gt_annotations["velocities"],
            )
        ax1.legend(loc="upper right", fontsize=8, facecolor="#333", edgecolor="white",
                  labelcolor="white", framealpha=0.8)

        # --- Panel 2: Detection + Motion Forecasting ---
        ax2 = fig.add_subplot(gs[0, 1])
        self._setup_axes(ax2, "Model: Detection & Motion")

        if bev_features is not None:
            self.draw_bev_heatmap(ax2, bev_features, alpha=0.3)

        # Show GT boxes faded for reference
        if gt_annotations is not None and gt_annotations["num_objects"] > 0:
            self.draw_gt_boxes(
                ax2,
                gt_annotations["centers"],
                gt_annotations["sizes"],
                gt_annotations["yaws"],
                gt_annotations["labels"],
                gt_annotations["velocities"],
                alpha=0.3,
            )

        # Predicted boxes (dashed)
        if "detection_boxes" in outputs:
            boxes = outputs["detection_boxes"][0].cpu().numpy()
            classes = outputs["detection_classes"][0].cpu().numpy()
            scores = outputs["detection_scores"][0].cpu().numpy()
            self.draw_predicted_boxes(ax2, boxes, classes, scores, score_threshold=0.25)

        # Motion predictions (relative to GT centers)
        if "motion_predictions" in outputs and gt_annotations is not None:
            motion = outputs["motion_predictions"][0].cpu().numpy()
            mode_probs = outputs["motion_mode_probs"][0].cpu().numpy()
            if gt_annotations["num_objects"] > 0:
                origins = gt_annotations["centers"][:, [1, 0]]  # swap for BEV
                self.draw_motion_predictions(ax2, motion, mode_probs, origins,
                                            top_k=min(10, gt_annotations["num_objects"]))

        ax2.legend(loc="upper right", fontsize=8, facecolor="#333", edgecolor="white",
                  labelcolor="white", framealpha=0.8)

        # --- Panel 3: Planning ---
        ax3 = fig.add_subplot(gs[0, 2])
        self._setup_axes(ax3, "Planning: GT vs Predicted")

        # GT boxes faded
        if gt_annotations is not None and gt_annotations["num_objects"] > 0:
            self.draw_gt_boxes(
                ax3,
                gt_annotations["centers"],
                gt_annotations["sizes"],
                gt_annotations["yaws"],
                gt_annotations["labels"],
                alpha=0.25,
            )

        traj = outputs["planned_trajectory"][0].cpu().numpy()
        col = outputs["collision_scores"][0].cpu().numpy()

        # GT trajectory
        if gt_trajectory is not None:
            self.draw_trajectory(ax3, gt_trajectory, color="white", label="Ground Truth",
                               linewidth=3.0, marker="s", markersize=7, zorder=85,
                               show_timesteps=True)

        # Predicted trajectory with collision coloring
        self.draw_trajectory_with_collision(ax3, traj, col)

        # Also draw predicted trajectory as solid line
        self.draw_trajectory(ax3, traj, color="#00ff88", label=f"Predicted ({'trained' if trained else 'untrained'})",
                           linewidth=2.5, markersize=6, zorder=95)

        # Auto-zoom: fit both GT and predicted trajectories with padding
        all_points = [np.array([[0, 0]]), traj]
        if gt_trajectory is not None:
            all_points.append(gt_trajectory)
        all_pts = np.vstack(all_points)
        x_range = all_pts[:, 0]
        y_range = all_pts[:, 1]
        pad = max(5.0, (x_range.max() - x_range.min()) * 0.25, (y_range.max() - y_range.min()) * 0.25)
        ax3.set_ylim(x_range.min() - pad, x_range.max() + pad)
        ax3.set_xlim(y_range.min() - pad, y_range.max() + pad)

        ax3.legend(loc="upper right", fontsize=9, facecolor="#333", edgecolor="white",
                  labelcolor="white", framealpha=0.8)

        fig.suptitle(title, fontsize=18, fontweight="bold", color="white", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            print(f"Saved to {save_path}")

        return fig

    def visualize_world_model(
        self,
        current_bev: np.ndarray,
        imagined_sequences: dict[str, np.ndarray],
        planned_actions: dict[str, np.ndarray] = None,
        title: str = "World Model — Imagined Futures",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Visualize world model: current BEV + imagined futures for multiple action scenarios.

        Args:
            current_bev: (C, H, W) current BEV features
            imagined_sequences: {"scenario_name": (T, C, H, W)} predicted futures
            planned_actions: {"scenario_name": (T, 3)} actions taken
        """
        scenarios = list(imagined_sequences.keys())
        n_scenarios = len(scenarios)
        T = imagined_sequences[scenarios[0]].shape[0]

        def to_scalar(bev):
            if bev.ndim == 3:
                return np.linalg.norm(bev, axis=0)
            return bev

        # Per-frame normalization so internal structure is always visible
        def normalize_local(bev):
            s = to_scalar(bev)
            vmin, vmax = np.percentile(s, [2, 98])
            return np.clip((s - vmin) / (vmax - vmin + 1e-8), 0, 1)

        current_disp = normalize_local(current_bev)

        scenario_colors = {"go_straight": "#00ff88", "turn_left": "#ff6b6b",
                          "turn_right": "#4ecdc4", "brake": "#ffe66d",
                          "Go Straight": "#00ff88", "Turn Left": "#ff6b6b",
                          "Turn Right": "#4ecdc4", "Brake": "#ffe66d"}

        # Layout: 2 rows per scenario (predicted BEV + difference map)
        n_rows = n_scenarios * 2
        fig, axes = plt.subplots(n_rows, T + 1,
                                figsize=(2.8 * (T + 1), 2.2 * n_rows),
                                facecolor="#1a1a1a")

        for row_idx, scenario in enumerate(scenarios):
            seq = imagined_sequences[scenario]
            color = scenario_colors.get(scenario, "white")
            bev_row = row_idx * 2
            diff_row = row_idx * 2 + 1

            # --- BEV row: column 0 = current, columns 1-T = imagined ---
            ax = axes[bev_row, 0]
            ax.imshow(current_disp, cmap="inferno", origin="lower", aspect="equal")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title("t=0", fontsize=10, color="white", fontweight="bold")
            # Scenario label
            label_text = scenario
            if planned_actions and scenario in planned_actions:
                act = planned_actions[scenario][0]
                label_text += f"\nsteer={act[0]:+.1f}\naccel={act[1]:+.1f}"
            ax.text(0.02, 0.98, label_text, transform=ax.transAxes, fontsize=8,
                   color=color, va="top", ha="left", fontweight="bold",
                   path_effects=TEXT_OUTLINE)
            for sp in ["left"]:
                ax.spines[sp].set_color(color)
                ax.spines[sp].set_linewidth(4)
                ax.spines[sp].set_visible(True)

            for t in range(T):
                ax = axes[bev_row, t + 1]
                frame_disp = normalize_local(seq[t])
                ax.imshow(frame_disp, cmap="inferno", origin="lower", aspect="equal")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"t+{t+1}", fontsize=10, color="white", fontweight="bold")
                for sp in ax.spines.values():
                    sp.set_color(color)
                    sp.set_linewidth(1.5)
                    sp.set_visible(True)

            # --- Difference row: shows what changed from current ---
            ax = axes[diff_row, 0]
            ax.axis("off")
            ax.text(0.5, 0.5, "diff", transform=ax.transAxes, fontsize=8,
                   color=color, ha="center", va="center", fontstyle="italic",
                   path_effects=TEXT_OUTLINE)

            for t in range(T):
                ax = axes[diff_row, t + 1]
                frame_disp = normalize_local(seq[t])
                diff = np.abs(frame_disp - current_disp)
                ax.imshow(diff, cmap="hot", origin="lower", aspect="equal", vmin=0, vmax=1)
                ax.axis("off")
                for sp in ax.spines.values():
                    sp.set_color(color)
                    sp.set_linewidth(1.0)
                    sp.set_visible(True)

        fig.suptitle(title, fontsize=15, fontweight="bold", color="white", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            print(f"Saved to {save_path}")

        return fig

    def visualize_mpc_planning(
        self,
        current_bev: np.ndarray,
        planned_bev_sequence: np.ndarray,
        planned_actions: np.ndarray,
        all_costs: np.ndarray = None,
        title: str = "World Model — MPC Planning",
        save_path: str = None,
    ) -> plt.Figure:
        """Visualize MPC planned sequence with action and cost annotations."""
        T = planned_bev_sequence.shape[0]

        fig, axes = plt.subplots(2, T + 1, figsize=(3.5 * (T + 1), 7),
                                facecolor="#1a1a1a",
                                gridspec_kw={"height_ratios": [3, 1]})

        def to_scalar(bev):
            if bev.ndim == 3:
                return np.linalg.norm(bev, axis=0)
            return bev

        def normalize_local(bev):
            s = to_scalar(bev)
            vmin, vmax = np.percentile(s, [2, 98])
            return np.clip((s - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Top row: BEV frames (per-frame normalization for visibility)
        current_disp = normalize_local(current_bev)
        axes[0, 0].imshow(current_disp, cmap="inferno", origin="lower", aspect="equal")
        axes[0, 0].set_title("t=0 (Now)", fontsize=11, color="white", fontweight="bold")
        axes[0, 0].axis("off")

        for t in range(T):
            frame_disp = normalize_local(planned_bev_sequence[t])
            axes[0, t + 1].imshow(frame_disp, cmap="inferno", origin="lower", aspect="equal")
            axes[0, t + 1].set_title(f"t+{t+1}", fontsize=11, color="#00ff88", fontweight="bold")
            axes[0, t + 1].axis("off")
            for spine in axes[0, t + 1].spines.values():
                spine.set_color("#00ff88")
                spine.set_linewidth(2)
                spine.set_visible(True)

        # Bottom row: action bar charts
        action_labels = ["Steer", "Accel", "Yaw"]
        colors = ["#ff6b6b", "#00ff88", "#4ecdc4"]

        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "Actions →", transform=axes[1, 0].transAxes,
                       fontsize=12, color="white", ha="center", va="center", fontweight="bold")

        for t in range(T):
            ax = axes[1, t + 1]
            ax.set_facecolor("#2d2d2d")
            acts = planned_actions[t]
            bars = ax.bar(action_labels, acts, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("white")
            for bar, val in zip(bars, acts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                       fontsize=7, color="white", fontweight="bold")

        fig.suptitle(title, fontsize=16, fontweight="bold", color="white", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            print(f"Saved to {save_path}")

        return fig

    def visualize_camera_views(
        self,
        images: np.ndarray,
        camera_names: list[str],
        img_mean: np.ndarray,
        img_std: np.ndarray,
        title: str = "Multi-Camera Surround View",
        save_path: str = None,
    ) -> plt.Figure:
        """Visualize all 6 camera views in a grid."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), facecolor="#1a1a1a")
        for i, (ax, cam) in enumerate(zip(axes.flat, camera_names)):
            img = images[i].transpose(1, 2, 0)  # CHW → HWC
            img = img * img_std + img_mean
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(cam, fontsize=11, color="white", fontweight="bold")
            ax.axis("off")

        fig.suptitle(title, fontsize=16, fontweight="bold", color="white")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            print(f"Saved to {save_path}")

        return fig
