"""
Trajectory visualization utilities.

Provides comparison plots for different planning approaches,
and overlays trajectories on camera images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TrajectoryVisualizer:
    """Compare trajectories from different planning paradigms."""

    def __init__(self, bev_range: tuple = (-50, -50, 50, 50)):
        self.bev_range = bev_range

    def compare_trajectories(
        self,
        trajectories: dict[str, np.ndarray],
        gt_trajectory: np.ndarray = None,
        title: str = "Planning Paradigm Comparison",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot multiple trajectories from different methods side by side.

        Args:
            trajectories: {"method_name": (T, 2) array} dict
            gt_trajectory: (T, 2) ground truth trajectory
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_xlim(self.bev_range[1], self.bev_range[3])
        ax.set_ylim(self.bev_range[0], self.bev_range[2])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Lateral (m)")
        ax.set_ylabel("Longitudinal (m)")

        # Ego vehicle
        ego = patches.FancyBboxPatch(
            (-1, -2.5), 2, 5, boxstyle="round,pad=0.2",
            facecolor="blue", edgecolor="white", alpha=0.8, linewidth=2,
        )
        ax.add_patch(ego)

        colors = ["lime", "red", "cyan", "magenta", "orange"]

        # Ground truth
        if gt_trajectory is not None:
            full_gt = np.vstack([[0, 0], gt_trajectory])
            ax.plot(full_gt[:, 1], full_gt[:, 0],
                   "--", color="white", linewidth=2, label="Ground Truth",
                   alpha=0.7, zorder=5)

        # Each method's trajectory
        for i, (name, traj) in enumerate(trajectories.items()):
            full_traj = np.vstack([[0, 0], traj])
            color = colors[i % len(colors)]
            ax.plot(full_traj[:, 1], full_traj[:, 0],
                   "-o", color=color, linewidth=2.5, markersize=5,
                   label=name, zorder=10 + i)

        ax.legend(loc="upper right", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_metrics(
        self,
        results: dict[str, dict],
        save_path: str = None,
    ) -> plt.Figure:
        """
        Bar chart comparing planning metrics across methods.

        Args:
            results: {"method": {"L2": float, "collision_rate": float, ...}}
        """
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[m][metric] for m in methods]
            bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
            ax.set_title(metric, fontsize=13, fontweight="bold")
            ax.set_ylabel("Value")

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        fig.suptitle("Planning Performance Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
