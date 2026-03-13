import os
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np

from rocket_hovering_env import EnvParams as HoveringEnvParams, EnvState as HoveringEnvState


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


colours = {
    'background': '#0a0a0a',
    'surface': '#e8e8e8',
    'target': '#00ff88',
    'trajectory': '#00bfff',
    'rocket': '#ffffff',
    'rocket_translucent': '#888888',
    'text': '#ffffff',
    'grid': '#333333',
}


def draw_rocket(ax, x, y, theta, height=10.0, width=2.0, color='white', alpha=1.0):
    half_h = height / 2
    half_w = width / 2
    nose_height = height * 0.15

    body_points = np.array([
        [-half_w, -half_h],
        [-half_w, half_h - nose_height],
        [0, half_h],
        [half_w, half_h - nose_height],
        [half_w, -half_h],
    ])

    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = body_points @ rotation.T
    translated = rotated + np.array([x, y])

    rocket = patches.Polygon(translated, closed=True, facecolor=color,
                             edgecolor='#333333', linewidth=0.5, alpha=alpha)
    ax.add_patch(rocket)

    fin_width = width * 0.5
    fin_height = height * 0.12
    for side in [-1, 1]:
        fin_points = np.array([
            [side * half_w, -half_h],
            [side * (half_w + fin_width), -half_h - fin_height],
            [side * half_w, -half_h + fin_height],
        ])
        rotated_fin = fin_points @ rotation.T + np.array([x, y])
        fin = patches.Polygon(rotated_fin, closed=True, facecolor='#666666', edgecolor='#333333', linewidth=0.3, alpha=alpha)
        ax.add_patch(fin)


def visualize_trajectory(
        states: List[HoveringEnvState],
        params: HoveringEnvParams,
        episode: int = 0,
        reward: float = 0.0,
        filename: str = "trajectory.png",
        show_rockets: bool = True,
        n_rockets: int = 8,
):
    fig = plt.figure(figsize=(13, 7), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.4, 1], hspace=0.4, wspace=0.25)

    fig.suptitle(f'Episode {episode}', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(gs[:, 0])

    xs = np.array([float(s.x) for s in states])
    ys = np.array([float(s.y) for s in states])
    thetas = np.array([float(s.theta) for s in states])
    throttles = np.array([float(s.throttle) for s in states])

    ax.set_facecolor(colours['background'])

    max_dist = getattr(params, 'max_distance', 200.0)
    plot_radius = max_dist * 0.5
    x_range = plot_radius * 1.1
    y_min = -plot_radius * 1.1
    y_max = plot_radius * 1.1

    ax.scatter(0, 0, s=200, c=colours['target'], marker='+', zorder=5, linewidths=1)
    ax.add_patch(plt.Circle((0, 0), 5, fill=False, color=colours['target'], linewidth=1, alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), max_dist, fill=False, color=colours['grid'],
                            linewidth=1, linestyle='--', alpha=0.3))

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=2, alpha=0.9)
    lc.set_array(throttles[:-1])
    ax.add_collection(lc)

    ax.scatter(xs[0], ys[0], s=80, c=colours['target'], marker='o', zorder=5,
               edgecolors='white', linewidth=1)

    if show_rockets and len(states) > 1:
        indices = np.linspace(0, len(states) - 1, n_rockets, dtype=int)
        for i, idx in enumerate(indices[:-1]):
            alpha = 0.15 + 0.4 * (i / len(indices))
            draw_rocket(ax, xs[idx], ys[idx], thetas[idx],
                        height=params.rocket_height, width=2.0,
                        color=colours['rocket_translucent'], alpha=alpha)

    if len(states) > 0:
        draw_rocket(ax, xs[-1], ys[-1], thetas[-1],
                    height=params.rocket_height, width=2.0,
                    color=colours['rocket'], alpha=1.0)

    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)', fontsize=10, color=colours['text'])
    ax.set_ylabel('Altitude (m)', fontsize=10, color=colours['text'])
    ax.tick_params(axis='both', colors=colours['text'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colours['grid'])

    ax.grid(True, alpha=0.2, color=colours['grid'], linestyle='-', linewidth=0.5)

    x_markers = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    for xm in x_markers:
        if -x_range < xm < x_range:
            ax.axvline(x=xm, color=colours['grid'], linestyle='--', alpha=0.3, linewidth=0.5)

    label_style = dict(
        transform=ax.transAxes, fontsize=10, color=colours['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', alpha=0.8, edgecolor='#333333')
    )

    ax.text(0.98, 0.98, 'Hovering', ha='right', va='top', **label_style)
    ax.text(0.02, 0.98, f'Reward: {reward:.1f}', ha='left', va='top', **label_style)

    times = np.arange(len(states)) * params.dt
    dxs = np.array([float(s.dx) for s in states])
    dys = np.array([float(s.dy) for s in states])
    omegas = np.array([float(s.omega) for s in states])
    left_thrusters = np.array([float(s.left_thruster) for s in states])
    right_thrusters = np.array([float(s.right_thruster) for s in states])

    plot_style = {'linewidth': 0.8, 'alpha': 0.9}

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(times, dxs, '#0066cc', label='Horizontal', **plot_style)
    ax1.plot(times, dys, '#cc3333', label='Vertical', **plot_style)
    ax1.set_ylabel('Vel (m/s)', fontsize=8)
    ax1.legend(loc='upper right', fontsize=6)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(times, np.degrees(thetas), '#009933', label='Angle', **plot_style)
    ax2.plot(times, np.degrees(omegas), '#cc6600', label='Angular vel', **plot_style)
    ax2.set_ylabel('Angle (°)', fontsize=8)
    ax2.legend(loc='upper right', fontsize=6)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(times, throttles, '#cc3333', label='Main', **plot_style)
    ax3.plot(times, left_thrusters, '#0066cc', label='Left', linewidth=0.6, alpha=0.7)
    ax3.plot(times, right_thrusters, '#cc6600', label='Right', linewidth=0.6, alpha=0.7)
    ax3.set_ylabel('Throttle', fontsize=8)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper right', fontsize=6)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=7)
    ax3.set_xlabel('Time (s)', fontsize=8)

    # fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(rewards, lengths, stage_iterations=None, save_dir=None, window=100):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    def smooth(data, window):
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i + 1]))
        return smoothed

    def shade_stages(ax, stage_iters):
        start = 0
        for i, n in enumerate(stage_iters):
            end = start + n
            if i % 2 == 0:
                ax.axvspan(start, end, color='#e0e0e0', alpha=0.5, zorder=0)
            start = end

    axes[0].plot(rewards, alpha=0.3, color='blue')
    axes[0].plot(smooth(rewards, window), color='blue', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lengths, alpha=0.3, color='green')
    axes[1].plot(smooth(lengths, window), color='green', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)

    if stage_iterations:
        for ax in axes:
            shade_stages(ax, stage_iterations)

    plt.tight_layout()

    target_dir = save_dir if save_dir else PLOTS_DIR
    save_path = os.path.join(target_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
