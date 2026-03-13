import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_training_curves(rewards, lengths, stage_iterations=None, window=100):
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

    save_path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)




def plot_reward_structure():
    fig, ax = plt.subplots(figsize=(8, 5))

    alpha_d, alpha_v, alpha_theta, alpha_omega = 2.0, 1.0, 1.0, 1.0
    decay_d, decay_v, decay_theta, decay_omega = 5.0, 5.0, 0.5, 1.0

    d = np.linspace(0, 50, 500)
    v = np.linspace(0, 30, 500)
    theta = np.linspace(0, np.pi, 500)
    omega = np.linspace(0, 5, 500)

    ax.plot(d, alpha_d * np.exp(-d / decay_d), label=f'Distance (m)')
    ax.plot(v, alpha_v * np.exp(-v / decay_v), label=f'Velocity (m/s)')
    ax.plot(theta, alpha_theta * np.exp(-theta / decay_theta),  label=f'Angle (rad)')
    ax.plot(omega, alpha_omega * np.exp(-omega / decay_omega), label=f'Angular vel (rad/s)')

    ax.set_xlabel('State variable value', fontsize=10)
    ax.set_ylabel('Component-wise reward', fontsize=10)
    ax.set_title('Reward Structure', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 2.15)
    ax.set_xlim(0, 50)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'reward_structure.png')
    plt.savefig(save_path)
    plt.close(fig)


def plot_difficulty_coefficient():
    """c(p) = 1 / (1 + e^(-e^e * (p - 1/2)))"""
    c = lambda p: 1.0 / (1.0 + np.exp(-(np.e ** np.e) * (p - (1/2))))
    p = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots()
    ax.plot(p, c(p))

    ax.set_title("Difficulty Coefficient Scaling")
    ax.set_xlabel('Training progress $p$', fontsize=11)
    ax.set_ylabel('Difficulty coefficient $c(p)$', fontsize=11)

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'difficulty_coefficient.png')
    fig.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    plot_difficulty_coefficient()
    plot_reward_structure()