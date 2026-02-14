"""
Training I/O utilities: visualization and logging.
"""

import os
import csv
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import config


def save_wave_video_mp4(
    wave_output: Dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    epoch: int,
    output_dir: str = config.TRAIN_CONFIG["output_dir"],
    target_pos: Optional[tuple] = None,
    fps: int = config.VISUAL_CONFIG["video_fps"],
    codec: str = config.VISUAL_CONFIG["video_codec"],
    bitrate: Optional[str] = config.VISUAL_CONFIG["video_bitrate"],
    dpi: int = config.VISUAL_CONFIG["video_dpi"],
    frame_stride: int = config.VISUAL_CONFIG["video_frame_stride"],
    max_frames: Optional[int] = config.VISUAL_CONFIG["video_max_frames"],
):
    """
    Create MP4 video of the wave field with trajectory overlay.
    Optimized for performance and readability using an Agg canvas.
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        try:
            import imageio
        except ImportError:
            print("imageio not installed, skipping MP4 generation")
            return

    os.makedirs(output_dir, exist_ok=True)

    # Pre-fetch data to CPU
    stress_history = wave_output["stress_history"].detach().cpu().numpy()
    trajectory_np = trajectory.detach().cpu().numpy()
    stress_history = np.transpose(stress_history, (0, 2, 1))
    n_steps = stress_history.shape[0]

    # Build frame indices with optional stride/limit
    frame_indices = np.arange(0, n_steps, max(1, frame_stride))
    if max_frames is not None and len(frame_indices) > max_frames:
        frame_indices = np.linspace(0, n_steps - 1, max_frames).astype(int)

    # Expand trajectory to cover all frames
    traj_len = trajectory_np.shape[0]
    repeats = int(np.ceil(n_steps / traj_len))
    trajectory_full = np.tile(trajectory_np, (repeats, 1))[:n_steps]
    trajectory_cm = trajectory_full * config.VISUAL_CONFIG["cm_scale"]

    # Constants
    Lx, Ly = config.PHYSICS_CONFIG["Lx"], config.PHYSICS_CONFIG["Ly"]
    data_min = float(np.min(stress_history))
    data_max = float(np.max(stress_history))
    if data_min == data_max:
        data_max = data_min + 1e-12

    # Setup Figure (Create once, reuse for all frames)
    fig, ax = plt.subplots(figsize=config.VISUAL_CONFIG["wave_figsize"], dpi=dpi)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    canvas = FigureCanvas(fig)

    # Initial Image Plot (Step 0)
    im = ax.imshow(
        stress_history[0],
        origin="lower",
        extent=[0, Lx * config.VISUAL_CONFIG["cm_scale"], 0, Ly * config.VISUAL_CONFIG["cm_scale"]],
        cmap="RdBu_r",
        vmin=data_min,
        vmax=data_max,
        interpolation="nearest",
    )

    # Initial Plot Elements (Empty data initially)
    traj_line, = ax.plot([], [], "k-", linewidth=1, alpha=0.5, label="Trajectory")
    focus_point, = ax.plot([], [], "go", markersize=8, label="Current focus")

    # Target Point (Static, draw once)
    if target_pos:
        ax.plot(
            target_pos[0] * config.VISUAL_CONFIG["cm_scale"],
            target_pos[1] * config.VISUAL_CONFIG["cm_scale"],
            "r*",
            markersize=15,
            label="Target",
        )

    # Static Formatting
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, Lx * config.VISUAL_CONFIG["cm_scale"])
    ax.set_ylim(0, Ly * config.VISUAL_CONFIG["cm_scale"])
    fig.colorbar(im, ax=ax, label="Shear Stress (Pa)")

    video_path = os.path.join(output_dir, f"wave_epoch_{epoch:04d}.mp4")
    writer_kwargs = {
        "fps": fps,
        "codec": codec,
        "format": "ffmpeg",
    }
    if bitrate:
        writer_kwargs["bitrate"] = bitrate

    with imageio.get_writer(video_path, **writer_kwargs) as writer:
        for t in frame_indices:
            im.set_data(stress_history[t])
            traj_line.set_data(trajectory_cm[:t + 1, 0], trajectory_cm[:t + 1, 1])
            focus_point.set_data([trajectory_cm[t, 0]], [trajectory_cm[t, 1]])
            ax.set_title(f"Shear Stress Field - Step {t + 1}/{n_steps}")
            canvas.draw()

            frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            writer.append_data(frame)

    plt.close(fig)
    print(f"Saved: {video_path}")


def save_trajectory_plot(
    trajectory: torch.Tensor,
    epoch: int,
    output_dir: str = config.TRAIN_CONFIG["output_dir"],
    target_pos: Optional[tuple] = None,
    loss_dict: Optional[Dict[str, float]] = None,
):
    """
    Save a static plot of the optimized trajectory.

    Args:
        trajectory: [n_steps, 2] tensor
        epoch: Current epoch number
        output_dir: Directory to save plots
        target_pos: Optional (x, y) target position
        loss_dict: Optional dict of loss values to display
    """
    os.makedirs(output_dir, exist_ok=True)

    traj_np = trajectory.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=config.VISUAL_CONFIG["traj_figsize"])

    # Plot trajectory with color gradient (time)
    points = traj_np.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    norm = plt.Normalize(0, len(traj_np))
    lc = LineCollection(segments * config.VISUAL_CONFIG["cm_scale"], cmap='viridis', norm=norm)
    lc.set_array(np.arange(len(traj_np)))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Mark start and end
    ax.plot(traj_np[0, 0] * config.VISUAL_CONFIG["cm_scale"], traj_np[0, 1] * config.VISUAL_CONFIG["cm_scale"], 'b^', markersize=10, label='Start')
    ax.plot(traj_np[-1, 0] * config.VISUAL_CONFIG["cm_scale"], traj_np[-1, 1] * config.VISUAL_CONFIG["cm_scale"], 'rv', markersize=10, label='End')

    # Mark target
    if target_pos:
        ax.plot(
            target_pos[0] * config.VISUAL_CONFIG["cm_scale"],
            target_pos[1] * config.VISUAL_CONFIG["cm_scale"],
            'r*',
            markersize=20,
            label='Target'
        )

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Optimized Trajectory (Epoch {epoch})')

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time step')

    ax.legend(loc='upper right')
    ax.set_xlim(0, config.VISUAL_CONFIG["plot_xlim_cm"])
    ax.set_ylim(0, config.VISUAL_CONFIG["plot_ylim_cm"])
    ax.set_aspect('equal')

    # Add loss info as text
    if loss_dict:
        text = '\n'.join([f'{k}: {v:.6f}' for k, v in loss_dict.items()])
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'trajectory_epoch_{epoch:04d}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_training_history(history: Dict[str, list], output_dir: str):
    """Plot training loss history."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    axes[0, 0].plot(history['total'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')

    axes[0, 1].plot(history['contrast'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Contrast Loss')
    axes[0, 1].set_title('Spatiotemporal Contrast')

    axes[0, 2].plot(history['signal'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Signal')
    axes[0, 2].set_title('Target Signal')

    axes[0, 3].plot(history['noise'])
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('Noise')
    axes[0, 3].set_title('Off-Target Peak Noise')

    axes[1, 0].plot(history['snr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SNR')
    axes[1, 0].set_title('Signal-to-Noise Ratio')

    axes[1, 1].plot(history['trajectory_length'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Trajectory Length')
    axes[1, 1].set_title('Trajectory Length')

    axes[1, 2].plot(history['jerk'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Jerk')
    axes[1, 2].set_title('Trajectory Jerk')

    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/training_history.png")


def setup_training_logger(output_dir: str) -> logging.Logger:
    """Setup file logger for training run (overwrite each run)."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.log")
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_training_logger(logger: logging.Logger) -> None:
    """Close and remove all handlers on the training logger."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def create_csv_logger(
    output_dir: str,
    loss_dict: Dict[str, float],
) -> Tuple[object, csv.DictWriter, list]:
    """Create CSV log file and write header."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "training_log.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch",
        "lr",
        "target_pos_x",
        "target_pos_y",
    ] + list(loss_dict.keys())
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    return csv_file, csv_writer, fieldnames


def write_csv_row(
    csv_writer: csv.DictWriter,
    csv_file: object,
    epoch: int,
    lr: float,
    target_pos: tuple,
    loss_dict: Dict[str, float],
) -> None:
    """Write one CSV row and flush."""
    csv_row = {
        "epoch": epoch,
        "lr": lr,
        "target_pos_x": target_pos[0],
        "target_pos_y": target_pos[1],
    }
    for key in loss_dict:
        csv_row[key] = loss_dict[key]
    csv_writer.writerow(csv_row)
    csv_file.flush()
