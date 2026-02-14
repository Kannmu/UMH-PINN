SEED = 42

PHYSICS_CONFIG = {
    "Lx": 0.1,
    "Ly": 0.1,
    "nx": 100,
    "ny": 100,
    "dt": 25e-6,
    "n_steps": 200,
    "sim_cycles": 3,
    "rho": 1000.0,
    "cs": 5.0,
    "eta": 2.0,
    "bulk_damping": 250.0,
    "source_sigma": 0.0036,
    "n_pml": 5,
    "max_damping": 0.5,
}

MODEL_CONFIG = {
    "n_steps": 200,
    "n_control_points": 16,
    "hidden_dim": 128,
    "num_layers": 4,
    "latent_dim": 16,
}

TRAIN_CONFIG = {
    "n_epochs": 200,
    "lr": 1e-3,
    "target_pos": (0.05, 0.05),
    "save_interval": 100,
    "output_dir": "output",
}

LOSS_WEIGHTS = {
    "contrast": 5.0,
    "trajectory_length": 1.0,
    "jerk": 1.0,
}

LOSS_TARGET_DIAMETER = 0.00857
LOSS_TARGET_RADIUS = LOSS_TARGET_DIAMETER / 2
LOSS_EPS = 1e-8
PHYSICS_EPS = 1e-8

VISUAL_CONFIG = {
    "video_fps": 20,
    "video_codec": "libx264",
    "video_bitrate": "2M",
    "video_dpi": 120,
    "video_frame_stride": 1,
    "video_max_frames": None,
    "wave_figsize": (8, 8),
    "traj_figsize": (8, 8),
    "vmax_scale": 0.8,
    "cm_scale": 100,
    "plot_xlim_cm": 10,
    "plot_ylim_cm": 10,
}
