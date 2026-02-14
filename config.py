SEED = 42

PHYSICS_CONFIG = {
    "Lx": 0.1,
    "Ly": 0.1,
    "Lz": 0.02,
    "nx": 100,
    "ny": 100,
    "nz": 20,
    "dt": 1.25e-5,
    "n_steps": 400,
    "sim_cycles": 3,
    "bulk_damping": 60.0,
    "source_sigma_xy": 0.0025,
    "source_sigma_z": 0.0012,
    "source_depth": 0.0015,
    "n_pml": 8,
    "max_damping": 0.35,
    "layers": [
        {
            "name": "epidermis",
            "z_max": 0.001,
            "rho": 1150.0,
            "vp": 30.0,
            "vs": 12.0,
            "eta": 8.0,
        },
        {
            "name": "dermis",
            "z_max": 0.003,
            "rho": 1100.0,
            "vp": 20.0,
            "vs": 6.5,
            "eta": 6.0,
        },
        {
            "name": "fat",
            "z_max": 0.010,
            "rho": 950.0,
            "vp": 12.0,
            "vs": 3.5,
            "eta": 4.0,
        },
        {
            "name": "muscle",
            "z_max": 0.020,
            "rho": 1050.0,
            "vp": 24.0,
            "vs": 8.0,
            "eta": 7.0,
        },
    ],
}

MODEL_CONFIG = {
    "n_steps": 200,
    "n_control_points": 16,
    "hidden_dim": 128,
    "num_layers": 5,
    "latent_dim": 16,
}

TRAIN_CONFIG = {
    "n_epochs": 1000,
    "lr": 1e-3,
    "target_pos": (0.05, 0.05),
    "save_interval": 200,
    "output_dir": "output",
    "compile": False,
    "use_tf32": True,
    "batch_size": 1,
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
    "gif_fps": 20,
    "wave_figsize": (8, 8),
    "traj_figsize": (8, 8),
    "vmax_scale": 0.8,
    "cm_scale": 100,
    "plot_xlim_cm": 10,
    "plot_ylim_cm": 10,
    "z_slice_index": 1,
}
