"""
Spatiotemporal Contrast Loss for PINN-based Inverse Design

Core idea:
1) Maximum Intensity Projection (MIP) over time.
2) Maximize contrast between target region and off-target region.
"""

import torch
from typing import Dict, Tuple, Optional

import config


def compute_loss(
    wave_output: Dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    target_pos: Tuple[float, float],
    physics_engine,
    target_radius: float = config.LOSS_TARGET_RADIUS,
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute spatiotemporal contrast loss.

    Args:
        wave_output: Dict from physics layer with 'stress_history'
        trajectory: [n_steps, 2] tensor of (x, y) coordinates in meters
        target_pos: (x_target, y_target) in meters
        physics_engine: ViscoelasticWave2D instance for interpolation
        target_radius: Radius of target region in meters
        weights: Optional dict of loss weights

    Returns:
        total_loss: Scalar loss tensor
        loss_dict: Dict of individual loss components for logging
    """
    if weights is None:
        weights = config.LOSS_WEIGHTS

    stress_history = wave_output["stress_history"]
    device = stress_history.device
    dtype = stress_history.dtype
    eps = config.LOSS_EPS

    grid_x, grid_y = physics_engine.grid_x, physics_engine.grid_y
    x_target, y_target = target_pos

    dist_sq = (grid_x - x_target) ** 2 + (grid_y - y_target) ** 2
    sigma = target_radius / 2.0
    target_mask = torch.exp(-dist_sq / (2 * sigma ** 2)).to(device=device, dtype=dtype)
    off_mask = 1.0 - target_mask

    mip_map = torch.amax(stress_history, dim=0)

    signal = (mip_map * target_mask).sum() / (target_mask.sum() + eps)

    p = 6.0
    noise_power = ((mip_map * off_mask).abs() ** p).sum()
    noise_norm = (off_mask.abs() ** p).sum() + eps
    noise = (noise_power / noise_norm).pow(1.0 / p)

    contrast_loss = noise / (signal + eps)

    trajectory_diff = trajectory[1:] - trajectory[:-1]
    step_lengths = torch.sqrt((trajectory_diff ** 2).sum(dim=-1) + eps)
    trajectory_length = step_lengths.sum()

    jerk = torch.zeros((), device=device, dtype=dtype)
    if trajectory.shape[0] >= 4:
        velocity = trajectory[1:] - trajectory[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        jerk_vec = acceleration[1:] - acceleration[:-1]
        jerk = (jerk_vec ** 2).sum()

    weighted_contrast = weights.get("contrast", 1.0) * contrast_loss
    weighted_trajectory_length = weights.get("trajectory_length", 0.0) * trajectory_length
    weighted_jerk = weights.get("jerk", 0.0) * jerk

    total_loss = weighted_contrast + weighted_trajectory_length + weighted_jerk

    loss_dict = {
        "total": total_loss.item(),
        "contrast": weighted_contrast.item(),
        "signal": signal.item(),
        "noise": noise.item(),
        "snr": (signal / (noise + eps)).item(),
        "trajectory_length": trajectory_length.item(),
        "trajectory_penalty": weighted_trajectory_length.item(),
        "jerk": jerk.item(),
        "jerk_penalty": weighted_jerk.item(),
    }

    return total_loss, loss_dict


def test_loss():
    """Test loss function."""
    print("=" * 60)
    print("Testing loss function...")

    from physics_layer import ViscoelasticWave2D
    from model import NeuralSplineTrajectory

    physics = ViscoelasticWave2D(
        Lx=0.1, Ly=0.1, nx=50, ny=50,
        dt=25e-6, n_steps=20,
        device=torch.device('cpu')
    )

    model = NeuralSplineTrajectory(n_steps=20)
    trajectory = model()

    wave_output = physics(trajectory)

    target = (0.05, 0.05)
    loss, loss_dict = compute_loss(
        wave_output, trajectory, target, physics,
        target_radius=0.005,
    )

    print(f"Total loss: {loss.item():.6f}")
    print(f"Loss components: {loss_dict}")

    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Model has gradient: {has_grad}")

    print("Loss test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_loss()
