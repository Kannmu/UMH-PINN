"""
Spatiotemporal Pulse-Focused Loss for PINN-based Inverse Design.

Core idea:
1) Temporal concentration at a desired pulse time.
2) Spatial silence off-target over the full time window.
3) Energy efficiency with pre-pulse suppression.
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
    Compute spatiotemporal pulse-focused loss.

    Core objectives:
    1) Temporal concentration at a desired pulse time.
    2) Spatial silence off-target over the full time window.
    3) Energy efficiency (high peak with low total energy).
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

    mask_sum = target_mask.sum() + eps
    target_trace = (stress_history * target_mask).sum(dim=(1, 2)) / mask_sum

    n_steps = stress_history.shape[0]
    dt = getattr(physics_engine, "dt", 1.0)
    time = torch.arange(n_steps, device=device, dtype=dtype) * dt

    freq_cfg = getattr(config, "LOSS_FREQ_CONFIG", {})
    target_hz = kwargs.get("target_hz", freq_cfg.get("target_hz", 200.0))
    sigma_hz = kwargs.get("sigma_hz", freq_cfg.get("sigma_hz", 25.0))
    min_cycles = kwargs.get("min_cycles", freq_cfg.get("min_cycles", 2))
    target_hz = torch.tensor(target_hz, device=device, dtype=dtype)
    sigma_hz = torch.tensor(sigma_hz, device=device, dtype=dtype)
    min_cycles = torch.tensor(min_cycles, device=device, dtype=dtype)

    target_trace_detrend = target_trace - target_trace.mean()
    window = torch.hann_window(n_steps, device=device, dtype=dtype)
    fft_vals = torch.fft.rfft(target_trace_detrend * window)
    fft_mag = fft_vals.abs() + eps
    freqs = torch.fft.rfftfreq(n_steps, d=dt).to(device=device, dtype=dtype)
    target_weight = torch.exp(-0.5 * ((freqs - target_hz) / (sigma_hz + eps)) ** 2)
    off_weight = 1.0 - target_weight
    target_band_power = (fft_mag * target_weight).sum() / (target_weight.sum() + eps)
    off_band_power = (fft_mag * off_weight).sum() / (off_weight.sum() + eps)
    cycles = target_hz * n_steps * dt
    cycle_weight = torch.clamp(cycles / (min_cycles + eps), 0.0, 1.0)
    frequency_loss = cycle_weight * (off_band_power / (target_band_power + eps))

    t_peak = kwargs.get("pulse_t_peak", 0.5 * (n_steps - 1) * dt)
    sigma_t = kwargs.get("pulse_sigma", 0.08 * n_steps * dt)
    t_peak = torch.tensor(t_peak, device=device, dtype=dtype)
    sigma_t = torch.tensor(sigma_t, device=device, dtype=dtype)

    pulse_window = torch.exp(-0.5 * ((time - t_peak) / (sigma_t + eps)) ** 2)
    target_peak = (target_trace * pulse_window).sum() / (pulse_window.sum() + eps)
    pulse_loss = -target_peak

    pre_mask = time < (t_peak - 2.0 * sigma_t)
    if pre_mask.any():
        pre_pulse = (target_trace[pre_mask] ** 2).mean()
    else:
        pre_pulse = torch.zeros((), device=device, dtype=dtype)

    time_weight = torch.sigmoid((time - (t_peak - sigma_t)) / (sigma_t / 2.0 + eps))
    silence_loss = ((stress_history * off_mask * time_weight[:, None, None]) ** 2).mean()
    total_energy = (stress_history ** 2).mean()
    efficiency_loss = total_energy / (target_peak ** 2 + eps)

    contrast_ratio = kwargs.get("contrast_ratio", config.LOSS_CONTRAST_RATIO)
    contrast_ratio = torch.tensor(contrast_ratio, device=device, dtype=dtype)
    spatial_max_stress = stress_history.max(dim=0).values
    sidelobe_peak = (spatial_max_stress * off_mask).max()
    contrast_loss = torch.relu(sidelobe_peak - target_peak * contrast_ratio)

    trajectory_diff = trajectory[1:] - trajectory[:-1]
    step_lengths = torch.sqrt((trajectory_diff ** 2).sum(dim=-1) + eps)
    trajectory_length = step_lengths.sum()

    jerk = torch.zeros((), device=device, dtype=dtype)
    if trajectory.shape[0] >= 4:
        velocity = trajectory[1:] - trajectory[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        jerk_vec = acceleration[1:] - acceleration[:-1]
        jerk = (jerk_vec ** 2).sum()

    traj_cfg = getattr(config, "TRAJ_CONSTRAINT_CONFIG", {})
    soft_target_radius = torch.tensor(
        kwargs.get("traj_target_radius", traj_cfg.get("target_radius", 0.03)),
        device=device,
        dtype=dtype,
    )
    boundary_margin = torch.tensor(
        kwargs.get("traj_boundary_margin", traj_cfg.get("boundary_margin", 0.005)),
        device=device,
        dtype=dtype,
    )
    softplus_beta = kwargs.get("traj_softplus_beta", traj_cfg.get("softplus_beta", 50.0))

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    x_min = boundary_margin
    y_min = boundary_margin
    x_max = torch.tensor(physics_engine.Lx, device=device, dtype=dtype) - boundary_margin
    y_max = torch.tensor(physics_engine.Ly, device=device, dtype=dtype) - boundary_margin

    center = torch.tensor([physics_engine.Lx, physics_engine.Ly], device=device, dtype=dtype) / 2.0
    radial_dist = torch.sqrt(((trajectory - center) ** 2).sum(dim=-1) + eps)
    radial_violation = torch.relu(radial_dist - soft_target_radius)

    bound_violation = (
        torch.relu(x_min - x)
        + torch.relu(x - x_max)
        + torch.relu(y_min - y)
        + torch.relu(y - y_max)
    )

    trajectory_region_loss = (
        torch.nn.functional.softplus(radial_violation, beta=softplus_beta).mean()
        + torch.nn.functional.softplus(bound_violation, beta=softplus_beta).mean()
    )

    weighted_pulse = weights.get("pulse", 1.0) * pulse_loss
    weighted_silence = weights.get("silence", 1.0) * silence_loss
    weighted_contrast = weights.get("contrast", 0.0) * contrast_loss
    weighted_efficiency = weights.get("efficiency", 1.0) * efficiency_loss
    weighted_pre_pulse = weights.get("pre_pulse", 1.0) * pre_pulse
    weighted_trajectory_length = weights.get("trajectory_length", 0.0) * trajectory_length
    weighted_jerk = weights.get("jerk", 0.0) * jerk
    weighted_frequency = weights.get("frequency", 0.0) * frequency_loss
    weighted_trajectory_region = weights.get("trajectory_region", 0.0) * trajectory_region_loss

    total_loss = (
        weighted_pulse
        + weighted_silence
        + weighted_contrast
        + weighted_efficiency
        + weighted_pre_pulse
        + weighted_trajectory_length
        + weighted_jerk
        + weighted_frequency
        + weighted_trajectory_region
    )

    loss_dict = {
        "total": total_loss.item(),
        "pulse": weighted_pulse.item(),
        "silence": weighted_silence.item(),
        "contrast": weighted_contrast.item(),
        "efficiency": weighted_efficiency.item(),
        "pre_pulse": weighted_pre_pulse.item(),
        "frequency": weighted_frequency.item(),
        "frequency_loss": frequency_loss.item(),
        "target_peak": target_peak.item(),
        "sidelobe_peak": sidelobe_peak.item(),
        "total_energy": total_energy.item(),
        "trajectory_length": trajectory_length.item(),
        "trajectory_penalty": weighted_trajectory_length.item(),
        "jerk": jerk.item(),
        "jerk_penalty": weighted_jerk.item(),
        "trajectory_region": trajectory_region_loss.item(),
        "trajectory_region_penalty": weighted_trajectory_region.item(),
        "target_band_power": target_band_power.item(),
        "off_band_power": off_band_power.item(),
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
