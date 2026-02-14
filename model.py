"""
Neural Spline Trajectory Model for PINN-based Inverse Design

The trajectory is a [n_steps, 2] tensor representing (x, y) coordinates
of the focal point at each timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


def _catmull_rom_weights(t: torch.Tensor) -> torch.Tensor:
    """
    Compute Catmull-Rom spline weights for local parameter t in [0, 1].

    Returns:
        weights: [..., 4] tensor of weights for points P0..P3
    """
    t2 = t * t
    t3 = t2 * t
    w0 = 0.5 * (-t + 2.0 * t2 - t3)
    w1 = 0.5 * (2.0 - 5.0 * t2 + 3.0 * t3)
    w2 = 0.5 * (t + 4.0 * t2 - 3.0 * t3)
    w3 = 0.5 * (-t2 + t3)
    return torch.stack([w0, w1, w2, w3], dim=-1)


class NeuralSplineTrajectory(nn.Module):
    """
    Neural spline trajectory generator.

    - Uses a small MLP to predict control point offsets from a learned latent code.
    - Uses Catmull-Rom spline interpolation to generate smooth trajectories.
    - Initializes control points as a centered circle.
    """

    def __init__(
        self,
        n_steps: int = config.MODEL_CONFIG["n_steps"],
        n_control_points: int = config.MODEL_CONFIG["n_control_points"],
        hidden_dim: int = config.MODEL_CONFIG["hidden_dim"],
        num_layers: int = config.MODEL_CONFIG["num_layers"],
        latent_dim: int = config.MODEL_CONFIG["latent_dim"],
    ):
        super().__init__()
        if n_control_points < 4:
            raise ValueError("Catmull-Rom spline requires at least 4 control points.")
        if num_layers < 2:
            raise ValueError("NeuralSplineTrajectory requires at least 2 layers.")

        self.n_steps = n_steps
        self.n_control_points = n_control_points

        layers = [nn.Linear(latent_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, n_control_points * 2))
        self.mlp = nn.Sequential(*layers)

        self.latent = nn.Parameter(torch.zeros(latent_dim))

        time_points = torch.linspace(0.0, 1.0, n_steps)
        self.register_buffer("time_points", time_points)

        space_scale = torch.tensor(
            [config.PHYSICS_CONFIG["Lx"], config.PHYSICS_CONFIG["Ly"]],
            dtype=time_points.dtype,
        )
        self.register_buffer("space_scale", space_scale)

        self.base_logits = nn.Parameter(self._init_circle_logits())

        basis_indices, basis_weights = self._build_catmull_rom_basis(
            n_steps=n_steps,
            n_control_points=n_control_points,
            dtype=time_points.dtype,
        )
        self.register_buffer("basis_indices", basis_indices)
        self.register_buffer("basis_weights", basis_weights)

    def _init_circle_logits(self) -> torch.Tensor:
        """Initialize control point logits as a circle in the center of the domain."""
        eps = 1e-6
        center = self.space_scale / 2.0
        radius = torch.min(self.space_scale) / 4.0

        t = torch.linspace(0, 2 * torch.pi, self.n_control_points)
        circle_pos = torch.zeros(self.n_control_points, 2)
        circle_pos[:, 0] = center[0] + radius * torch.cos(t)
        circle_pos[:, 1] = center[1] + radius * torch.sin(t)

        normed = (circle_pos / self.space_scale).clamp(eps, 1.0 - eps)
        logits = torch.log(normed) - torch.log(1.0 - normed)
        return logits

    def _build_catmull_rom_basis(
        self,
        n_steps: int,
        n_control_points: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build Catmull-Rom spline basis indices and weights.

        Returns:
            basis_indices: [n_steps, 4] indices of control points
            basis_weights: [n_steps, 4] spline weights
        """
        segment_count = n_control_points - 3
        s = torch.linspace(0.0, segment_count - 1e-6, n_steps, dtype=dtype)
        seg_idx = torch.floor(s).to(torch.long)
        local_t = (s - seg_idx.to(dtype)).clamp(0.0, 1.0)

        i0 = seg_idx
        i1 = seg_idx + 1
        i2 = seg_idx + 2
        i3 = seg_idx + 3
        basis_indices = torch.stack([i0, i1, i2, i3], dim=-1)
        basis_weights = _catmull_rom_weights(local_t)
        return basis_indices, basis_weights

    def _compute_control_points(self) -> torch.Tensor:
        latent = self.latent.unsqueeze(0)
        offsets = self.mlp(latent).view(self.n_control_points, 2)
        logits = self.base_logits + offsets
        control_points = torch.sigmoid(logits) * self.space_scale
        return control_points

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Return [batch, n_steps, 2] trajectory in physical coordinates."""
        latent = self.latent.unsqueeze(0).repeat(batch_size, 1)
        offsets = self.mlp(latent).view(batch_size, self.n_control_points, 2)
        logits = self.base_logits.unsqueeze(0) + offsets
        control_points = torch.sigmoid(logits) * self.space_scale

        selected = control_points[:, self.basis_indices]
        weights = self.basis_weights.unsqueeze(0).unsqueeze(-1)
        trajectory = (selected * weights).sum(dim=2)
        return trajectory


def test_model():
    print("=" * 60)
    print("Testing NeuralSplineTrajectory model...")

    model = NeuralSplineTrajectory(n_steps=200)

    traj = model()
    print(f"Trajectory shape: {traj.shape}")
    print(f"X range: [{traj[:, 0].min():.4f}, {traj[:, 0].max():.4f}]")
    print(f"Y range: [{traj[:, 1].min():.4f}, {traj[:, 1].max():.4f}]")

    print("Model test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
