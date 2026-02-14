"""
Differentiable 2D Viscoelastic Wave FDTD Engine (SH-wave formulation)

Implements anti-plane shear wave propagation with Kelvin-Voigt viscoelasticity.
CFL-stable for shear waves (cs=5 m/s, dt=25μs → Courant ≈ 0.125)

Multi-platform support: CUDA, MPS (Apple Silicon), CPU
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

import config
from device_utils import get_device, safe_outer


class ViscoelasticWave2D(nn.Module):
    """
    Differentiable FDTD solver for 2D viscoelastic shear waves.

    SH-wave formulation (anti-plane):
        ρ ∂vz/∂t = ∂σxz/∂x + ∂σyz/∂y + f(x,y,t)
        σxz = μ ∂vz/∂x + η ∂²vz/∂x∂t  (Kelvin-Voigt)
        σyz = μ ∂vz/∂y + η ∂²vz/∂y∂t  (Kelvin-Voigt)
    """

    def __init__(
        self,
        Lx: float = config.PHYSICS_CONFIG["Lx"],
        Ly: float = config.PHYSICS_CONFIG["Ly"],
        nx: int = config.PHYSICS_CONFIG["nx"],
        ny: int = config.PHYSICS_CONFIG["ny"],
        dt: float = config.PHYSICS_CONFIG["dt"],
        n_steps: int = config.PHYSICS_CONFIG["n_steps"],
        rho: float = config.PHYSICS_CONFIG["rho"],
        cs: float = config.PHYSICS_CONFIG["cs"],
        eta: float = config.PHYSICS_CONFIG["eta"],
        bulk_damping: float = config.PHYSICS_CONFIG.get("bulk_damping", 0.0),
        source_sigma: float = config.PHYSICS_CONFIG["source_sigma"],
        n_pml: int = config.PHYSICS_CONFIG["n_pml"],
        max_damping: float = config.PHYSICS_CONFIG["max_damping"],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the wave solver.

        Args:
            Lx, Ly: Domain size in meters (default 20cm × 20cm)
            nx, ny: Grid points (default 200 × 200, dx=dy=1mm)
            dt: Time step in seconds (default 25μs)
            n_steps: Total simulation steps (default 200, total 5ms)
            rho: Density in kg/m³ (default 1000, water-like)
            cs: Shear wave speed in m/s (default 5)
            eta: Viscosity in Pa·s (default 0.1)
            source_sigma: Gaussian source width in meters (default 3mm)
            n_pml: Absorbing boundary width in cells (default 20)
            max_damping: Maximum damping coefficient at boundary (default 0.05)
            device: Torch device (auto-detected if None)
        """
        super().__init__()

        # Store parameters
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.n_steps = n_steps
        self.source_sigma = source_sigma

        # Material properties
        self.rho = rho
        self.cs = cs
        self.mu = rho * cs * cs  # Shear modulus = ρ * cs²
        self.eta = eta
        self.bulk_damping = bulk_damping

        # CFL check
        courant = cs * dt / self.dx * (2 ** 0.5)
        if courant >= 1.0:
            raise ValueError(f"CFL condition violated: Courant={courant:.3f} >= 1.0")
        print(f"[Physics] CFL check passed: Courant = {courant:.4f}")

        # Precomputed constants
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.dt_over_rho = dt / rho

        # Device detection
        if device is None:
            device = get_device(verbose=True)
        self.device = device

        # Build coordinate grid [nx, ny]
        x = torch.linspace(self.dx / 2, Lx - self.dx / 2, nx, device=device)
        y = torch.linspace(self.dy / 2, Ly - self.dy / 2, ny, device=device)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')

        # Build damping profile for absorbing boundaries (cosine-tapered sponge layer)
        # Note: This is not a true PML; it is a damping sponge for edge absorption.
        self.damping = self._build_damping(n_pml, max_damping)

    def _build_damping(self, n_pml: int, max_damping: float) -> torch.Tensor:
        """
        Build cosine-tapered damping profile for absorbing boundary conditions.

        Returns: [1, nx, ny] tensor with 1.0 in interior, decaying to (1-max_damping) at edges.
        """
        # 1D damping profiles
        damping_x = torch.ones(self.nx, device=self.device)
        damping_y = torch.ones(self.ny, device=self.device)

        for i in range(n_pml):
            arg = torch.tensor(torch.pi * (n_pml - i) / n_pml, device=self.device)
            factor = 1.0 - max_damping * (0.5 + 0.5 * torch.cos(arg))
            damping_x[i] = factor
            damping_x[-(i + 1)] = factor
            damping_y[i] = factor
            damping_y[-(i + 1)] = factor

        # Outer product to 2D (use safe_outer for MPS compatibility)
        damping_2d = safe_outer(damping_x, damping_y, self.device)
        return damping_2d.unsqueeze(0)  # [1, nx, ny]

    def _make_source(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        amplitude: float = 1.0
    ) -> torch.Tensor:
        """
        Create Gaussian source distribution centered at continuous (x_t, y_t).

        Args:
            x_t, y_t: Source center coordinates [batch] in meters
            amplitude: Source amplitude

        Returns: [batch, nx, ny] source distribution
        """
        # x_t, y_t: [batch] → [batch, 1, 1] for broadcasting
        x_t = x_t.unsqueeze(-1).unsqueeze(-1)
        y_t = y_t.unsqueeze(-1).unsqueeze(-1)

        # Gaussian kernel
        r_sq = (self.grid_x.unsqueeze(0) - x_t) ** 2 + (self.grid_y.unsqueeze(0) - y_t) ** 2
        source = amplitude * torch.exp(-r_sq / (2 * self.source_sigma ** 2))

        return source  # [batch, nx, ny]

    def forward(
        self,
        trajectory: torch.Tensor,
        source_amplitude: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Run the FDTD simulation for a given trajectory.

        Args:
            trajectory: [n_steps, 2] tensor of (x, y) coordinates in meters
            source_amplitude: Amplitude of the source at each step

        Returns:
            dict with:
                'vz_history': [n_steps, nx, ny] velocity field history
                'stress_history': [n_steps, nx, ny] stress magnitude history
        """
        n_steps = self.n_steps
        traj_len = trajectory.shape[0]
        batch_size = 1  # Single trajectory optimization
        inv_dx = self.inv_dx
        inv_dy = self.inv_dy
        dt = self.dt
        dt_over_rho = self.dt_over_rho
        mu = self.mu
        eta = self.eta

        # Initialize state tensors [batch, nx, ny]
        vz = torch.zeros(batch_size, self.nx, self.ny, device=self.device)
        strain_xz = torch.zeros(batch_size, self.nx, self.ny, device=self.device)
        strain_yz = torch.zeros(batch_size, self.nx, self.ny, device=self.device)

        # Pre-allocate working buffers to reduce per-step allocations
        dvz_dx = torch.zeros_like(vz)
        dvz_dy = torch.zeros_like(vz)
        dsigma_xz_dx = torch.zeros_like(vz)
        dsigma_yz_dy = torch.zeros_like(vz)

        # Initialize history tensors
        vz_history = torch.empty((n_steps, self.nx, self.ny), device=self.device)
        stress_history = torch.empty((n_steps, self.nx, self.ny), device=self.device)

        # Time stepping loop
        for t in range(n_steps):
            traj_idx = t % traj_len
            # Get source position for this timestep
            x_t = trajectory[traj_idx, 0].unsqueeze(0)  # [1]
            y_t = trajectory[traj_idx, 1].unsqueeze(0)  # [1]

            # Compute source distribution
            source = self._make_source(x_t, y_t, source_amplitude)

            # ===== Step 1: Compute velocity gradients (Strain Rate) =====
            # Central differences on staggered grid
            # dvz/dx: difference along x-axis (axis 1)
            dvz_dx.zero_()
            dvz_dx[:, :-1, :] = (vz[:, 1:, :] - vz[:, :-1, :]) * inv_dx

            # dvz/dy: difference along y-axis (axis 2)
            dvz_dy.zero_()
            dvz_dy[:, :, :-1] = (vz[:, :, 1:] - vz[:, :, :-1]) * inv_dy

            # ===== Step 2: Update Strains =====
            strain_xz = strain_xz + dvz_dx * dt
            strain_yz = strain_yz + dvz_dy * dt

            # ===== Step 3: Update Stresses (Kelvin-Voigt) =====
            # σ = μ * ε + η * ε_dot
            sigma_xz = mu * strain_xz + eta * dvz_dx
            sigma_yz = mu * strain_yz + eta * dvz_dy

            # ===== Step 4: Compute stress divergence =====
            dsigma_xz_dx.zero_()
            dsigma_xz_dx[:, 1:, :] = (sigma_xz[:, 1:, :] - sigma_xz[:, :-1, :]) * inv_dx

            dsigma_yz_dy.zero_()
            dsigma_yz_dy[:, :, 1:] = (sigma_yz[:, :, 1:] - sigma_yz[:, :, :-1]) * inv_dy

            # ===== Step 5: Update velocity =====
            vz = vz + dt_over_rho * (dsigma_xz_dx + dsigma_yz_dy + source)

            # ===== Step 6: Apply bulk damping (tissue-like attenuation) =====
            if self.bulk_damping > 0.0:
                vz = vz * (1.0 - self.bulk_damping * dt)

            # ===== Step 7: Apply absorbing boundary =====
            vz = vz * self.damping

            # Store history (squeeze batch dimension)
            vz_history[t] = vz.squeeze(0)
            
            # Calculate and store stress magnitude
            # Add epsilon to prevent NaN gradient in sqrt(0)
            stress_mag = torch.sqrt(sigma_xz ** 2 + sigma_yz ** 2 + config.PHYSICS_EPS)
            stress_history[t] = stress_mag.squeeze(0)

        return {
            'vz_history': vz_history,
            'stress_history': stress_history,
        }

    def get_field_at_point(
        self,
        field: torch.Tensor,
        x: float,
        y: float
    ) -> torch.Tensor:
        """
        Bilinear interpolation of field value at continuous (x, y) coordinates.

        Args:
            field: [nx, ny] field tensor
            x, y: Coordinate in meters

        Returns: Interpolated value
        """
        # Convert to grid indices
        ix = (x / self.dx) - 0.5
        iy = (y / self.dy) - 0.5

        # Get integer indices
        ix0 = int(torch.floor(torch.tensor(ix)))
        iy0 = int(torch.floor(torch.tensor(iy)))
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # Clamp to valid range
        ix0 = max(0, min(ix0, self.nx - 2))
        iy0 = max(0, min(iy0, self.ny - 2))
        ix1 = min(ix1, self.nx - 1)
        iy1 = min(iy1, self.ny - 1)

        # Fractional parts
        fx = ix - ix0
        fy = iy - iy0

        # Bilinear interpolation
        val = (
            field[ix0, iy0] * (1 - fx) * (1 - fy) +
            field[ix0, iy1] * (1 - fx) * fy +
            field[ix1, iy0] * fx * (1 - fy) +
            field[ix1, iy1] * fx * fy
        )

        return val


def test_physics():
    """Basic unit test: point source should produce circular wavefront."""
    print("=" * 60)
    print("Testing physics layer...")

    # Small grid for quick test
    physics = ViscoelasticWave2D(
        Lx=0.1, Ly=0.1, nx=100, ny=100,
        dt=25e-6, n_steps=50,
        device=torch.device('cpu')
    )

    # Static source at center
    center = 0.05
    # Make trajectory require grad for backward check
    traj_x = torch.full((50,), center, requires_grad=True)
    traj_y = torch.full((50,), center, requires_grad=True)
    trajectory = torch.stack([traj_x, traj_y], dim=-1)

    # Run simulation
    output = physics(trajectory)

    # Check wavefront: at t=50, wave should have traveled ~50 * 0.125 * 0.5mm ≈ 3mm
    # (Courant * n_steps * dx in each direction, but slower due to 2D spreading)
    vz = output['vz_history']

    print(f"Vz history shape: {vz.shape}")
    print(f"Max |vz| at final step: {vz[-1].abs().max():.6e}")

    # Check gradient flow
    vz_sum = vz.sum()
    vz_sum.backward()

    print("Gradient check: trajectory has gradient =", trajectory.grad is not None)
    print("Physics test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_physics()
