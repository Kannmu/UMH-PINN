"""
Differentiable 3D Viscoelastic Elastic Wave FDTD Engine

- Full 3D elastic wave propagation (P and S waves)
- Kelvin-Voigt viscoelasticity (dashpot in shear)
- Multi-layer skin model with depth-dependent material properties
- Absorbing sponge boundary

Coordinates:
- x: lateral (0..Lx)
- y: lateral (0..Ly)
- z: depth (0..Lz), z=0 is skin surface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.jit import script

import config
from device_utils import get_device


@script
def compute_source_kernel(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    grid_z: torch.Tensor,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    source_depth: float,
    sigma_xy: float,
    sigma_z: float,
) -> torch.Tensor:
    x_t = x_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    y_t = y_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    r_xy = (grid_x.unsqueeze(0) - x_t) ** 2 + (grid_y.unsqueeze(0) - y_t) ** 2
    r_z = (grid_z.unsqueeze(0) - source_depth) ** 2

    source = torch.exp(-r_xy / (2 * sigma_xy ** 2))
    source = source * torch.exp(-r_z / (2 * sigma_z ** 2))
    return source


@script
def fdtd_step_kernel(
    vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor,
    sxx: torch.Tensor, syy: torch.Tensor, szz: torch.Tensor,
    sxy: torch.Tensor, sxz: torch.Tensor, syz: torch.Tensor,
    rho: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor, eta: torch.Tensor,
    damping: torch.Tensor, source: torch.Tensor,
    dt: float, inv_dx: float, inv_dy: float, inv_dz: float,
    bulk_damping: float,
    physics_eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Velocity gradients
    dvx_dx = torch.zeros_like(vx)
    dvx_dy = torch.zeros_like(vx)
    dvx_dz = torch.zeros_like(vx)
    dvy_dx = torch.zeros_like(vx)
    dvy_dy = torch.zeros_like(vx)
    dvy_dz = torch.zeros_like(vx)
    dvz_dx = torch.zeros_like(vx)
    dvz_dy = torch.zeros_like(vx)
    dvz_dz = torch.zeros_like(vx)

    dvx_dx[:, :-1, :, :] = (vx[:, 1:, :, :] - vx[:, :-1, :, :]) * inv_dx
    dvx_dy[:, :, :-1, :] = (vx[:, :, 1:, :] - vx[:, :, :-1, :]) * inv_dy
    dvx_dz[:, :, :, :-1] = (vx[:, :, :, 1:] - vx[:, :, :, :-1]) * inv_dz

    dvy_dx[:, :-1, :, :] = (vy[:, 1:, :, :] - vy[:, :-1, :, :]) * inv_dx
    dvy_dy[:, :, :-1, :] = (vy[:, :, 1:, :] - vy[:, :, :-1, :]) * inv_dy
    dvy_dz[:, :, :, :-1] = (vy[:, :, :, 1:] - vy[:, :, :, :-1]) * inv_dz

    dvz_dx[:, :-1, :, :] = (vz[:, 1:, :, :] - vz[:, :-1, :, :]) * inv_dx
    dvz_dy[:, :, :-1, :] = (vz[:, :, 1:, :] - vz[:, :, :-1, :]) * inv_dy
    dvz_dz[:, :, :, :-1] = (vz[:, :, :, 1:] - vz[:, :, :, :-1]) * inv_dz

    # Strain-rate tensor components
    exx = dvx_dx
    eyy = dvy_dy
    ezz = dvz_dz
    exy = 0.5 * (dvx_dy + dvy_dx)
    exz = 0.5 * (dvx_dz + dvz_dx)
    eyz = 0.5 * (dvy_dz + dvz_dy)

    # Stress updates (Kelvin-Voigt in shear)
    trace = exx + eyy + ezz
    sxx = sxx + dt * ((lam + 2 * mu) * exx + lam * (trace - exx))
    syy = syy + dt * ((lam + 2 * mu) * eyy + lam * (trace - eyy))
    szz = szz + dt * ((lam + 2 * mu) * ezz + lam * (trace - ezz))

    sxy = sxy + dt * (2 * mu * exy + 2 * eta * exy)
    sxz = sxz + dt * (2 * mu * exz + 2 * eta * exz)
    syz = syz + dt * (2 * mu * eyz + 2 * eta * eyz)

    # Stress divergence
    dsxx_dx = torch.zeros_like(vx)
    dsxy_dy = torch.zeros_like(vx)
    dsxz_dz = torch.zeros_like(vx)

    dsyx_dx = torch.zeros_like(vx)
    dsyy_dy = torch.zeros_like(vx)
    dsyz_dz = torch.zeros_like(vx)

    dszx_dx = torch.zeros_like(vx)
    dszy_dy = torch.zeros_like(vx)
    dszz_dz = torch.zeros_like(vx)

    dsxx_dx[:, 1:, :, :] = (sxx[:, 1:, :, :] - sxx[:, :-1, :, :]) * inv_dx
    dsxy_dy[:, :, 1:, :] = (sxy[:, :, 1:, :] - sxy[:, :, :-1, :]) * inv_dy
    dsxz_dz[:, :, :, 1:] = (sxz[:, :, :, 1:] - sxz[:, :, :, :-1]) * inv_dz

    dsyx_dx[:, 1:, :, :] = (sxy[:, 1:, :, :] - sxy[:, :-1, :, :]) * inv_dx
    dsyy_dy[:, :, 1:, :] = (syy[:, :, 1:, :] - syy[:, :, :-1, :]) * inv_dy
    dsyz_dz[:, :, :, 1:] = (syz[:, :, :, 1:] - syz[:, :, :, :-1]) * inv_dz

    dszx_dx[:, 1:, :, :] = (sxz[:, 1:, :, :] - sxz[:, :-1, :, :]) * inv_dx
    dszy_dy[:, :, 1:, :] = (syz[:, :, 1:, :] - syz[:, :, :-1, :]) * inv_dy
    dszz_dz[:, :, :, 1:] = (szz[:, :, :, 1:] - szz[:, :, :, :-1]) * inv_dz

    # Velocity updates
    vx = vx + dt * (dsxx_dx + dsxy_dy + dsxz_dz) / rho
    vy = vy + dt * (dsyx_dx + dsyy_dy + dsyz_dz) / rho
    vz = vz + dt * (dszx_dx + dszy_dy + dszz_dz + source) / rho

    if bulk_damping > 0.0:
        factor = 1.0 - bulk_damping * dt
        vx = vx * factor
        vy = vy * factor
        vz = vz * factor

    vx = vx * damping
    vy = vy * damping
    vz = vz * damping

    stress_mag = torch.sqrt(sxy ** 2 + sxz ** 2 + syz ** 2 + physics_eps)
    
    return vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, stress_mag


@dataclass
class MaterialFields:
    rho: torch.Tensor
    vp: torch.Tensor
    vs: torch.Tensor
    eta: torch.Tensor

    @property
    def mu(self) -> torch.Tensor:
        return self.rho * (self.vs ** 2)

    @property
    def lam(self) -> torch.Tensor:
        return self.rho * (self.vp ** 2) - 2.0 * self.mu


class ViscoelasticWave3D(nn.Module):
    """
    Differentiable 3D elastic wave solver with Kelvin-Voigt shear viscosity.

    State variables:
    - Particle velocities: vx, vy, vz
    - Stresses: sxx, syy, szz, sxy, sxz, syz

    Update scheme:
    - Explicit finite differences (collocated grid)
    """

    def __init__(
        self,
        Lx: float = config.PHYSICS_CONFIG["Lx"],
        Ly: float = config.PHYSICS_CONFIG["Ly"],
        Lz: float = config.PHYSICS_CONFIG["Lz"],
        nx: int = config.PHYSICS_CONFIG["nx"],
        ny: int = config.PHYSICS_CONFIG["ny"],
        nz: int = config.PHYSICS_CONFIG["nz"],
        dt: float = config.PHYSICS_CONFIG["dt"],
        n_steps: int = config.PHYSICS_CONFIG["n_steps"],
        bulk_damping: float = config.PHYSICS_CONFIG.get("bulk_damping", 0.0),
        source_sigma_xy: float = config.PHYSICS_CONFIG["source_sigma_xy"],
        source_sigma_z: float = config.PHYSICS_CONFIG["source_sigma_z"],
        source_depth: float = config.PHYSICS_CONFIG["source_depth"],
        n_pml: int = config.PHYSICS_CONFIG["n_pml"],
        max_damping: float = config.PHYSICS_CONFIG["max_damping"],
        layers: Optional[List[Dict[str, float]]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz
        self.dt = dt
        self.n_steps = n_steps
        self.bulk_damping = bulk_damping
        self.source_sigma_xy = source_sigma_xy
        self.source_sigma_z = source_sigma_z
        self.source_depth = source_depth

        if device is None:
            device = get_device(verbose=True)
        self.device = device
        self._mem_logged = False

        # CFL check based on maximum vp
        layer_defs = layers if layers is not None else config.PHYSICS_CONFIG["layers"]
        max_vp = max(layer["vp"] for layer in layer_defs)
        min_dx = min(self.dx, self.dy, self.dz)
        courant = max_vp * dt / min_dx * (3 ** 0.5)
        if courant >= 1.0:
            raise ValueError(f"CFL condition violated: Courant={courant:.3f} >= 1.0")
        print(f"[Physics] CFL check passed: Courant = {courant:.4f}")

        # Grid coordinates
        x = torch.linspace(self.dx / 2, Lx - self.dx / 2, nx, device=device)
        y = torch.linspace(self.dy / 2, Ly - self.dy / 2, ny, device=device)
        z = torch.linspace(self.dz / 2, Lz - self.dz / 2, nz, device=device)
        self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(x, y, z, indexing="ij")

        # Material fields
        self.material = self._build_material_fields(layer_defs)

        # Damping (sponge) boundary
        self.damping = self._build_damping(n_pml, max_damping)

        # Precomputed constants
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.inv_dz = 1.0 / self.dz

    def _build_material_fields(self, layers: List[Dict[str, float]]) -> MaterialFields:
        """Build depth-dependent material tensors for rho, vp, vs, eta."""
        z = self.grid_z
        rho = torch.empty_like(z)
        vp = torch.empty_like(z)
        vs = torch.empty_like(z)
        eta = torch.empty_like(z)

        z_prev = 0.0
        for layer in layers:
            z_max = layer["z_max"]
            mask = (z >= z_prev) & (z < z_max)
            rho = torch.where(mask, torch.tensor(layer["rho"], device=self.device), rho)
            vp = torch.where(mask, torch.tensor(layer["vp"], device=self.device), vp)
            vs = torch.where(mask, torch.tensor(layer["vs"], device=self.device), vs)
            eta = torch.where(mask, torch.tensor(layer["eta"], device=self.device), eta)
            z_prev = z_max

        # Fill any remaining depth with last layer
        mask = z >= z_prev
        rho = torch.where(mask, torch.tensor(layers[-1]["rho"], device=self.device), rho)
        vp = torch.where(mask, torch.tensor(layers[-1]["vp"], device=self.device), vp)
        vs = torch.where(mask, torch.tensor(layers[-1]["vs"], device=self.device), vs)
        eta = torch.where(mask, torch.tensor(layers[-1]["eta"], device=self.device), eta)

        return MaterialFields(rho=rho, vp=vp, vs=vs, eta=eta)

    def _build_damping(self, n_pml: int, max_damping: float) -> torch.Tensor:
        """Cosine-tapered damping sponge for 3D boundaries."""
        damping_x = torch.ones(self.nx, device=self.device)
        damping_y = torch.ones(self.ny, device=self.device)
        damping_z = torch.ones(self.nz, device=self.device)

        for i in range(n_pml):
            arg = torch.tensor(torch.pi * (n_pml - i) / n_pml, device=self.device)
            factor = 1.0 - max_damping * (0.5 + 0.5 * torch.cos(arg))
            damping_x[i] = factor
            damping_x[-(i + 1)] = factor
            damping_y[i] = factor
            damping_y[-(i + 1)] = factor
            damping_z[i] = factor
            damping_z[-(i + 1)] = factor

        damping_3d = damping_x[:, None, None] * damping_y[None, :, None] * damping_z[None, None, :]
        return damping_3d.unsqueeze(0)

    def _make_source(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Create anisotropic Gaussian source centered at (x_t, y_t, z0).
        Helper wrapper around the JIT kernel.
        """
        return compute_source_kernel(
            self.grid_x, self.grid_y, self.grid_z,
            x_t, y_t,
            self.source_depth,
            self.source_sigma_xy,
            self.source_sigma_z
        )

    def forward(
        self,
        trajectory: torch.Tensor,
        source_amplitude: float = 1.0,
        return_vz: bool = False,
        return_stress_history: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run 3D elastic wave simulation.

        Args:
            trajectory: [n_steps, 2] (x, y) coordinates in meters
            source_amplitude: scalar amplitude (full power by default)
            return_vz: Whether to return vertical velocity history (memory intensive)
            return_stress_history: Whether to return full stress history (very memory intensive)

        Returns:
            dict with:
                'mip_map': [batch, nx, ny, nz] (log-sum-exp MIP over time)
                'vz_history': [n_steps, nx, ny, nz] (vertical velocity) - ONLY if return_vz=True
                'stress_history': [n_steps, nx, ny, nz] (shear stress magnitude) - ONLY if return_stress_history=True
        """
        n_steps = self.n_steps
        traj_len = trajectory.shape[0]
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        batch_size = trajectory.shape[0]

        vx = torch.zeros(batch_size, self.nx, self.ny, self.nz, device=self.device)
        vy = torch.zeros_like(vx)
        vz = torch.zeros_like(vx)

        sxx = torch.zeros_like(vx)
        syy = torch.zeros_like(vx)
        szz = torch.zeros_like(vx)
        sxy = torch.zeros_like(vx)
        sxz = torch.zeros_like(vx)
        syz = torch.zeros_like(vx)


        # Optimize memory: Only allocate histories if requested
        vz_history = None
        if return_vz:
            vz_history = torch.empty((batch_size, n_steps, self.nx, self.ny, self.nz), device=self.device)
            print(f"[Physics] vz_history allocated: {vz_history.numel() * 4 / (1024**3):.3f} GB")

        stress_history = None
        if return_stress_history:
            stress_history = torch.empty((batch_size, n_steps, self.nx, self.ny, self.nz), device=self.device)
            print(f"[Physics] stress_history allocated: {stress_history.numel() * 4 / (1024**3):.3f} GB")

        beta = 1000.0
        running_max = torch.full((batch_size, self.nx, self.ny, self.nz), -1e9, device=self.device)
        running_sum = torch.zeros((batch_size, self.nx, self.ny, self.nz), device=self.device)

        inv_dx = self.inv_dx
        inv_dy = self.inv_dy
        inv_dz = self.inv_dz
        dt = self.dt
        bulk_damping = self.bulk_damping
        physics_eps = config.PHYSICS_EPS

        rho = self.material.rho
        mu = self.material.mu
        lam = self.material.lam
        eta = self.material.eta
        damping = self.damping
        
        # Grid for source generation
        grid_x, grid_y, grid_z = self.grid_x, self.grid_y, self.grid_z
        source_depth = self.source_depth
        source_sigma_xy = self.source_sigma_xy
        source_sigma_z = self.source_sigma_z

        for t in range(n_steps):
            traj_idx = t % traj_len
            x_t = trajectory[:, traj_idx, 0]
            y_t = trajectory[:, traj_idx, 1]

            # JIT-compiled source generation
            source = compute_source_kernel(
                grid_x, grid_y, grid_z,
                x_t, y_t,
                source_depth,
                source_sigma_xy,
                source_sigma_z
            ) * source_amplitude


            # JIT-compiled FDTD update
            vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, stress_mag = fdtd_step_kernel(
                vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                rho, mu, lam, eta, damping, source,
                dt, inv_dx, inv_dy, inv_dz,
                bulk_damping, physics_eps
            )

            if return_vz and vz_history is not None:
                vz_history[:, t] = vz

            if return_stress_history and stress_history is not None:
                stress_history[:, t] = stress_mag

            stress_scaled = stress_mag * beta
            prev_max = running_max
            running_max = torch.maximum(prev_max, stress_scaled)
            running_sum = running_sum * torch.exp(prev_max - running_max) + torch.exp(stress_scaled - running_max)

        mip_map = (running_max + torch.log(running_sum + config.PHYSICS_EPS)) / beta

        output = {
            "mip_map": mip_map,
        }
        if stress_history is not None:
            output["stress_history"] = stress_history
        if vz_history is not None:
            output["vz_history"] = vz_history

        return output

    def get_surface_field(self, field: torch.Tensor, z_index: int = 1) -> torch.Tensor:
        """Return a surface slice at given z index (near surface)."""
        return field[:, :, :, z_index]


if __name__ == "__main__":
    physics = ViscoelasticWave3D(device=torch.device("cpu"))
    traj = torch.stack([torch.full((10,), 0.05), torch.full((10,), 0.05)], dim=-1)
    output = physics(traj)
    print(output["stress_history"].shape)
