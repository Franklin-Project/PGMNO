# -*- coding: utf-8 -*-
"""
GPU-accelerated Burgers equation data generator using PyTorch FFT.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GPUBurgersGenerator:
    """
    GPU-accelerated Burgers equation solver using spectral method.

    All computations are performed on GPU using PyTorch FFT.
    """

    def __init__(
        self,
        n_grid: int = 64,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        nu: float = 0.01 / np.pi,
        device: Optional[torch.device] = None
    ):
        self.n_grid = n_grid
        self.x_range = x_range
        self.nu = nu

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Pre-compute grid and wavenumbers on device
        x_min, x_max = x_range
        self.dx = (x_max - x_min) / n_grid
        self.x = torch.linspace(x_min, x_max, n_grid, device=self.device)

        # Wavenumbers for spectral derivatives
        self.k = 2 * np.pi * torch.fft.fftfreq(n_grid, d=self.dx, device=self.device)

        # Dealiasing mask (2/3 rule)
        k_max = torch.max(torch.abs(self.k))
        self.dealias_mask = (torch.abs(self.k) <= (2.0/3.0) * k_max).float()

    def generate_initial_conditions(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate diverse initial conditions on GPU.

        Returns:
            u0: [n_samples, n_grid]
        """
        if seed is not None:
            torch.manual_seed(seed)

        x = self.x.unsqueeze(0).expand(n_samples, -1)  # [n_samples, n_grid]

        # Mix of different initial condition types
        u0 = torch.zeros(n_samples, self.n_grid, device=self.device)

        # Random parameters for each sample
        modes = torch.randint(0, 4, (n_samples,), device=self.device)

        # Mode 0: Gaussian
        mask_gaussian = (modes == 0).float().unsqueeze(1)
        centers = torch.rand(n_samples, 1, device=self.device) - 0.5
        widths = torch.rand(n_samples, 1, device=self.device) * 0.2 + 0.1
        amplitudes = torch.rand(n_samples, 1, device=self.device) * 1.5 + 0.5
        gaussian = amplitudes * torch.exp(-((x - centers)**2) / (2 * widths**2))
        u0 += mask_gaussian * gaussian

        # Mode 1: Sine waves
        mask_sine = (modes == 1).float().unsqueeze(1)
        freqs = torch.randint(1, 4, (n_samples, 1), device=self.device).float()
        phases = torch.rand(n_samples, 1, device=self.device) * 2 * np.pi
        sine_amps = torch.rand(n_samples, 1, device=self.device) + 0.5
        sine_wave = sine_amps * torch.sin(freqs * np.pi * x + phases)
        u0 += mask_sine * sine_wave

        # Mode 2: Combined
        mask_combined = (modes == 2).float().unsqueeze(1)
        combined = (
            torch.exp(-((x - torch.rand(n_samples, 1, device=self.device) * 0.5)**2) / 0.1) +
            0.5 * torch.sin(np.pi * x) +
            0.3 * torch.sin(2 * np.pi * x + torch.rand(n_samples, 1, device=self.device) * np.pi)
        )
        u0 += mask_combined * combined

        # Mode 3: Random Fourier modes
        mask_random = (modes == 3).float().unsqueeze(1)
        random_field = torch.zeros(n_samples, self.n_grid, device=self.device)
        for k in range(1, 6):
            amp = torch.randn(n_samples, 1, device=self.device) / k
            phase = torch.rand(n_samples, 1, device=self.device) * 2 * np.pi
            random_field += amp * torch.sin(k * np.pi * x + phase)
        u0 += mask_random * random_field

        return u0

    @torch.no_grad()
    def solve_batch(
        self,
        u0: torch.Tensor,
        nt: int,
        dt: float,
        return_all_steps: bool = True
    ) -> torch.Tensor:
        """
        Solve Burgers equation for a batch of initial conditions.

        Uses operator splitting:
        1. Exact linear diffusion step in Fourier space
        2. Pseudo-spectral nonlinear convection step

        Args:
            u0: Initial conditions [batch, n_grid]
            nt: Number of time steps
            dt: Time step size
            return_all_steps: If True, return [batch, nt+1, n_grid]

        Returns:
            Solution trajectory or final state
        """
        batch_size = u0.shape[0]
        u = u0.clone()

        # Pre-compute diffusion operator
        diffusion_factor = torch.exp(-self.nu * self.k**2 * dt)

        if return_all_steps:
            trajectory = [u.clone()]

        # Transform to Fourier space
        ut = torch.fft.fft(u, dim=-1)

        for t in range(nt):
            # Step 1: Linear diffusion (exact in Fourier space)
            ut = ut * diffusion_factor.unsqueeze(0)

            # Step 2: Nonlinear convection (pseudo-spectral)
            # Apply dealiasing
            ut_dealiased = ut * self.dealias_mask.unsqueeze(0)
            u_space = torch.fft.ifft(ut_dealiased, dim=-1).real

            # Compute nonlinear term: -d(u^2/2)/dx
            u_sq = 0.5 * u_space**2
            u_sq_fft = torch.fft.fft(u_sq, dim=-1)
            u_sq_x = torch.fft.ifft(1j * self.k.unsqueeze(0) * u_sq_fft, dim=-1).real

            # Forward Euler for nonlinear term
            u_new = u_space - dt * u_sq_x

            # Stability clipping
            u_new = torch.clamp(u_new, -1e3, 1e3)

            # Back to Fourier space
            ut = torch.fft.fft(u_new, dim=-1)

            if return_all_steps:
                trajectory.append(u_new.clone())

        if return_all_steps:
            return torch.stack(trajectory, dim=1)  # [batch, nt+1, n_grid]
        else:
            return torch.fft.ifft(ut, dim=-1).real

    def generate_dataset(
        self,
        n_samples: int,
        nt: int,
        dt: float,
        seed: Optional[int] = None,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a full dataset of Burgers equation trajectories.

        Uses batching to handle large datasets efficiently.

        Args:
            n_samples: Number of samples
            nt: Number of time steps
            dt: Time step size
            seed: Random seed
            batch_size: Batch size for generation

        Returns:
            data: [n_samples, nt+1, n_grid]
            x_grid: [n_grid]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Check CFL condition
        self._check_cfl(dt)

        all_trajectories = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            current_batch_size = end - start

            # Generate initial conditions
            u0 = self.generate_initial_conditions(
                current_batch_size,
                seed=seed + i if seed is not None else None
            )

            # Solve
            trajectory = self.solve_batch(u0, nt, dt, return_all_steps=True)
            all_trajectories.append(trajectory)

            if (i + 1) % max(1, n_batches // 4) == 0:
                print(f"[GPU Data] Generated {end}/{n_samples} samples")

        data = torch.cat(all_trajectories, dim=0)
        return data, self.x

    def _check_cfl(self, dt: float):
        """Check CFL stability condition"""
        # Estimate max velocity from typical initial conditions
        max_u_estimate = 2.0  # Conservative estimate
        cfl = max_u_estimate * dt / self.dx
        if cfl > 0.5:
            recommended_dt = 0.4 * self.dx / max_u_estimate
            print(f"[Warning] CFL={cfl:.2f} > 0.5. Recommended dt <= {recommended_dt:.4f}")


def generate_burgers_data_gpu(
    n_samples: int = 100,
    n_grid: int = 64,
    nt: int = 50,
    dt: float = 0.01,
    nu: float = 0.01 / np.pi,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for GPU-accelerated Burgers data generation.

    Drop-in replacement for generate_burgers_data_v2 with GPU acceleration.

    Returns:
        data: [n_samples, nt+1, n_grid] on CPU
        x_grid: [n_grid] on CPU
    """
    generator = GPUBurgersGenerator(
        n_grid=n_grid,
        nu=nu,
        device=device
    )

    # Adjust dt if needed for stability
    max_dt = 0.4 * generator.dx / 2.0  # Conservative CFL
    if dt > max_dt:
        print(f"[GPU Data] Adjusting dt from {dt:.4f} to {max_dt:.4f} for stability")
        dt = max_dt

    data, x_grid = generator.generate_dataset(
        n_samples=n_samples,
        nt=nt,
        dt=dt,
        seed=seed
    )

    # Return on CPU for compatibility
    return data.cpu(), x_grid.cpu()


# Backward compatibility
def generate_burgers_data_v2_gpu(*args, **kwargs):
    """Alias for generate_burgers_data_gpu"""
    return generate_burgers_data_gpu(*args, **kwargs)
