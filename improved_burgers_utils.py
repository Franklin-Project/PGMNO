# -*- coding: utf-8 -*-
import torch
import numpy as np
import warnings
from typing import Tuple, Optional
from scipy.fftpack import fft, ifft, fftfreq


def generate_burgers_data_v2(
    n_samples=100,
    n_grid=256,
    nt=100,
    dt=0.01,
    nu=0.01/np.pi,
    x_range=(-1, 1),
    noise_level=0.0,
    seed=None,
    method='spectral'
):
    """
    Generate Burgers equation data using spectral or finite difference methods.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Spatial grid
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_grid, endpoint=False)
    dx = x[1] - x[0]

    data = []
    print(f"[Data] Generating {n_samples} trajectories using {method} method...")

    for sample_idx in range(n_samples):
        # Generate initial condition
        u0 = _generate_initial_condition(x, sample_idx, noise_level)

        if method == 'spectral':
            u_traj = _solve_burgers_spectral(u0, nt, dt, nu, dx)
        elif method == 'finite_difference':
            u_traj = _solve_burgers_finite_difference(u0, nt, dt, nu, dx)
        else:
            raise ValueError(f"Unknown method: {method}")

        data.append(np.array(u_traj))

        if (sample_idx + 1) % 10 == 0:
            print(f"[Data] Generated {sample_idx + 1}/{n_samples} samples")

    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


def _generate_initial_condition(x, seed, noise_level=0.0):
    """Generate diverse initial conditions."""
    np.random.seed(seed)

    # Multiple initial condition modes
    mode = np.random.choice(['gaussian', 'sine', 'combined', 'random'])

    if mode == 'gaussian':
        # Gaussian wave packet
        center = np.random.uniform(-0.5, 0.5)
        width = np.random.uniform(0.1, 0.3)
        amplitude = np.random.uniform(0.5, 2.0)
        u0 = amplitude * np.exp(-((x - center)**2) / (2 * width**2))

    elif mode == 'sine':
        # Sine wave combination
        freq = np.random.randint(1, 4)
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        u0 = amplitude * np.sin(freq * np.pi * x + phase)

    elif mode == 'combined':
        # Combined mode
        u0 = np.exp(-((x - np.random.uniform(-0.5, 0.5))**2) / 0.1)
        u0 += 0.5 * np.sin(np.pi * x)
        u0 += 0.3 * np.sin(2 * np.pi * x + np.random.uniform(0, np.pi))

    else:  # random
        # Random smooth field
        n_modes = 5
        u0 = np.zeros_like(x)
        for k in range(1, n_modes + 1):
            amplitude = np.random.randn() / k
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amplitude * np.sin(k * np.pi * x + phase)

    # Add noise
    if noise_level > 0:
        u0 += noise_level * np.random.randn(*u0.shape)

    return u0


def _solve_burgers_spectral(u0, nt, dt, nu, dx):
    """Solve Burgers equation using spectral method."""
    n_grid = len(u0)
    u = u0.copy()
    u_traj = [u.copy()]

    # Wavenumbers
    k = 2 * np.pi * fftfreq(n_grid, d=dx)

    # Transform to frequency domain
    ut = fft(u)

    # CFL stability check
    max_u = np.max(np.abs(u0))
    cfl = max_u * dt / dx
    if cfl > 0.5:
        print(f"[Warning] CFL={cfl:.2f} > 0.5, may be unstable")

    for t in range(nt):
        # Linear diffusion step (exact in Fourier space)
        ut = ut * np.exp(-nu * k**2 * dt)

        # Nonlinear advection step (pseudo-spectral with dealiasing)
        # 2/3 dealiasing rule
        k_dealias = (np.abs(k) <= (2.0/3.0) * np.max(np.abs(k)))
        u_space = np.real(ifft(ut * k_dealias))

        # Numerical stability check
        if np.any(np.isnan(u_space)) or np.any(np.abs(u_space) > 1e6):
            print(f"[Warning] Numerical instability at t={t}, using last stable state")
            break

        # Compute nonlinear term with clipping
        u_space_clipped = np.clip(u_space, -1e3, 1e3)
        u_sq = 0.5 * u_space_clipped**2
        u_sq_x = np.real(ifft(1j * k * fft(u_sq)))

        # Time integration (forward Euler)
        u_new = u_space - dt * u_sq_x

        # Clip to prevent explosion
        u_new = np.clip(u_new, -1e3, 1e3)

        ut = fft(u_new)
        u_traj.append(u_new.copy())

    return u_traj


def _solve_burgers_finite_difference(u0, nt, dt, nu, dx):
    """Solve Burgers equation using finite difference method."""
    n_grid = len(u0)
    u = u0.copy()
    u_traj = [u]

    for t in range(nt):
        u_new = u.copy()

        # Central difference with periodic boundary
        for i in range(n_grid):
            i_prev = (i - 1) % n_grid
            i_next = (i + 1) % n_grid

            # Spatial derivatives
            u_x = (u[i_next] - u[i_prev]) / (2 * dx)
            u_xx = (u[i_next] - 2*u[i] + u[i_prev]) / (dx**2)

            # Burgers: u_t = -u*u_x + nu*u_xx
            du_dt = -u[i] * u_x + nu * u_xx

            # Forward Euler
            u_new[i] = u[i] + dt * du_dt

        u = u_new
        u_traj.append(u)

    return u_traj


def physics_loss_burgers_v2(
    u_pred: torch.Tensor,
    u_history: torch.Tensor,
    dt: float,
    dx: float,
    nu: float,
    method: str = 'spectral',
    loss_type: str = 'mse',
    return_components: bool = False
) -> torch.Tensor:
    """
    Improved Physics Loss with BDF-k Support
    
    Arguments:
        u_pred: Predicted state at time t [Batch, N]
        u_history: History states [Batch, k, N] containing [t-k, ..., t-1]
                   OR single state [Batch, N] for backward compatibility (BDF-1)
        dt: Time step
        dx: Spatial step
        nu: Viscosity
        method: Derivative calculation method
        loss_type: Loss type
        return_components: Whether to return components
    """
    # Check inputs
    if torch.isnan(u_pred).any() or torch.isnan(u_history).any():
        warnings.warn("NaN detected in inputs to physics loss")
        return torch.tensor(1e6, device=u_pred.device, dtype=u_pred.dtype)

    # Handle backward compatibility or history input
    if u_history.dim() == 2:
        # If passed as [Batch, N], treat as single previous step (BDF-1)
        # Reshape to [Batch, 1, N]
        history_states = u_history.unsqueeze(1)
    else:
        history_states = u_history

    batch_size, k_steps, n_grid = history_states.shape
    device = u_pred.device

    # --- Time Derivative Approximation (BDF) ---
    # Combine history and prediction: [t-k, ..., t-1, t]
    # Shape: [Batch, k_steps+1, N]
    all_states = torch.cat([history_states, u_pred.unsqueeze(1)], dim=1)
    
    # Determine BDF order based on available history
    # We can support up to BDF-5 if we have enough history
    # If k_steps=1, we have 2 points (t-1, t) -> BDF-1
    # If k_steps=2, we have 3 points (t-2, t-1, t) -> BDF-2
    # Extended to support BDF-4 and BDF-5 as claimed in Table 2
    bdf_order = k_steps  # Remove the min(k_steps, 3) limitation 
    
    if bdf_order == 1:
        # BDF-1: u_t - u_{t-1}
        # coeffs for [t-1, t]: [-1, 1]
        du_dt = (all_states[:, -1, :] - all_states[:, -2, :]) / dt
    elif bdf_order == 2:
        # BDF-2: 3/2 u_t - 2 u_{t-1} + 1/2 u_{t-2}
        # coeffs for [t-2, t-1, t]: [0.5, -2, 1.5]
        du_dt = (1.5 * all_states[:, -1, :] - 2.0 * all_states[:, -2, :] + 0.5 * all_states[:, -3, :]) / dt
    elif bdf_order == 3:
        # BDF-3: 11/6 u_t - 3 u_{t-1} + 3/2 u_{t-2} - 1/3 u_{t-3}
        # coeffs for [t-3, ..., t]: [-1/3, 1.5, -3, 11/6]
        coeffs = torch.tensor([-1/3, 1.5, -3.0, 11/6], device=device, dtype=u_pred.dtype)
        # Weighted sum along time dimension
        states_to_use = all_states[:, -4:, :] # last 4 states
        # [Batch, 4, N] * [4] -> need careful broadcasting or manual sum
        term1 = coeffs[0] * states_to_use[:, 0, :]
        term2 = coeffs[1] * states_to_use[:, 1, :]
        term3 = coeffs[2] * states_to_use[:, 2, :]
        term4 = coeffs[3] * states_to_use[:, 3, :]
        du_dt = (term1 + term2 + term3 + term4) / dt
    elif bdf_order == 4:
        # BDF-4: 25/12*u_t - 4*u_{t-1} + 3*u_{t-2} - 4/3*u_{t-3} + 1/4*u_{t-4}
        # coeffs for [t-4, t-3, t-2, t-1, t]: [1/4, -4/3, 3, -4, 25/12]
        coeffs = torch.tensor([0.25, -4.0/3.0, 3.0, -4.0, 25.0/12.0],
                              device=device, dtype=u_pred.dtype)
        states_to_use = all_states[:, -5:, :]  # last 5 states
        term1 = coeffs[0] * states_to_use[:, 0, :]
        term2 = coeffs[1] * states_to_use[:, 1, :]
        term3 = coeffs[2] * states_to_use[:, 2, :]
        term4 = coeffs[3] * states_to_use[:, 3, :]
        term5 = coeffs[4] * states_to_use[:, 4, :]
        du_dt = (term1 + term2 + term3 + term4 + term5) / dt
    elif bdf_order == 5:
        # BDF-5: 137/60*u_t - 5*u_{t-1} + 10/3*u_{t-2} - 5/4*u_{t-3} + 1/5*u_{t-4} - 1/6*u_{t-5}
        # Note: BDF-5 is unstable (beyond Dahlquist barrier) but included for completeness
        # coeffs for [t-5, t-4, t-3, t-2, t-1, t]: [-1/6, 1/5, -5/4, 10/3, -5, 137/60]
        coeffs = torch.tensor([-1.0/6.0, 1.0/5.0, -5.0/4.0, 10.0/3.0, -5.0, 137.0/60.0],
                              device=device, dtype=u_pred.dtype)
        states_to_use = all_states[:, -6:, :]  # last 6 states
        term1 = coeffs[0] * states_to_use[:, 0, :]
        term2 = coeffs[1] * states_to_use[:, 1, :]
        term3 = coeffs[2] * states_to_use[:, 2, :]
        term4 = coeffs[3] * states_to_use[:, 3, :]
        term5 = coeffs[4] * states_to_use[:, 4, :]
        term6 = coeffs[5] * states_to_use[:, 5, :]
        du_dt = (term1 + term2 + term3 + term4 + term5 + term6) / dt
    else:
        # Fallback to BDF-1 for any other k value
        du_dt = (u_pred - history_states[:, -1, :]) / dt

    # --- Spatial Derivatives & PDE RHS ---
    if method == 'spectral':
        ux, uxx = _compute_derivatives_spectral(u_pred, dx)
    elif method == 'finite_difference':
        ux, uxx = _compute_derivatives_finite_difference(u_pred, dx)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Burgers equation RHS: -u u_x + ν u_xx
    N_u = -u_pred * ux + nu * uxx

    # --- Residual ---
    # eqn: u_t = N(u)
    residual = du_dt - N_u

    # Clip for stability (more conservative clamping for gradient stability)
    residual = torch.clamp(residual, -100, 100)

    # --- Loss Calculation ---
    if loss_type == 'mse':
        loss = torch.mean(residual**2)
    elif loss_type == 'l1':
        loss = torch.mean(torch.abs(residual))
    elif loss_type == 'huber':
        loss = F.huber_loss(residual, torch.zeros_like(residual))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if return_components:
        components = {
            'time_derivative_sq': torch.mean(du_dt**2),
            'nonlinear_term': torch.mean((u_pred * ux)**2),
            'diffusion_term': torch.mean((nu * uxx)**2),
            'residual_std': torch.std(residual),
            'max_residual': torch.max(torch.abs(residual)),
            'bdf_order': bdf_order
        }
        return loss, components

    return loss


def _compute_derivatives_spectral(u: torch.Tensor, dx: float) -> tuple:
    """Compute spatial derivatives using spectral method."""
    batch_size, n_grid = u.shape
    device = u.device

    # FFT
    u_fft = torch.fft.fft(u, dim=-1)

    # Wavenumbers
    k = 2 * np.pi * torch.fft.fftfreq(n_grid, d=dx).to(device)

    # First derivative
    ux_fft = 1j * k * u_fft
    ux = torch.fft.ifft(ux_fft).real

    # Second derivative
    uxx_fft = -k**2 * u_fft
    uxx = torch.fft.ifft(uxx_fft).real

    return ux, uxx


def _compute_derivatives_finite_difference(u: torch.Tensor, dx: float) -> tuple:
    """Compute spatial derivatives using finite difference."""
    # Periodic boundary via torch.roll
    u_right = torch.roll(u, -1, dims=1)
    u_left = torch.roll(u, 1, dims=1)

    # First derivative (central difference)
    ux = (u_right - u_left) / (2 * dx)

    # Second derivative (central difference)
    uxx = (u_right - 2*u + u_left) / (dx**2)

    return ux, uxx


def compute_pde_metrics(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    dt: float,
    dx: float,
    nu: float
) -> dict:
    """Compute PDE-related evaluation metrics."""
    # L2 error
    l2_error = torch.mean((u_pred - u_true)**2)

    # L_inf error
    linf_error = torch.max(torch.abs(u_pred - u_true))

    # Relative error
    relative_error = l2_error / (torch.mean(u_true**2) + 1e-8)

    # Physics constraint violation
    _, components = physics_loss_burgers_v2(
        u_pred, u_true, dt, dx, nu, return_components=True
    )

    # Conservation check (mass should be conserved with periodic BC)
    mass_pred = torch.mean(u_pred, dim=1)
    mass_true = torch.mean(u_true, dim=1)
    mass_error = torch.mean((mass_pred - mass_true)**2)

    return {
        'l2_error': l2_error.item(),
        'linf_error': linf_error.item(),
        'relative_error': relative_error.item(),
        'mass_error': mass_error.item(),
        'residual_std': components['residual_std'].item(),
        'max_residual': components['max_residual'].item()
    }


def create_validation_dataset(
    n_val=20,
    n_grid=128,
    nt=100,
    dt=0.01,
    nu=0.01/np.pi,
    seed=42
):
    """Create validation dataset with challenging initial conditions."""
    print(f"[Data] Creating validation dataset with {n_val} challenging samples...")
    return generate_burgers_data_v2(
        n_samples=n_val,
        n_grid=n_grid,
        nt=nt,
        dt=dt,
        nu=nu,
        noise_level=0.01,  # 添加少量噪声
        seed=seed,
        method='spectral'
    )