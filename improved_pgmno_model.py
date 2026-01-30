# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Prefer GPU-optimized FastMambaLayer
try:
    from fast_mamba_layer import FastMambaLayer as MambaLayer
    print("[Model] Using FastMambaLayer (GPU optimized)")
except ImportError:
    from improved_mamba_layer import MambaLayer
    print("[Model] Using standard MambaLayer")


class LatentMambaOperatorV2(nn.Module):
    """Latent Mamba operator with improved initialization and normalization."""
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_dim,
        d_model,
        num_layers,
        activation='gelu',
        use_layer_norm=True,
        dropout=0.1,
        gradient_clip=1.0
    ):
        super().__init__()

        self.gradient_clip = gradient_clip

        # 1. Lifting Operator P
        self.lifting = nn.Linear(in_channels, d_model // 2)

        # 2. Encoder E
        self.encoder = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.BatchNorm1d(d_model) if not use_layer_norm else nn.LayerNorm(d_model),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # 3. Latent Layers L (Mamba Blocks)
        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model) for _ in range(num_layers)
        ])

        # Add LayerNorm after each layer
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # 4. Decoder D
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            self._get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # 5. Projection Operator Q
        self.projection = nn.Linear(d_model // 2, out_channels)

        # 初始化权重
        self._initialize_weights()

    def _get_activation(self, activation):
        if activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'swish':
            return nn.SiLU()
        else:
            return nn.GELU()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
            # Xavier initialization for GELU activation
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [Batch, N_grid, Channels]
        batch_size, seq_len, _ = x.shape

        # P: Lifting
        v = self.lifting(x)  # [Batch, N_grid, d_model//2]
        v = F.layer_norm(v, v.shape[-1:])  # 预归一化

        # E: Encoding - reshape for BatchNorm1d, then reshape back
        v_flat = v.view(-1, v.size(-1))  # [Batch*N_grid, d_model//2]
        z_flat = self.encoder(v_flat)  # [Batch*N_grid, d_model]
        z = z_flat.view(batch_size, seq_len, -1)  # [Batch, N_grid, d_model]

        # L: Latent Integration - improved residual connection
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Pre-LN architecture
            z_norm = norm(z)
            layer_out = layer(z_norm)

            # Progressive residual connection
            if i == 0:
                z = z + layer_out
            else:
                z = z + 0.1 * layer_out

            # Gradient clipping
            if self.gradient_clip > 0:
                z = torch.clamp(z, -self.gradient_clip, self.gradient_clip)

        # D: Decoding - reshape for processing, then reshape back
        z_flat = z.view(-1, z.size(-1))  # [Batch*N_grid, d_model]
        v_out_flat = self.decoder(z_flat)  # [Batch*N_grid, d_model//2]
        v_out = v_out_flat.view(batch_size, seq_len, -1)  # [Batch, N_grid, d_model//2]

        # Q: Projection
        u = self.projection(v_out)

        return u


class PGMNOV2(nn.Module):
    """Physics-Guided Latent Mamba Operator with BDF multi-step integration."""
    def __init__(
        self,
        k_steps,
        dt,
        spatial_dim,
        hidden_dim=64,
        num_layers=4,
        init_lambda_scale=1.0,
        init_delta_scale=0.1,
        use_regularization=True,
        reg_weight=1e-4,
        boundary_type='periodic'  # 边界类型: 'periodic' 或 'dirichlet'
    ):
        super().__init__()
        self.k_steps = k_steps
        self.dt = dt
        self.spatial_dim = spatial_dim
        self.boundary_type = boundary_type
        self.use_regularization = use_regularization
        self.reg_weight = reg_weight

        # BDF coefficients - fixed buffers (not learnable)
        # Corresponds to paper: "where α_j and β_j are fixed coefficients"
        if k_steps == 1:
            # BDF1 (Backward Euler): u_{n+1} = u_n + dt f(u_{n+1})
            # Coeffs for [u_n]: [1.0]
            self.register_buffer('lambdas', torch.tensor([1.0]))
            self.register_buffer('deltas', torch.tensor([1.0]))
        elif k_steps == 2:
            # BDF2: u_{n+1} = 4/3 u_n - 1/3 u_{n-1} + 2/3 dt f(u)
            # Coeffs for [u_{n-1}, u_n]: [-1/3, 4/3]
            self.register_buffer('lambdas', torch.tensor([-1.0/3.0, 4.0/3.0]))
            self.register_buffer('deltas', torch.tensor([0.0, 2.0/3.0]))
        elif k_steps == 3:
            # BDF3: u_{n+1} = 18/11 u_n - 9/11 u_{n-1} + 2/11 u_{n-2} + 6/11 dt f(u)
            # Coeffs for [u_{n-2}, u_{n-1}, u_n]: [2/11, -9/11, 18/11]
            self.register_buffer('lambdas', torch.tensor([2.0/11.0, -9.0/11.0, 18.0/11.0]))
            self.register_buffer('deltas', torch.tensor([0.0, 0.0, 6.0/11.0]))
        elif k_steps == 4:
            # BDF4: u_{n+1} = 48/25 u_n - 36/25 u_{n-1} + 16/25 u_{n-2} - 3/25 u_{n-3} + 12/25 dt f(u)
            # Coeffs for [u_{n-3}, u_{n-2}, u_{n-1}, u_n]: [-3/25, 16/25, -36/25, 48/25]
            self.register_buffer('lambdas', torch.tensor([-3.0/25.0, 16.0/25.0, -36.0/25.0, 48.0/25.0]))
            self.register_buffer('deltas', torch.tensor([0.0, 0.0, 0.0, 12.0/25.0]))
        elif k_steps == 5:
            # BDF5: u_{n+1} = 300/137 u_n - 300/137 u_{n-1} + 200/137 u_{n-2} - 75/137 u_{n-3} + 12/137 u_{n-4} + 60/137 dt f(u)
            # Coeffs for [u_{n-4}, u_{n-3}, u_{n-2}, u_{n-1}, u_n]: [12/137, -75/137, 200/137, -300/137, 300/137]
            self.register_buffer('lambdas', torch.tensor([12.0/137.0, -75.0/137.0, 200.0/137.0, -300.0/137.0, 300.0/137.0]))
            self.register_buffer('deltas', torch.tensor([0.0, 0.0, 0.0, 0.0, 60.0/137.0]))

        # Learnable decay factor
        self.decay_factor = nn.Parameter(torch.tensor(0.95))

        # Boundary condition enforcement - u(x) = M(x) G(u)(x) + (1-M(x)) g(x)
        self.boundary_mask = None
        self.boundary_values = None
        if boundary_type == 'dirichlet':
            # Dirichlet: fixed to 0 at boundaries
            mask = torch.ones(spatial_dim)
            mask[0] = mask[-1] = 0.0
            self.register_buffer('boundary_mask', mask.unsqueeze(0))
            self.register_buffer('boundary_values', torch.zeros(1, spatial_dim))
        # Periodic boundary handled via FFT/circular padding

        # Core Latent Mamba operator
        self.operator = LatentMambaOperatorV2(
            in_channels=2,
            out_channels=1,
            latent_dim=hidden_dim,
            d_model=hidden_dim,
            num_layers=num_layers,
            activation='gelu',
            use_layer_norm=True,
            dropout=0.1
        )

    def forward(self, past_states, grid_coords):
        """
        Args:
            past_states: [Batch, k_steps, N_grid] - 过去 k 个时刻的解
            grid_coords: [Batch, N_grid, 1]       - 空间网格坐标

        Returns:
            u_next: [Batch, N_grid] - 预测的下一时刻解
            reg_loss: Optional[float] - 正则化损失
        """
        batch_size, k, n_grid = past_states.shape
        device = past_states.device

        # Initialize accumulator
        u_next_accum = torch.zeros(batch_size, n_grid, device=device)

        # Compute decay weights (recomputed each forward to avoid stale graph)
        decay_weights = torch.pow(
            self.decay_factor, torch.arange(k-1, -1, -1, device=device)
        )

        for j in range(k):
            # Get history state at step j
            u_curr = past_states[:, j, :].unsqueeze(-1)

            # Concatenate coordinate information
            inp = torch.cat([u_curr, grid_coords], dim=-1)

            # Apply neural operator
            g_out = self.operator(inp).squeeze(-1)

            # Weighted computation with decay factor
            effective_lambda = self.lambdas[j] * decay_weights[j]
            effective_delta = self.deltas[j] * decay_weights[j]

            # Compute increment term
            term = effective_lambda * past_states[:, j, :] + \
                   self.dt * effective_delta * g_out

            u_next_accum = u_next_accum + term

        # Compute regularization loss
        reg_loss = None
        if self.use_regularization:
            # L2 regularization on learnable decay_factor
            reg_loss = self.reg_weight * (self.decay_factor - 0.95)**2

        # Boundary condition enforcement
        if self.boundary_type == 'dirichlet' and self.boundary_mask is not None:
            batch_mask = self.boundary_mask.expand(batch_size, -1).to(device)
            batch_values = self.boundary_values.expand(batch_size, -1).to(device)
            u_next_accum = batch_mask * u_next_accum + (1 - batch_mask) * batch_values

        return u_next_accum, reg_loss

    def predict_multi_step(self, initial_history, grid_coords, n_steps):
        """
        Multi-step prediction with optimized memory usage.

        Args:
            initial_history: [Batch, k_steps, N_grid] - initial history
            grid_coords: [Batch, N_grid, 1] - grid coordinates
            n_steps: int - number of prediction steps

        Returns:
            predictions: [Batch, n_steps, N_grid] - predicted sequence
        """
        batch_size, k, n_grid = initial_history.shape
        device = initial_history.device

        # Pre-allocate output tensor
        predictions = torch.zeros(batch_size, n_steps, n_grid, device=device)
        current_history = initial_history.clone()

        for step in range(n_steps):
            u_next, _ = self.forward(current_history, grid_coords)

            # Store directly to pre-allocated tensor
            predictions[:, step, :] = u_next

            # Update history window using roll for efficiency
            current_history = torch.roll(current_history, shifts=-1, dims=1)
            current_history[:, -1, :] = u_next

        return predictions

    def get_effective_coefficients(self):
        """获取有效的系数值，用于分析"""
        decay_weights = torch.pow(self.decay_factor, torch.arange(self.k_steps-1, -1, -1, device=self.decay_factor.device))
        effective_lambdas = self.lambdas * decay_weights
        effective_deltas = self.deltas * decay_weights

        return {
            'raw_lambdas': self.lambdas.detach().cpu().numpy(),
            'raw_deltas': self.deltas.detach().cpu().numpy(),
            'effective_lambdas': effective_lambdas.detach().cpu().numpy(),
            'effective_deltas': effective_deltas.detach().cpu().numpy(),
            'decay_factor': self.decay_factor.item()
        }