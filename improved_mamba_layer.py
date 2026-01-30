# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Try importing official mamba_ssm library
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("[Info] Successfully imported official mamba_ssm library.")
except ImportError:
    HAS_MAMBA = False
    print("[Info] mamba_ssm not found. Using improved PyTorch implementation (ImprovedMockMamba).")


class ImprovedMockMamba(nn.Module):
    """Selective state space model with HiPPO initialization."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_bias=True,
        conv_bias=True,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # 1. Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=use_bias)

        # 2. Causal convolution along sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            bias=conv_bias,
            groups=self.d_inner  # depthwise convolution
        )

        # 3. Selective parameters
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False
        )

        # 4. dt projection - from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 5. State transition matrix A (HiPPO initialization)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # 6. Discretization parameters D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 7. Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        # 8. Layer Normalization
        self.dt_layernorm = nn.LayerNorm(self.dt_rank)
        self.B_layernorm = nn.LayerNorm(self.d_state)
        self.C_layernorm = nn.LayerNorm(self.d_state)

        # 9. Initialize weights
        self._initialize_weights(dt_min, dt_max, dt_init)

    def _initialize_weights(self, dt_min, dt_max, dt_init):
        """Initialize weights using Xavier initialization."""
        # Initialize projection layers
        nn.init.xavier_uniform_(self.in_proj.weight, gain=2 ** 0.5)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        # Initialize convolution layer
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='linear')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize x_proj
        nn.init.xavier_uniform_(self.x_proj.weight, gain=1e-2)
        if self.x_proj.bias is not None:
            nn.init.zeros_(self.x_proj.bias)

        # Initialize A_log
        with torch.no_grad():
            self.A_log.fill_(0.0)

        # Initialize D
        nn.init.constant_(self.D, 0.0)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Initialize dt projection
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=1.0)
        if self.dt_proj.bias is not None:
            nn.init.zeros_(self.dt_proj.bias)

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan algorithm with parallel-like implementation.

        Args:
            u: [B, L, d_inner] - input sequence
            delta: [B, L, d_inner] - discretization step
            A: [1, 1, d_inner, d_state] - state transition matrix
            B: [B, L, d_state] - input projection
            C: [B, L, d_state] - output projection
            D: [d_inner] - skip connection
        """
        batch_size, L, d_inner = u.shape
        d_state = self.d_state

        # A matrix: [d_inner, d_state]
        A_matrix = A.squeeze(0).squeeze(0)

        # Initialize state: [batch, d_inner, d_state]
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for i in range(L):
            delta_i = delta[:, i, :]

            # Discretize A: dA = exp(delta * A)
            dA = torch.exp(delta_i.unsqueeze(-1) * A_matrix.unsqueeze(0))

            # Discretize B: dB = delta * B
            B_i = B[:, i, :]
            dB = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)

            u_i = u[:, i, :]

            # State update: x = dA * x + dB * u
            x = dA * x + dB * u_i.unsqueeze(-1)

            # Output: y = (x @ C) + D * u
            C_i = C[:, i, :]
            y = torch.einsum('bds,bs->bd', x, C_i)
            y = y + D * u_i

            outputs.append(y)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        """Forward pass with Pre-LN architecture."""
        batch_size, seq_len, _ = x.shape

        # 1. Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 2. 1D convolution (causal)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # 3. Activation
        x = F.silu(x)

        # 4. Compute selective parameters
        x_proj = self.x_proj(x)

        # Split parameters
        dt = self.dt_layernorm(x_proj[..., :self.dt_rank])
        dt = self.dt_proj(dt)

        B = self.B_layernorm(x_proj[..., self.dt_rank:self.dt_rank + self.d_state])
        C = self.C_layernorm(x_proj[..., self.dt_rank + self.d_state:])

        # Apply activation
        dt = F.softplus(dt)
        B = F.silu(B)
        C = F.silu(C)

        # 5. Selective scan
        A = torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)

        u = x
        delta = dt
        y = self.selective_scan(u, delta, A, B, C, self.D)

        # 6. Gating
        y = y * F.silu(z)

        # 7. Output projection
        output = self.out_proj(y)

        return output


class MambaLayerV2(nn.Module):
    """Mamba layer wrapper with optional official implementation."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_official_mamba=None,
        layer_norm_position="pre"
    ):
        super().__init__()

        # Decide implementation
        if use_official_mamba is None:
            use_official_mamba = HAS_MAMBA

        self.use_official = use_official_mamba and HAS_MAMBA
        self.layer_norm_position = layer_norm_position

        # Selective scan layer
        if self.use_official:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba = ImprovedMockMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )

        # Add LayerNorm
        if layer_norm_position == "pre":
            self.norm = nn.LayerNorm(d_model)
        elif layer_norm_position == "post":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()

        # Performance monitoring
        self.forward_count = 0
        self.total_time = 0

    def forward(self, x):
        """Forward pass with pre/post normalization."""
        import time
        start_time = time.time()

        if self.layer_norm_position == "pre":
            x = self.norm(x)
            out = self.mamba(x)
        elif self.layer_norm_position == "post":
            out = self.mamba(x)
            out = self.norm(out)
        else:
            out = self.mamba(x)

        # Update performance stats
        self.forward_count += 1
        self.total_time += time.time() - start_time

        return out

    def get_performance_stats(self):
        """Get performance statistics."""
        if self.forward_count > 0:
            avg_time = self.total_time / self.forward_count
        else:
            avg_time = 0

        return {
            'implementation': 'official' if self.use_official else 'mock',
            'forward_count': self.forward_count,
            'total_time': self.total_time,
            'avg_time_per_forward': avg_time
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.forward_count = 0
        self.total_time = 0


# Backward compatibility
MambaLayer = MambaLayerV2