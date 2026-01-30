# -*- coding: utf-8 -*-
"""
GPU-optimized Mamba layer with chunked parallel selective scan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try importing official mamba_ssm library
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# Check if PyTorch supports torch.compile
HAS_COMPILE = hasattr(torch, 'compile')


def _selective_scan_loop(
    u: torch.Tensor,      # [B, L, d_inner]
    delta: torch.Tensor,  # [B, L, d_inner]
    A: torch.Tensor,      # [d_inner, d_state]
    B: torch.Tensor,      # [B, L, d_state]
    C: torch.Tensor,      # [B, L, d_state]
    D: torch.Tensor,      # [d_inner]
    d_state: int,
    initial_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-optimized selective scan loop.

    Optimization: avoids creating large 4D tensors by computing per-timestep.
    """
    B_size, L, d_inner = u.shape
    device = u.device
    dtype = u.dtype

    # Initialize state
    if initial_state is None:
        state = torch.zeros(B_size, d_inner, d_state, device=device, dtype=dtype)
    else:
        state = initial_state.contiguous()

    # Pre-allocate output
    outputs = torch.empty(B_size, L, d_inner, device=device, dtype=dtype)

    # Expand A to avoid repeated operations
    A_expanded = A.unsqueeze(0)

    # Optimized loop - compute per timestep
    for i in range(L):
        # Intermediate computations without gradient storage
        with torch.no_grad():
            delta_i = delta[:, i].unsqueeze(-1)
            B_i = B[:, i].unsqueeze(1)
            u_i = u[:, i].unsqueeze(-1)

            # Discretization parameters
            dA_i = torch.exp(delta_i * A_expanded)
            dB_i = delta_i * B_i

        # State update (keep gradient)
        state = dA_i * state + dB_i * u_i

        # Output (keep gradient)
        y_from_state = torch.bmm(state, C[:, i].unsqueeze(-1)).squeeze(-1)
        outputs[:, i] = y_from_state + D * u[:, i]

    return outputs, state


# Compile optimized scan function if supported
_selective_scan_compiled = _selective_scan_loop  # Default: non-compiled

if HAS_COMPILE:
    try:
        # Disabled by default to avoid triton dependency issues
        # Set PGMNO_ENABLE_COMPILE=1 to enable
        import os
        if os.environ.get('PGMNO_ENABLE_COMPILE', '0') == '1':
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            _selective_scan_compiled = torch.compile(_selective_scan_loop, mode='reduce-overhead')
            print("[FastMamba] torch.compile enabled")
    except Exception as e:
        print(f"[FastMamba] torch.compile disabled: {e}")
        _selective_scan_compiled = _selective_scan_loop


class FastParallelSSM(nn.Module):
    """GPU-optimized parallel state space model with chunked scanning."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.chunk_size = chunk_size
        self.dt_rank = math.ceil(d_model / 16)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D causal convolution (depthwise)
        # Manual implementation to avoid cuDNN errors
        self.d_conv = d_conv
        self.conv_weight = nn.Parameter(torch.randn(self.d_inner, 1, d_conv) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))

        # Selective parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State matrix A (HiPPO initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.1)

        # dt_proj special initialization
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_bias.log())

    def _parallel_scan_chunk(
        self,
        u: torch.Tensor,      # [B, chunk_size, d_inner]
        delta: torch.Tensor,  # [B, chunk_size, d_inner]
        A: torch.Tensor,      # [d_inner, d_state]
        B: torch.Tensor,      # [B, chunk_size, d_state]
        C: torch.Tensor,      # [B, chunk_size, d_state]
        D: torch.Tensor,      # [d_inner]
        initial_state: Optional[torch.Tensor] = None  # [B, d_inner, d_state]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked parallel scan using compiled function."""
        return _selective_scan_compiled(
            u, delta, A, B, C, D, self.d_state, initial_state
        )

    def _fast_parallel_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """Fast parallel scan with chunked processing for long sequences."""
        B_size, L, d_inner = u.shape
        chunk_size = min(self.chunk_size, L)

        if L <= chunk_size:
            # Short sequence: direct processing
            output, _ = self._parallel_scan_chunk(u, delta, A, B, C, D)
            return output

        # Long sequence: chunked processing
        outputs = []
        state = None
        n_chunks = (L + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, L)

            chunk_output, state = self._parallel_scan_chunk(
                u[:, start:end],
                delta[:, start:end],
                A, B[:, start:end], C[:, start:end], D,
                initial_state=state
            )
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B_size, L, _ = x.shape

        # 1. Input projection
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # 2. Causal convolution (manual to avoid cuDNN issues)
        x_conv = x_proj.transpose(1, 2).contiguous()
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0), mode='constant', value=0)
        x_conv = F.conv1d(x_conv, self.conv_weight, self.conv_bias, groups=self.d_inner)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_conv = F.silu(x_conv)

        # 3. Compute selective parameters
        x_dbl = self.x_proj(x_conv)
        dt = self.dt_proj(x_dbl[..., :self.dt_rank])
        dt = F.softplus(dt)

        B_param = x_dbl[..., self.dt_rank:self.dt_rank + self.d_state]
        C_param = x_dbl[..., self.dt_rank + self.d_state:]

        # 4. Get A matrix
        A = -torch.exp(self.A_log)

        # 5. Parallel selective scan
        y = self._fast_parallel_scan(x_conv, dt, A, B_param, C_param, self.D)

        # 6. Gating
        y = y * F.silu(z)

        # 7. Output projection
        return self.out_proj(y)


class FastMambaLayer(nn.Module):
    """GPU-optimized Mamba layer with automatic implementation selection."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 64,
        use_official: Optional[bool] = None
    ):
        super().__init__()

        # Decide which implementation to use
        if use_official is None:
            use_official = HAS_MAMBA

        self.use_official = use_official and HAS_MAMBA

        if self.use_official:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            print(f"[FastMamba] Using official mamba_ssm (CUDA optimized)")
        else:
            self.mamba = FastParallelSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                chunk_size=chunk_size
            )
            print(f"[FastMamba] Using FastParallelSSM (chunk_size={chunk_size})")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x))


# ============================================================================
# Simplified efficient SSM (for quick experiments)
# ============================================================================

class SimpleEfficientSSM(nn.Module):
    """Simplified SSM using conv approximation, suitable for short sequences."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Learnable conv kernel instead of SSM
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)

        # Time mixing
        self.time_mix = nn.Parameter(torch.ones(1, 1, self.d_inner) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Projection
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)

        # Time mixing
        x_mix = self.time_mix * x_conv + (1 - self.time_mix) * x_in

        # Gating
        y = F.silu(x_mix) * F.silu(z)

        return self.out_proj(y)


# Backward compatibility
MambaLayer = FastMambaLayer
