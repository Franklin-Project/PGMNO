# -*- coding: utf-8 -*-
"""
Extended baseline backbones (AFNO, Transolver) for comprehensive comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# AFNO (Adaptive Fourier Neural Operator)
# ============================================================================

class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator Block
    
    论文: "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers"
    (FourCastNet使用的核心组件)
    
    核心思想:
    - 使用FFT进行全局token mixing (替代attention)
    - 在频域进行自适应过滤
    - 复杂度为O(N log N)
    
    Args:
        hidden_dim: 隐藏维度
        num_blocks: 频域block数量
        sparsity_threshold: 稀疏阈值 (软阈值参数)
        hard_thresholding_fraction: 硬阈值保留比例
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        
        # 频域权重 (可学习)
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, hidden_dim // num_blocks, hidden_dim // num_blocks))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, hidden_dim // num_blocks, hidden_dim // num_blocks))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, hidden_dim // num_blocks))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, hidden_dim // num_blocks))
        
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # MLP for channel mixing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            [batch, seq_len, hidden_dim]
        """
        residual = x
        x = self.norm(x)
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape for block-wise processing
        # [batch, seq_len, num_blocks, block_size]
        block_size = hidden_dim // self.num_blocks
        x = x.view(batch_size, seq_len, self.num_blocks, block_size)
        
        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        
        # Hard thresholding - keep only fraction of modes
        n_modes = x_fft.shape[1]
        n_modes_keep = int(n_modes * self.hard_thresholding_fraction)
        x_fft = x_fft[:, :n_modes_keep, :, :]
        
        # 频域线性变换 (复数运算)
        # 分离实部和虚部
        x_real = x_fft.real
        x_imag = x_fft.imag
        
        # 两层MLP in frequency domain
        o1_real = torch.einsum('bfnc,ncd->bfnd', x_real, self.w1[0]) - \
                  torch.einsum('bfnc,ncd->bfnd', x_imag, self.w1[1]) + self.b1[0]
        o1_imag = torch.einsum('bfnc,ncd->bfnd', x_imag, self.w1[0]) + \
                  torch.einsum('bfnc,ncd->bfnd', x_real, self.w1[1]) + self.b1[1]
        
        # ReLU activation (applied to magnitude)
        o1_real = F.relu(o1_real)
        o1_imag = F.relu(o1_imag)
        
        o2_real = torch.einsum('bfnc,ncd->bfnd', o1_real, self.w2[0]) - \
                  torch.einsum('bfnc,ncd->bfnd', o1_imag, self.w2[1]) + self.b2[0]
        o2_imag = torch.einsum('bfnc,ncd->bfnd', o1_imag, self.w2[0]) + \
                  torch.einsum('bfnc,ncd->bfnd', o1_real, self.w2[1]) + self.b2[1]
        
        # Soft thresholding (sparsity)
        x_fft_out = torch.complex(o2_real, o2_imag)
        
        # Pad back to original size if needed
        if n_modes_keep < n_modes:
            padding = torch.zeros(batch_size, n_modes - n_modes_keep, self.num_blocks, block_size, 
                                  dtype=x_fft_out.dtype, device=x.device)
            x_fft_out = torch.cat([x_fft_out, padding], dim=1)
        
        # Inverse FFT
        x = torch.fft.irfft(x_fft_out, n=seq_len, dim=1, norm="ortho")
        
        # Reshape back
        x = x.view(batch_size, seq_len, hidden_dim)
        
        # Residual + MLP
        x = residual + x
        x = x + self.mlp(self.norm(x))
        
        return x


class AFNOBackbone(nn.Module):
    """
    AFNO骨干网络 - 用于消融实验
    
    基于FourCastNet的AFNO架构，用于对比SSM骨干。
    
    参数:
        hidden_dim: 隐藏维度
        n_layers: AFNO块数量
        num_blocks: 频域block数量
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        num_blocks: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 确保hidden_dim可被num_blocks整除
        if hidden_dim % num_blocks != 0:
            num_blocks = 4  # 回退到安全值
        
        self.layers = nn.ModuleList([
            AFNOBlock(hidden_dim, num_blocks)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            [batch, seq_len, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Transolver (Physics-Attention Block)
# ============================================================================

class PhysicsAttention(nn.Module):
    """
    Physics-Aware Attention from Transolver
    
    论文: "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
    
    核心思想:
    - 将空间位置编码融入attention
    - 使用物理先验指导attention pattern
    - 支持非结构化网格
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        use_physics_bias: 是否使用物理位置偏置
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        use_physics_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_physics_bias = use_physics_bias
        
        # QKV投影
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Physics bias (可学习的位置相关偏置)
        if use_physics_bias:
            self.physics_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
            self.distance_scale = nn.Parameter(torch.ones(n_heads))
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            positions: [batch, seq_len, 1] 空间坐标 (可选)
            
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算QKV
        qkv = self.w_qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Physics-aware bias (基于距离的衰减)
        if self.use_physics_bias and positions is not None:
            # 计算位置距离矩阵
            pos = positions.squeeze(-1)  # [batch, seq_len]
            dist = torch.abs(pos.unsqueeze(-1) - pos.unsqueeze(-2))  # [batch, seq, seq]
            dist = dist.unsqueeze(1)  # [batch, 1, seq, seq]
            
            # 添加距离相关偏置
            physics_bias = -self.distance_scale.view(1, -1, 1, 1) * dist + self.physics_bias
            attn = attn + physics_bias
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)


class TransolverBlock(nn.Module):
    """
    Transolver编码器块
    
    结构:
        x -> LayerNorm -> PhysicsAttention -> + -> LayerNorm -> FFN -> +
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhysicsAttention(d_model, n_heads)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransolverBackbone(nn.Module):
    """
    Transolver骨干网络 - 用于消融实验
    
    基于Physics-Attention的Transformer变体。
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 层数
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        
        self.layers = nn.ModuleList([
            TransolverBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # 存储positions供forward使用
        self._positions = None
    
    def set_positions(self, positions: torch.Tensor):
        """设置空间位置 (用于physics-aware attention)"""
        self._positions = positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, self._positions)
        return self.final_norm(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# U-Net (2D)
# ============================================================================

class UNet2D(nn.Module):
    """
    Standard U-Net architecture for PDE surrogates.
    Can be used for 2D fields or 1D-time fields reshaped as images.
    """
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.dconv_down1 = double_conv(in_channels, hidden_dim)
        self.dconv_down2 = double_conv(hidden_dim, hidden_dim*2)
        self.dconv_down3 = double_conv(hidden_dim*2, hidden_dim*4)
        self.dconv_down4 = double_conv(hidden_dim*4, hidden_dim*8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(hidden_dim*4 + hidden_dim*8, hidden_dim*4)
        self.dconv_up2 = double_conv(hidden_dim*2 + hidden_dim*4, hidden_dim*2)
        self.dconv_up1 = double_conv(hidden_dim + hidden_dim*2, hidden_dim)

        self.conv_last = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        # Expect x: [batch, channels, height, width]
        conv1 = self.dconv_down1(x)
        x1 = self.maxpool(conv1)

        conv2 = self.dconv_down2(x1)
        x2 = self.maxpool(conv2)

        conv3 = self.dconv_down3(x2)
        x3 = self.maxpool(conv3)

        x = self.dconv_down4(x3)

        x = self.upsample(x)
        # Handle padding if dimensions don't match exactly due to pooling
        if x.size(2) != conv3.size(2) or x.size(3) != conv3.size(3):
            x = F.interpolate(x, size=(conv3.size(2), conv3.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        if x.size(2) != conv2.size(2) or x.size(3) != conv2.size(3):
            x = F.interpolate(x, size=(conv2.size(2), conv2.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        if x.size(2) != conv1.size(2) or x.size(3) != conv1.size(3):
            x = F.interpolate(x, size=(conv1.size(2), conv1.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


# ============================================================================
# DeepONet
# ============================================================================

class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet)
    """
    def __init__(self, in_features_branch, in_features_trunk, hidden_dim=128, num_layers=4, output_dim=1):
        super().__init__()
        self.output_dim = output_dim

        # Branch Net: Encodes input function u(x) at sensors
        layers_branch = [nn.Linear(in_features_branch, hidden_dim)]
        for _ in range(num_layers - 1):
            layers_branch.append(nn.ReLU())
            layers_branch.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net = nn.Sequential(*layers_branch)

        # Trunk Net: Encodes query coordinates y
        layers_trunk = [nn.Linear(in_features_trunk, hidden_dim)]
        for _ in range(num_layers - 1):
            layers_trunk.append(nn.ReLU())
            layers_trunk.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net = nn.Sequential(*layers_trunk)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        # u: [batch, m] (sensors)
        # y: [batch, p] or [batch, N, p] (locations)

        B_out = self.branch_net(u) # [batch, hidden]
        T_out = self.trunk_net(y)  # [batch, hidden] or [batch, N, hidden]

        if T_out.dim() == 2:
            # y is [batch, p]
            # B_out: [batch, hidden], T_out: [batch, hidden]
            out = torch.sum(B_out * T_out, dim=-1, keepdim=True)
        else:
            # y is [batch, N, p]
            # B_out: [batch, 1, hidden]
            out = torch.sum(B_out.unsqueeze(1) * T_out, dim=-1, keepdim=True)

        return out + self.bias


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Extended Backbones")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 128
    hidden_dim = 64
    
    # Test AFNO
    print("\n[1/2] Testing AFNO Backbone...")
    afno = AFNOBackbone(hidden_dim=hidden_dim, n_layers=4)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = afno(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Parameters: {afno.count_parameters():,}")
    
    # Test Transolver
    print("\n[2/2] Testing Transolver Backbone...")
    transolver = TransolverBackbone(d_model=hidden_dim, n_layers=4)
    positions = torch.linspace(-1, 1, seq_len).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
    transolver.set_positions(positions)
    y = transolver(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Parameters: {transolver.count_parameters():,}")

    # Test UNet2D
    print("\n[3/2] Testing UNet2D...")
    unet = UNet2D(in_channels=1, out_channels=1, hidden_dim=16)
    x = torch.randn(2, 1, 64, 64)
    y = unet(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Test DeepONet
    print("\n[4/2] Testing DeepONet...")
    don = DeepONet(in_features_branch=10, in_features_trunk=2, hidden_dim=32)
    u = torch.randn(2, 10)
    y_coords = torch.randn(2, 20, 2)
    out = don(u, y_coords)
    print(f"  Input u shape: {u.shape}")
    print(f"  Input y shape: {y_coords.shape}")
    print(f"  Output shape: {out.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
