# -*- coding: utf-8 -*-
"""
Baseline backbone networks for ablation experiments (Transformer and FNO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# Transformer骨干网络 (用于消融实验)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    
    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    
    参数:
        d_model (int): 模型维度
        n_heads (int): 注意力头数
        dropout (float): Dropout比率
    
    注意:
        计算复杂度为O(L^2), 其中L为序列长度
        这是与SSM的关键区别 - SSM具有O(L)复杂度
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性投影层 (Q, K, V)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            mask: 可选的注意力掩码
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V并reshape为多头格式
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数: [batch, n_heads, seq_len, seq_len]
        # 这里的O(L^2)复杂度是Transformer的主要瓶颈
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力权重
        context = torch.matmul(attn_probs, V)
        
        # 合并多头: [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层 (Pre-LN架构)
    
    结构:
        x -> LayerNorm -> MHSA -> + -> LayerNorm -> FFN -> +
        |__________________|      |_____________________|
    
    使用Pre-LN架构提升训练稳定性(参考论文: "On Layer Normalization in the 
    Transformer Architecture")
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # 默认FFN维度为4倍模型维度
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: 先归一化再计算
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    """
    Transformer骨干网络 - 用于消融实验
    
    这是PGMNO消融实验中用于对比SSM骨干的Transformer替代方案。
    保持等参数量以确保公平对比。
    
    对应论文消融实验:
        "Study on the impact of neural architecture Vs numerical integration choice"
    
    参数:
        d_model (int): 模型隐藏维度 (默认64, 与PGMNO一致)
        n_heads (int): 注意力头数 (默认4)
        n_layers (int): 编码器层数 (默认4, 与PGMNO一致)
        dropout (float): Dropout比率
    
    复杂度分析:
        - 时间复杂度: O(L^2 * d_model) per layer
        - 空间复杂度: O(L^2) for attention matrix
        对比SSM的O(L * d_model)时间复杂度
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 堆叠Transformer层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
    
    def count_parameters(self) -> int:
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FNO骨干网络 (用于消融实验)
# ============================================================================

class SpectralConv1d(nn.Module):
    """
    1D傅里叶谱卷积层
    
    论文: "Fourier Neural Operator for Parametric Partial Differential Equations"
    
    核心思想:
        1. FFT将信号变换到频域
        2. 在频域进行线性变换(保留前modes个模式)
        3. IFFT变换回时域
    
    公式:
        y = IFFT(R * FFT(x))
        其中R是可学习的频域权重矩阵
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        modes (int): 保留的傅里叶模式数 (决定表达能力)
    
    优势:
        - 全局感受野 (通过频域表示)
        - O(L log L) 复杂度 (FFT)
    
    局限:
        - 需要均匀网格
        - 周期性边界假设
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 保留的傅里叶模式数
        
        # 频域权重 (复数参数)
        # 形状: [in_channels, out_channels, modes]
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def compl_mul1d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        复数矩阵乘法
        
        Args:
            input_tensor: [batch, in_channels, modes]
            weights: [in_channels, out_channels, modes]
            
        Returns:
            [batch, out_channels, modes]
        """
        # 使用einsum进行批量复数矩阵乘法
        return torch.einsum("bim,iom->bom", input_tensor, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, in_channels, seq_len]
            
        Returns:
            输出张量 [batch, out_channels, seq_len]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        
        # 计算FFT
        x_ft = torch.fft.rfft(x)
        
        # 初始化输出频域表示
        out_ft = torch.zeros(
            batch_size, self.out_channels, seq_len // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # 对低频模式应用可学习变换
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        
        # 逆FFT回到时域
        x = torch.fft.irfft(out_ft, n=seq_len)
        
        return x


class FNOBlock(nn.Module):
    """
    FNO基本块
    
    结构:
        x ---> SpectralConv ---> +  ---> GELU ---> output
           |                     |
           +---> Linear (W) -----+
    
    包含:
        1. 频域卷积分支 (全局信息)
        2. 局部线性变换分支 (局部信息)
    """
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes
        
        # 频域卷积
        self.spectral_conv = SpectralConv1d(width, width, modes)
        
        # 局部线性变换 (bypass)
        # 使用Linear代替Conv1d(1)以避免cuDNN "unable to find engine" 错误
        self.w = nn.Linear(width, width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, width, seq_len]
        """
        x1 = self.spectral_conv(x)
        # Conv1d(1) equivalent using Linear: [B, C, L] -> [B, L, C] -> Linear -> [B, C, L]
        x2 = self.w(x.permute(0, 2, 1)).permute(0, 2, 1)
        return F.gelu(x1 + x2)


class FNOBackbone(nn.Module):
    """
    FNO骨干网络 - 用于消融实验
    
    这是PGMNO消融实验中用于对比SSM骨干的FNO替代方案。
    
    对应论文消融实验:
        "Implement ablation for latent SSM architecture vs attention vs FNO backbone"
    
    参数:
        in_channels (int): 输入通道数 (通常为1或2: u + x)
        out_channels (int): 输出通道数 (通常为1: u)
        width (int): 隐藏层宽度 (默认64, 与PGMNO一致)
        modes (int): 傅里叶模式数 (默认16)
        n_layers (int): FNO块数量 (默认4)
    
    复杂度分析:
        - 时间复杂度: O(L log L) per layer (FFT)
        - 空间复杂度: O(L)
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4
    ):
        super().__init__()
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        
        # 输入提升层 (lifting)
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO块堆叠
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes) for _ in range(n_layers)
        ])
        
        # 输出投影层 (projection)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, in_channels]
            
        Returns:
            输出张量 [batch, seq_len, out_channels]
        """
        # 提升维度: [batch, seq_len, in_channels] -> [batch, seq_len, width]
        x = self.fc0(x)
        
        # 转换为通道优先格式: [batch, width, seq_len]
        x = x.permute(0, 2, 1).contiguous()
        
        # 通过FNO块
        for block in self.blocks:
            x = block(x)
        
        # 转换回序列优先格式: [batch, seq_len, width]
        x = x.permute(0, 2, 1)
        
        # 投影到输出维度
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试Transformer骨干
    print("=" * 60)
    print("Testing TransformerBackbone")
    print("=" * 60)
    
    transformer = TransformerBackbone(d_model=64, n_heads=4, n_layers=4)
    x = torch.randn(2, 128, 64)  # [batch, seq_len, d_model]
    y = transformer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {transformer.count_parameters():,}")
    
    # 测试FNO骨干
    print("\n" + "=" * 60)
    print("Testing FNOBackbone")
    print("=" * 60)
    
    fno = FNOBackbone(in_channels=2, out_channels=1, width=64, modes=16, n_layers=4)
    x = torch.randn(2, 128, 2)  # [batch, seq_len, in_channels]
    y = fno(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {fno.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
