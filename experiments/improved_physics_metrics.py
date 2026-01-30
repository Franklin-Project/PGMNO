# -*- coding: utf-8 -*-
"""
Physics-based evaluation metrics for PDE solver quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


# ============================================================================
# 物理指标数据类
# ============================================================================

@dataclass
class ImprovedPhysicsMetrics:
    """
    改进的物理一致性指标汇总

    改进点:
    1. 质量守恒使用相对误差(%)
    2. 梯度保真度使用余弦相似度和Pearson相关
    3. PDE残差追踪时间演化
    4. 能量守恒添加（适用于能量耗散PDE）
    5. 激波位置精度
    """
    # PDE残差
    mean_residual: float
    std_residual: float
    max_residual: float
    residual_over_time: List[float]

    # 守恒量误差（相对误差 %）
    mass_conservation_error: float  # 质量守恒相对误差
    mass_over_time: List[float]  # 质量随时间变化

    energy_conservation_error: Optional[float]  # 能量守恒相对误差
    energy_over_time: Optional[List[float]]  # 能量随时间变化

    # 梯度保真度
    gradient_cosine_similarity: float  # 余弦相似度
    gradient_pearson_correlation: float  # Pearson相关系数
    gradient_magnitude_error: float  # 梯度幅值相对误差

    # 激波位置误差
    shock_location_error: Optional[float]
    shock_tracking_over_time: Optional[List[float]]

    # 总体物理一致性评分
    physics_consistency_score: float  # 综合评分 (0-1)

    def to_dict(self) -> Dict:
        return {
            "mean_residual": self.mean_residual,
            "std_residual": self.std_residual,
            "max_residual": self.max_residual,
            "residual_over_time": self.residual_over_time,
            "mass_conservation_error": self.mass_conservation_error,
            "mass_over_time": self.mass_over_time,
            "energy_conservation_error": self.energy_conservation_error,
            "energy_over_time": self.energy_over_time,
            "gradient_cosine_similarity": self.gradient_cosine_similarity,
            "gradient_pearson_correlation": self.gradient_pearson_correlation,
            "gradient_magnitude_error": self.gradient_magnitude_error,
            "shock_location_error": self.shock_location_error,
            "shock_tracking_over_time": self.shock_tracking_over_time,
            "physics_consistency_score": self.physics_consistency_score
        }


# ============================================================================
# 改进的物理指标计算器
# ============================================================================

class ImprovedPhysicsMetricsCalculator:
    """
    Physics consistency metrics calculator.

    Improvements:
    1. 相对守恒误差（而非绝对误差）
    2. 多种梯度保真度指标
    3. 综合物理一致性评分
    4. 可视化工具
    """

    def __init__(
        self,
        nu: float = 0.01 / np.pi,  # 粘性系数
        dx: float = 2.0 / 128,     # 空间步长
        dt: float = 0.01,          # 时间步长
        equation_type: str = 'burgers'
    ):
        self.nu = nu
        self.dx = dx
        self.dt = dt
        self.equation_type = equation_type

    # ========================================================================
    # 主要计算入口
    # ========================================================================

    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> ImprovedPhysicsMetrics:
        """
        计算所有物理一致性指标

        Args:
            predictions: [batch, time_steps, space] 预测序列
            targets: [batch, time_steps, space] 目标序列
            history: [batch, k_steps, space] 历史状态（用于PDE残差）

        Returns:
            ImprovedPhysicsMetrics: 所有物理指标的汇总
        """
        # 1. PDE残差分析
        residual_results = self._compute_pde_residual_analysis(predictions, targets, history)

        # 2. 质量守恒分析
        mass_results = self._compute_mass_conservation(predictions)

        # 3. 能量守恒分析（如果适用）
        energy_results = self._compute_energy_conservation(predictions)

        # 4. 梯度保真度分析
        gradient_results = self._compute_gradient_fidelity(predictions, targets)

        # 5. 激波位置追踪
        shock_results = self._compute_shock_tracking(predictions, targets)

        # 6. 综合物理一致性评分
        consistency_score = self._compute_consistency_score(
            residual_results,
            mass_results,
            energy_results,
            gradient_results,
            shock_results
        )

        return ImprovedPhysicsMetrics(
            # PDE残差
            mean_residual=residual_results['mean'],
            std_residual=residual_results['std'],
            max_residual=residual_results['max'],
            residual_over_time=residual_results['over_time'],

            # 质量守恒
            mass_conservation_error=mass_results['relative_error'],
            mass_over_time=mass_results['over_time'],

            # 能量守恒
            energy_conservation_error=energy_results['relative_error'] if energy_results else None,
            energy_over_time=energy_results['over_time'] if energy_results else None,

            # 梯度保真度
            gradient_cosine_similarity=gradient_results['cosine_similarity'],
            gradient_pearson_correlation=gradient_results['pearson_correlation'],
            gradient_magnitude_error=gradient_results['magnitude_error'],

            # 激波位置
            shock_location_error=shock_results['location_error'] if shock_results else None,
            shock_tracking_over_time=shock_results['tracking_over_time'] if shock_results else None,

            # 综合评分
            physics_consistency_score=consistency_score
        )

    # ========================================================================
    # 1. PDE残差分析
    # ========================================================================

    def _compute_pde_residual_analysis(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        history: Optional[torch.Tensor]
    ) -> Dict:
        """
        计算PDE残差随时间的变化

        Burgers方程: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

        改进：
        - 使用相对误差（相对于u的幅度）
        - 跟踪时间演化趋势
        """
        batch_size, n_steps, n_grid = predictions.shape

        # 计算每个时间步的PDE残差
        residuals = []

        for t in range(n_steps - 1):
            u_curr = predictions[:, t, :]
            u_next = predictions[:, t + 1, :]
            u_true = targets[:, t + 1, :]

            # 空间导数（中心差分）
            u_x_true = (u_true[:, 2:] - u_true[:, :-2]) / (2 * self.dx)

            # 二阶导数
            u_xx_true = (u_true[:, 2:] - 2 * u_true[:, 1:-1] + u_true[:, :-2]) / self.dx**2

            # 时间导数
            u_t = (u_next - u_curr) / self.dt

            # PDE残差 (在预测解上计算)
            residual = torch.abs(
                u_t[:, 1:-1] + u_curr[:, 1:-1] * u_x_true - self.nu * u_xx_true
            )

            # 相对误差（相对于解的幅度）
            u_magnitude = torch.abs(u_true[:, 1:-1]).mean(dim=-1, keepdim=True) + 1e-8
            relative_residual = (residual / u_magnitude).mean().item()

            residuals.append(relative_residual)

        return {
            'mean': float(np.mean(residuals)) if residuals else 0.0,
            'std': float(np.std(residuals)) if residuals else 0.0,
            'max': float(np.max(residuals)) if residuals else 0.0,
            'over_time': residuals
        }

    # ========================================================================
    # 2. 质量守恒分析（改进版）
    # ========================================================================

    def _compute_mass_conservation(self, predictions: torch.Tensor) -> Dict:
        """
        计算质量守恒误差 - 改进版

        Burgers方程质量守恒：
        M(t) = ∫_Ω u(x,t) dx

        周期边界下，理论上 M(t+1) = M(t)

        改进策略：
        1. 使用相对误差（%）而非绝对误差
        2. 跟踪质量随时间的变化趋势
        3. 计算质量变化率
        """
        batch_size, n_steps, n_grid = predictions.shape

        # 计算每个时间步的质量（使用数值积分）
        mass = torch.sum(predictions, dim=-1) * self.dx  # [batch, n_steps]

        # 初始质量（t=0）
        initial_mass = mass[:, 0]

        # 最终质量
        final_mass = mass[:, -1]

        # 相对误差 (%)
        relative_error = torch.abs(final_mass - initial_mass) / (torch.abs(initial_mass) + 1e-8) * 100
        mean_relative_error = relative_error.mean().item()

        # 质量变化率（每时间步的变化）
        mass_change_rate = torch.abs(mass[:, 1:] - mass[:, :-1]).mean(dim=-1).mean(dim=-1).item()

        # 质量随时间演化
        mass_over_time = mass.mean(dim=0).tolist()

        return {
            'relative_error': mean_relative_error,
            'initial_mass': initial_mass.mean().item(),
            'final_mass': final_mass.mean().item(),
            'mass_change_rate': mass_change_rate,
            'over_time': mass_over_time
        }

    # ========================================================================
    # 3. 能量守恒分析
    # ========================================================================

    def _compute_energy_conservation(self, predictions: torch.Tensor) -> Optional[Dict]:
        """
        计算能量守恒

        Burgers方程能量：
        E(t) = ∫_Ω u(x,t)² dx

        理论：由于粘性耗散，能量应单调递减

        注：能量守恒适用于无强迫的Burgers方程
        """
        batch_size, n_steps, n_grid = predictions.shape

        # 计算每个时间步的能量
        energy = torch.sum(predictions**2, dim=-1) * self.dx  # [batch, n_steps]

        # 初始能量
        initial_energy = energy[:, 0]
        final_energy = energy[:, -1]

        # 理论上能量应该递减（粘性耗散）
        # 计算能量递减率
        energy_decay = (initial_energy - final_energy).mean().item()
        total_energy = initial_energy.mean().item()

        # 相对能量耗散（%）
        relative_dissipation = (energy_decay / (total_energy + 1e-8)) * 100

        # 能量随时间演化
        energy_over_time = energy.mean(dim=0).tolist()

        # 验证能量是否单调递减（应该是）
        is_monotonic_decreasing = (energy[:, 1:] <= energy[:, :-1]).float().mean().item() > 0.9

        return {
            'relative_error': relative_dissipation,
            'initial_energy': initial_energy.mean().item(),
            'final_energy': final_energy.mean().item(),
            'is_monotonic_decreasing': is_monotonic_decreasing,
            'over_time': energy_over_time
        }

    # ========================================================================
    # 4. 梯度保真度分析（改进版）
    # ========================================================================

    def _compute_gradient_fidelity(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict:
        """
        计算梯度保真度 - 改进版

        改进策略：
        1. 使用中心差分计算梯度
        2. 计算余弦相似度（方向一致性）
        3. 计算Pearson相关系数（线性关系）
        4. 计算梯度幅值相对误差

        梯度计算：∂u/∂x = (u[i+1] - u[i-1]) / (2*dx)
        """
        batch_size, n_steps, n_grid = predictions.shape

        # 计算空间梯度（中心差分，避免边界）
        pred_grad = (predictions[:, :, 2:] - predictions[:, :, :-2]) / (2 * self.dx)
        target_grad = (targets[:, :, 2:] - targets[:, :, :-2]) / (2 * self.dx)

        # 展平所有维度
        pred_grad_flat = pred_grad.flatten()
        target_grad_flat = target_grad.flatten()

        # 1. 余弦相似度
        cosine_sim = F.cosine_similarity(
            pred_grad_flat.unsqueeze(0),
            target_grad_flat.unsqueeze(0)
        ).item()

        # 2. Pearson相关系数
        pearson_corr = self._compute_pearson_correlation(
            pred_grad_flat,
            target_grad_flat
        )

        # 3. 梯度幅值误差
        pred_grad_mag = torch.abs(pred_grad).mean().item()
        target_grad_mag = torch.abs(target_grad).mean().item()
        magnitude_error = abs(pred_grad_mag - target_grad_mag) / (target_grad_mag + 1e-8)

        # 4. 逐点梯度误差
        pointwise_error = torch.mean((pred_grad - target_grad)**2).item()

        return {
            'cosine_similarity': cosine_sim,
            'pearson_correlation': pearson_corr,
            'magnitude_error': magnitude_error,
            'pointwise_mse': pointwise_error,
            'pred_magnitude': pred_grad_mag,
            'target_magnitude': target_grad_mag
        }

    def _compute_pearson_correlation(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        计算Pearson相关系数

        r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi-x̄)² * Σ(yi-ȳ)²)
        """
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = torch.sqrt(
            ((x - x_mean)**2).sum() * ((y - y_mean)**2).sum()
        ) + 1e-8

        return (numerator / denominator).item()

    # ========================================================================
    # 5. 激波位置追踪
    # ========================================================================

    def _compute_shock_tracking(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Optional[Dict]:
        """
        追踪激波位置

        激波位置： Burgers方程中梯度最大的点
        """
        batch_size, n_steps, n_grid = predictions.shape

        # 计算每个时间步的梯度
        pred_grad = torch.abs(predictions[:, :, 2:] - predictions[:, :, :-2]) / (2 * self.dx)
        target_grad = torch.abs(targets[:, :, 2:] - targets[:, :, :-2]) / (2 * self.dx)

        # 激波位置（梯度最大的点）
        pred_shock_loc = pred_grad.argmax(dim=-1).float() / (n_grid - 2)
        target_shock_loc = target_grad.argmax(dim=-1).float() / (n_grid - 2)

        # 激波位置误差
        location_errors = torch.abs(pred_shock_loc - target_shock_loc).mean(dim=-1).tolist()

        return {
            'location_error': float(np.mean(location_errors)),
            'tracking_over_time': location_errors
        }

    # ========================================================================
    # 6. 综合物理一致性评分
    # ========================================================================

    def _compute_consistency_score(
        self,
        residual_results: Dict,
        mass_results: Dict,
        energy_results: Optional[Dict],
        gradient_results: Dict,
        shock_results: Optional[Dict]
    ) -> float:
        """
        计算综合物理一致性评分

        评分标准:
        1. PDE残差：越小越好（归一化）
        2. 质量守恒：相对误差<10%得满分
        3. 梯度保真度：Pearson相关>0.8得满分
        4. 激波位置：误差<0.05得满分

        总分范围：0-1
        """
        scores = []

        # 1. PDE残差评分 (权重0.2)
        residual_score = 1.0 / (1.0 + residual_results['mean'])
        scores.append(0.2 * residual_score)

        # 2. 质量守恒评分 (权重0.3)
        # 相对误差<10%得满分，>30%得0分
        mass_error = abs(mass_results['relative_error'])
        if mass_error < 5:
            mass_score = 1.0
        elif mass_error < 10:
            mass_score = 0.8
        elif mass_error < 20:
            mass_score = 0.5
        elif mass_error < 30:
            mass_score = 0.3
        else:
            mass_score = 0.1
        scores.append(0.3 * mass_score)

        # 3. 梯度保真度评分 (权重0.3)
        grad_corr = abs(gradient_results['pearson_correlation'])
        if grad_corr > 0.95:
            grad_score = 1.0
        elif grad_corr > 0.9:
            grad_score = 0.8
        elif grad_corr > 0.8:
            grad_score = 0.6
        elif grad_corr > 0.7:
            grad_score = 0.4
        elif grad_corr > 0.5:
            grad_score = 0.2
        else:
            grad_score = 0.0
        scores.append(0.3 * grad_score)

        # 4. 激波位置评分 (权重0.2)
        if shock_results is not None:
            shock_error = shock_results['location_error']
            if shock_error < 0.02:
                shock_score = 1.0
            elif shock_error < 0.05:
                shock_score = 0.8
            elif shock_error < 0.1:
                shock_score = 0.5
            else:
                shock_score = 0.2
            scores.append(0.2 * shock_score)

        total_score = float(np.sum(scores))
        return min(max(total_score, 0.0), 1.0)  # 确保在[0,1]范围内

    # ========================================================================
    # 可视化方法
    # ========================================================================

    def plot_pde_residual_evolution(
        self,
        residual_over_time: List[float],
        save_path: str = None
    ):
        """绘制PDE残差随时间演化"""
        fig, ax = plt.subplots(figsize=(8, 5))

        steps = range(len(residual_over_time))
        ax.plot(steps, residual_over_time, 'b-', linewidth=2)
        ax.fill_between(steps,
                       np.array(residual_over_time) - np.array([0.05] * len(residual_over_time)),
                       np.array(residual_over_time) + np.array([0.05] * len(residual_over_time)),
                       alpha=0.3, color='blue')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Relative PDE Residual', fontsize=12)
        ax.set_title('PDE Residual Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(residual_over_time) * 1.2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] PDE residual plot to {save_path}")
        plt.close()

    def plot_mass_conservation(
        self,
        mass_over_time: List[float],
        save_path: str = None
    ):
        """绘制质量守恒随时间演化"""
        fig, ax = plt.subplots(figsize=(8, 5))

        steps = range(len(mass_over_time))
        ax.plot(steps, mass_over_time, 'g-', linewidth=2, label='Mass')
        ax.axhline(y=mass_over_time[0], color='r', linestyle='--',
                   alpha=0.5, label='Initial Mass')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Mass (∫u dx)', fontsize=12)
        ax.set_title('Mass Conservation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] Mass conservation plot to {save_path}")
        plt.close()

    def plot_gradient_fidelity(
        self,
        pred_grad: np.ndarray,
        target_grad: np.ndarray,
        save_path: str = None
    ):
        """绘制梯度保真度散点图"""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(target_grad.flatten(), pred_grad.flatten(),
                   alpha=0.3, s=5, c='blue', edgecolors='none')

        # 添加理想对角线
        min_val = min(target_grad.min(), pred_grad.min())
        max_val = max(target_grad.max(), pred_grad.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, alpha=0.5, label='Perfect Correlation')

        ax.set_xlabel('Target Gradient', fontsize=12)
        ax.set_ylabel('Predicted Gradient', fontsize=12)
        ax.set_title('Gradient Fidelity', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 计算并显示Pearson相关
        pearson = np.corrcoef(pred_grad.flatten(), target_grad.flatten())[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {pearson:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] Gradient fidelity plot to {save_path}")
        plt.close()

    def generate_summary_table(self, metrics: ImprovedPhysicsMetrics) -> str:
        """生成物理一致性摘要表格（Markdown格式）"""
        lines = [
            "## Physics Consistency Analysis",
            "",
            "| Metric | Value | Target | Status |",
            "|---------|-------|--------|--------|",
        ]

        # PDE残差
        residual_status = "Good" if metrics.mean_residual < 0.01 else "Fair" if metrics.mean_residual < 0.05 else "Poor"
        lines.append(f"| Mean PDE Residual | {metrics.mean_residual:.4f} | <0.01 | {residual_status} |")

        # 质量守恒
        mass_status = "Excellent" if metrics.mass_conservation_error < 5 else \
                      "Good" if metrics.mass_conservation_error < 10 else \
                      "Fair" if metrics.mass_conservation_error < 20 else "Poor"
        lines.append(f"| Mass Conservation Error | {metrics.mass_conservation_error:.2f}% | <10% | {mass_status} |")

        # 梯度保真度
        grad_status = "Excellent" if metrics.gradient_pearson_correlation > 0.95 else \
                     "Good" if metrics.gradient_pearson_correlation > 0.85 else \
                     "Fair" if metrics.gradient_pearson_correlation > 0.7 else "Poor"
        lines.append(f"| Gradient Pearson Correlation | {metrics.gradient_pearson_correlation:.4f} | >0.85 | {grad_status} |")

        # 综合评分
        overall_status = "Excellent" if metrics.physics_consistency_score > 0.8 else \
                         "Good" if metrics.physics_consistency_score > 0.6 else \
                         "Fair" if metrics.physics_consistency_score > 0.4 else "Poor"
        lines.append("")
        lines.append(f"| **Overall Score** | **{metrics.physics_consistency_score:.4f}** | >0.8 | **{overall_status}** |")

        return '\n'.join(lines)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Improved Physics Metrics Calculator')
    parser.add_argument('--predictions', type=str, help='Path to predictions file')
    parser.add_argument('--targets', type=str, help='Path to targets file')
    parser.add_argument('--output', type=str, default='results/physics',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ImprovedPhysicsMetricsCalculator]")
    print(f"  Output directory: {output_dir}")

    # 这里可以添加实际的预测/目标数据加载逻辑
    # 为了演示，使用模拟数据
    print("\n[Note] This module provides improved physics metrics calculation.")
    print("       Use it within the unified experiment runner.")
    print("\nUsage:")
    print("  calc = ImprovedPhysicsMetricsCalculator(nu=0.01/np.pi)")
    print("  metrics = calc.compute_all_metrics(predictions, targets)")


if __name__ == "__main__":
    main()
