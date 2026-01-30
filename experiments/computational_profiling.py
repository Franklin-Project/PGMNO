# -*- coding: utf-8 -*-
"""
Computational cost profiling module for FLOPs, memory, and latency analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from improved_pgmno_model import PGMNOV2
from baselines.backbones import TransformerBackbone, FNOBackbone


# ============================================================================
# 计算成本数据类
# ============================================================================

@dataclass
class ProfileMetrics:
    """
    模型计算成本指标
    
    包含:
    - n_params: 参数量
    - flops: 浮点运算数 (估计值)
    - memory_peak_mb: 峰值内存 (MB)
    - inference_time_ms: 单次推理时间 (ms)
    - throughput: 吞吐量 (samples/sec)
    """
    n_params: int
    flops: float
    memory_peak_mb: float
    inference_time_ms: float
    throughput: float
    
    def to_dict(self) -> Dict:
        return {
            "n_params": self.n_params,
            "n_params_str": self._format_number(self.n_params),
            "flops": self.flops,
            "flops_str": self._format_number(self.flops, suffix='FLOPs'),
            "memory_peak_mb": self.memory_peak_mb,
            "inference_time_ms": self.inference_time_ms,
            "throughput": self.throughput
        }
    
    def _format_number(self, num: float, suffix: str = '') -> str:
        """格式化大数字"""
        if num >= 1e9:
            return f"{num/1e9:.2f}G{suffix}"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M{suffix}"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K{suffix}"
        else:
            return f"{num:.2f}{suffix}"


# ============================================================================
# 计算成本分析器
# ============================================================================

class ComputationalProfiler:
    """
    计算成本分析器
    
    使用PyTorch内置profiler进行FLOPs估计,避免外部依赖(thop/fvcore)。
    
    注意事项:
    - FLOPs是估计值,主要覆盖矩阵乘法和卷积
    - GPU内存使用torch.cuda.max_memory_allocated
    - CPU内存使用psutil (如可用)
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_cuda = self.device.type == 'cuda'
        print(f"[ComputationalProfiler] Using device: {self.device}")
    
    def profile_model(
        self,
        model: nn.Module,
        input_tensors: List[torch.Tensor],
        n_warmup: int = 5,
        n_runs: int = 20
    ) -> ProfileMetrics:
        """
        分析单个模型的计算成本
        
        Args:
            model: PyTorch模型
            input_tensors: 输入张量列表
            n_warmup: 预热次数
            n_runs: 测量次数
            
        Returns:
            ProfileMetrics: 计算成本指标
        """
        model = model.to(self.device)
        model.eval()
        
        # 移动输入到设备
        inputs = [x.to(self.device) for x in input_tensors]
        
        # 1. 参数量
        n_params = sum(p.numel() for p in model.parameters())
        
        # 2. FLOPs估计
        flops = self._estimate_flops(model, inputs)
        
        # 3. 内存分析
        memory_peak = self._measure_memory(model, inputs)
        
        # 4. 推理时间
        inference_time, throughput = self._measure_latency(
            model, inputs, n_warmup, n_runs
        )
        
        return ProfileMetrics(
            n_params=n_params,
            flops=flops,
            memory_peak_mb=memory_peak,
            inference_time_ms=inference_time,
            throughput=throughput
        )
    
    def _estimate_flops(self, model: nn.Module, inputs: List[torch.Tensor]) -> float:
        """
        使用PyTorch profiler估计FLOPs
        
        注意: with_flops选项需要PyTorch >= 1.10
        """
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ] + ([torch.profiler.ProfilerActivity.CUDA] if self.use_cuda else []),
                with_flops=True,
                profile_memory=False,
            ) as prof:
                with torch.no_grad():
                    _ = model(*inputs)
            
            # 汇总FLOPs
            total_flops = 0
            for event in prof.key_averages():
                if event.flops is not None:
                    total_flops += event.flops
            
            return total_flops
        
        except Exception as e:
            print(f"Warning: FLOPs estimation failed: {e}")
            # 回退到简单估计
            return self._simple_flops_estimate(model, inputs)
    
    def _simple_flops_estimate(self, model: nn.Module, inputs: List[torch.Tensor]) -> float:
        """
        简单FLOPs估计 (基于参数量)
        
        粗略估计: FLOPs ≈ 2 * n_params * batch_size * seq_len
        """
        n_params = sum(p.numel() for p in model.parameters())
        batch_size = inputs[0].shape[0]
        seq_len = inputs[0].shape[1] if inputs[0].dim() > 2 else 1
        
        return 2 * n_params * batch_size * seq_len
    
    def _measure_memory(self, model: nn.Module, inputs: List[torch.Tensor]) -> float:
        """测量峰值内存"""
        gc.collect()
        
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = model(*inputs)
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            # CPU: 使用psutil如果可用
            try:
                import psutil
                process = psutil.Process()
                before = process.memory_info().rss
                
                with torch.no_grad():
                    _ = model(*inputs)
                
                after = process.memory_info().rss
                peak_memory = (after - before) / (1024 ** 2)  # MB
            except ImportError:
                peak_memory = 0.0  # 无法测量
        
        return peak_memory
    
    def _measure_latency(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        n_warmup: int,
        n_runs: int
    ) -> Tuple[float, float]:
        """
        测量推理延迟和吞吐量
        
        Returns:
            (inference_time_ms, throughput_samples_per_sec)
        """
        # 预热
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(*inputs)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        # 计时
        times = []
        batch_size = inputs[0].shape[0]
        
        for _ in range(n_runs):
            if self.use_cuda:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(*inputs)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        inference_time_ms = avg_time * 1000
        throughput = batch_size / avg_time
        
        return inference_time_ms, throughput
    
    # ========================================================================
    # 模型对比
    # ========================================================================
    
    def compare_models(
        self,
        n_grid: int = 128,
        batch_size: int = 8,
        k_steps: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 4
    ) -> Dict[str, ProfileMetrics]:
        """
        对比不同模型的计算成本
        
        对比:
        1. PGMNO (Mamba骨干)
        2. PGMNO + Transformer骨干
        3. PGMNO + FNO骨干
        
        Returns:
            Dict: 每个模型的ProfileMetrics
        """
        print("\n" + "="*60)
        print("Model Computational Cost Comparison")
        print("="*60)
        
        results = {}
        
        # 创建测试输入
        past_states = torch.randn(batch_size, k_steps, n_grid)
        grid_coords = torch.linspace(-1, 1, n_grid).unsqueeze(0).unsqueeze(-1)
        grid_coords = grid_coords.expand(batch_size, -1, -1)
        
        # 1. PGMNO (Mamba)
        print("\n[1/3] Profiling PGMNO (Mamba backbone)...")
        pgmno = PGMNOV2(
            k_steps=k_steps,
            dt=0.01,
            spatial_dim=n_grid,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        results['PGMNO'] = self.profile_model(pgmno, [past_states, grid_coords])
        
        # 2. Transformer骨干
        print("\n[2/3] Profiling Transformer backbone...")
        from experiments.ablation_runner import PGMNOWithCustomBackbone
        transformer_model = PGMNOWithCustomBackbone(
            k_steps=k_steps,
            dt=0.01,
            spatial_dim=n_grid,
            backbone=TransformerBackbone(d_model=hidden_dim, n_heads=4, n_layers=num_layers),
            hidden_dim=hidden_dim
        )
        results['Transformer'] = self.profile_model(transformer_model, [past_states, grid_coords])
        
        # 3. FNO骨干
        # Note: FNO receives lifted input from PGMNOWithCustomBackbone, so in_channels=hidden_dim
        print("\n[3/3] Profiling FNO backbone...")
        fno_model = PGMNOWithCustomBackbone(
            k_steps=k_steps,
            dt=0.01,
            spatial_dim=n_grid,
            backbone=FNOBackbone(in_channels=hidden_dim, out_channels=hidden_dim, width=hidden_dim, n_layers=num_layers),
            hidden_dim=hidden_dim
        )
        results['FNO'] = self.profile_model(fno_model, [past_states, grid_coords])
        
        return results
    
    # ========================================================================
    # Scaling分析
    # ========================================================================
    
    def analyze_scaling(
        self,
        model_builder: Callable = None,
        batch_size: int = 8
    ) -> Dict:
        """
        分析模型计算成本随不同因素的scaling
        
        测试维度:
        1. 序列长度 (空间分辨率): 64 -> 128 -> 256 -> 512 -> 1024
        2. 潜在维度: 32 -> 64 -> 128 -> 256
        3. 层数: 2 -> 4 -> 6 -> 8
        
        Reviewer requirements addressed:
        "complexity scaling with sequence length, spatial resolution, and latent token count"
        
        Returns:
            Dict: 各维度的scaling结果
        """
        print("\n" + "="*60)
        print("Computational Scaling Analysis")
        print("="*60)
        
        results = {
            "sequence_length": self._analyze_sequence_scaling(batch_size),
            "hidden_dim": self._analyze_hidden_dim_scaling(batch_size),
            "num_layers": self._analyze_layer_scaling(batch_size)
        }
        
        return results
    
    def _analyze_sequence_scaling(self, batch_size: int) -> Dict:
        """分析序列长度(空间分辨率)scaling"""
        print("\n[Sequence Length Scaling]")
        
        resolutions = [64, 128, 256, 512, 1024]
        results = []
        
        for n_grid in resolutions:
            print(f"  Testing n_grid={n_grid}...", end=" ")
            
            model = PGMNOV2(
                k_steps=2, dt=0.01, spatial_dim=n_grid,
                hidden_dim=64, num_layers=4
            )
            
            past_states = torch.randn(batch_size, 2, n_grid)
            grid_coords = torch.linspace(-1, 1, n_grid).unsqueeze(0).unsqueeze(-1)
            grid_coords = grid_coords.expand(batch_size, -1, -1)
            
            metrics = self.profile_model(model, [past_states, grid_coords])
            
            results.append({
                "n_grid": n_grid,
                "inference_time_ms": metrics.inference_time_ms,
                "memory_mb": metrics.memory_peak_mb,
                "throughput": metrics.throughput
            })
            
            print(f"time={metrics.inference_time_ms:.2f}ms")
        
        return results
    
    def _analyze_hidden_dim_scaling(self, batch_size: int) -> Dict:
        """分析潜在维度scaling"""
        print("\n[Hidden Dimension Scaling]")
        
        hidden_dims = [32, 64, 128, 256]
        results = []
        
        for hidden_dim in hidden_dims:
            print(f"  Testing hidden_dim={hidden_dim}...", end=" ")
            
            model = PGMNOV2(
                k_steps=2, dt=0.01, spatial_dim=128,
                hidden_dim=hidden_dim, num_layers=4
            )
            
            past_states = torch.randn(batch_size, 2, 128)
            grid_coords = torch.linspace(-1, 1, 128).unsqueeze(0).unsqueeze(-1)
            grid_coords = grid_coords.expand(batch_size, -1, -1)
            
            metrics = self.profile_model(model, [past_states, grid_coords])
            
            results.append({
                "hidden_dim": hidden_dim,
                "n_params": metrics.n_params,
                "inference_time_ms": metrics.inference_time_ms,
                "memory_mb": metrics.memory_peak_mb
            })
            
            print(f"params={metrics.n_params:,}, time={metrics.inference_time_ms:.2f}ms")
        
        return results
    
    def _analyze_layer_scaling(self, batch_size: int) -> Dict:
        """分析层数scaling"""
        print("\n[Layer Count Scaling]")
        
        layer_counts = [2, 4, 6, 8]
        results = []
        
        for num_layers in layer_counts:
            print(f"  Testing num_layers={num_layers}...", end=" ")
            
            model = PGMNOV2(
                k_steps=2, dt=0.01, spatial_dim=128,
                hidden_dim=64, num_layers=num_layers
            )
            
            past_states = torch.randn(batch_size, 2, 128)
            grid_coords = torch.linspace(-1, 1, 128).unsqueeze(0).unsqueeze(-1)
            grid_coords = grid_coords.expand(batch_size, -1, -1)
            
            metrics = self.profile_model(model, [past_states, grid_coords])
            
            results.append({
                "num_layers": num_layers,
                "n_params": metrics.n_params,
                "inference_time_ms": metrics.inference_time_ms,
                "memory_mb": metrics.memory_peak_mb
            })
            
            print(f"params={metrics.n_params:,}, time={metrics.inference_time_ms:.2f}ms")
        
        return results
    
    # ========================================================================
    # 结果输出
    # ========================================================================
    
    def print_comparison_table(self, results: Dict[str, ProfileMetrics]):
        """打印对比表格"""
        print("\n" + "="*80)
        print("Computational Cost Comparison")
        print("="*80)
        print(f"{'Model':<20} {'Params':<15} {'FLOPs':<15} {'Memory (MB)':<15} {'Time (ms)':<12} {'Throughput':<15}")
        print("-"*80)
        
        for name, metrics in results.items():
            metrics_dict = metrics.to_dict()
            print(f"{name:<20} {metrics_dict['n_params_str']:<15} {metrics_dict['flops_str']:<15} "
                  f"{metrics.memory_peak_mb:<15.2f} {metrics.inference_time_ms:<12.2f} "
                  f"{metrics.throughput:<15.2f}")
        
        print("="*80)
    
    def save_results(self, results: Dict, output_path: str):
        """保存结果到JSON"""
        # 转换为可序列化格式
        serializable = {}
        for key, value in results.items():
            if isinstance(value, ProfileMetrics):
                serializable[key] = value.to_dict()
            else:
                serializable[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to {output_path}")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PGMNO Computational Profiling')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'scaling', 'both'],
                       help='Profiling mode')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default='profiling_results.json')
    parser.add_argument('--n_grid', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    profiler = ComputationalProfiler(device=args.device)
    
    all_results = {}
    
    if args.mode in ['compare', 'both']:
        comparison = profiler.compare_models(
            n_grid=args.n_grid,
            batch_size=args.batch_size
        )
        profiler.print_comparison_table(comparison)
        all_results['comparison'] = {k: v.to_dict() for k, v in comparison.items()}
    
    if args.mode in ['scaling', 'both']:
        scaling = profiler.analyze_scaling(batch_size=args.batch_size)
        all_results['scaling'] = scaling
    
    profiler.save_results(all_results, args.output)


if __name__ == "__main__":
    main()
