# -*- coding: utf-8 -*-
"""
GPU-optimized PGMNO experiment runner with AMP and acceleration techniques.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import nullcontext

# 添加父目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 抑制警告
warnings.filterwarnings('ignore')


# ============================================================================
# GPU优化配置
# ============================================================================

def setup_gpu_optimizations(gpu_id: int = 0, use_cudnn_benchmark: bool = True):
    """
    配置GPU优化选项

    Args:
        gpu_id: GPU设备ID
        use_cudnn_benchmark: 是否启用CuDNN benchmark模式

    Returns:
        torch.device: 配置好的设备
    """
    if not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        return torch.device('cpu')

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # CuDNN优化 - 自动选择最优算法
    if use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 启用TF32加速 (Ampere及以上GPU)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 打印GPU信息
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"[GPU] CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"[GPU] TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    return device


@dataclass
class GPUOptimizedConfig:
    """GPU优化实验配置"""
    # GPU设置
    gpu_id: int = 0
    use_amp: bool = True              # 混合精度训练
    use_compile: bool = False         # torch.compile (PyTorch 2.0+)
    use_cudnn_benchmark: bool = True  # CuDNN benchmark

    # 数据参数 - 调整dt以满足CFL条件
    n_grid: int = 64
    n_train: int = 20
    n_val: int = 5
    nt: int = 50
    dt: float = 0.005  # 减小dt以满足CFL条件 (原0.01)
    nu: float = 0.01 / np.pi

    # 模型参数
    hidden_dim: int = 32
    num_layers: int = 2
    k_steps: int = 2

    # 训练参数 - 论文Table 4声明: 500 epochs
    n_epochs: int = 500
    batch_size: int = 64             # 更大的batch_size利用GPU并行
    learning_rate: float = 1e-3
    gradient_accumulation: int = 1   # 梯度累积步数

    # 实验参数
    seeds: Tuple[int, ...] = (42,)
    forecast_horizon: int = 10

    # 输出
    output_dir: str = 'results'

    # 快速模式
    quick_mode: bool = False

    # 使用GPU数据生成
    use_gpu_data_gen: bool = True

    def __post_init__(self):
        if self.quick_mode:
            self.n_train = 10
            self.n_val = 3
            self.n_epochs = 5
            self.forecast_horizon = 5


# ============================================================================
# GPU优化数据加载
# ============================================================================

class GPUDataLoader:
    """
    GPU优化数据加载器

    特性:
    - 预先将数据加载到GPU
    - 使用pin_memory加速CPU->GPU传输
    - 支持non_blocking异步传输
    """

    def __init__(self, data: torch.Tensor, batch_size: int, device: torch.device,
                 shuffle: bool = True, pin_memory: bool = True):
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # 预先将数据放到pinned memory (加速GPU传输)
        if pin_memory and device.type == 'cuda':
            self.data = data.pin_memory()
        else:
            self.data = data

        self.n_samples = len(data)
        self.n_batches = max(1, self.n_samples // batch_size)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples)
        else:
            indices = torch.arange(self.n_samples)

        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            batch_indices = indices[start:end]

            # non_blocking=True允许异步传输
            batch = self.data[batch_indices].to(self.device, non_blocking=True)
            yield batch

    def __len__(self):
        return self.n_batches


# ============================================================================
# GPU优化实验运行器
# ============================================================================

class GPUOptimizedRunner:
    """
    GPU优化实验运行器

    采用多种优化技术提升执行效率。
    """

    def __init__(self, config: GPUOptimizedConfig):
        self.config = config
        self.results = {}

        # 设置GPU优化
        self.device = setup_gpu_optimizations(
            config.gpu_id,
            config.use_cudnn_benchmark
        )

        # 混合精度训练
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"[GPU] AMP (Mixed Precision): Enabled")
        else:
            self.scaler = None
            print(f"[GPU] AMP (Mixed Precision): Disabled")

        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 预分配常用张量 (避免重复分配)
        self._precompute_tensors()

    def _precompute_tensors(self):
        """预计算常用张量"""
        # 空间网格 - 预先放到GPU
        self.x_grid = torch.linspace(-1, 1, self.config.n_grid, device=self.device)
        self.x_grid_batch = self.x_grid.unsqueeze(0).unsqueeze(-1)

        # BDF系数 (预计算)
        self.bdf2_coeffs = torch.tensor([1.5, -2.0, 0.5], device=self.device)

        print(f"[GPU] Pre-allocated tensors on {self.device}")

    def _maybe_compile_model(self, model: nn.Module) -> nn.Module:
        """可选地编译模型 (PyTorch 2.0+)"""
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("[GPU] Model compiled with torch.compile")
            except Exception as e:
                print(f"[Warning] torch.compile failed: {e}")
        return model

    # ========================================================================
    # 数据生成 (GPU优化)
    # ========================================================================

    def generate_data(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练和验证数据 - GPU优化版"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.config.use_gpu_data_gen:
            # 使用GPU加速数据生成
            try:
                from fast_burgers_generator import generate_burgers_data_gpu
                print("[GPU Data] Using GPU-accelerated Burgers generator")

                train_data, _ = generate_burgers_data_gpu(
                    n_samples=self.config.n_train,
                    n_grid=self.config.n_grid,
                    nt=self.config.nt,
                    dt=self.config.dt,
                    nu=self.config.nu,
                    seed=seed,
                    device=self.device
                )

                val_data, _ = generate_burgers_data_gpu(
                    n_samples=self.config.n_val,
                    n_grid=self.config.n_grid,
                    nt=self.config.nt,
                    dt=self.config.dt,
                    nu=self.config.nu,
                    seed=seed + 1000,
                    device=self.device
                )

                return train_data, val_data

            except ImportError:
                print("[Warning] fast_burgers_generator not found, falling back to CPU")

        # 回退到CPU版本
        from improved_burgers_utils import generate_burgers_data_v2

        train_data, _ = generate_burgers_data_v2(
            n_samples=self.config.n_train,
            n_grid=self.config.n_grid,
            nt=self.config.nt,
            dt=self.config.dt,
            nu=self.config.nu,
            seed=seed
        )

        val_data, _ = generate_burgers_data_v2(
            n_samples=self.config.n_val,
            n_grid=self.config.n_grid,
            nt=self.config.nt,
            dt=self.config.dt,
            nu=self.config.nu,
            seed=seed + 1000
        )

        return train_data, val_data

    def prepare_windows(self, data: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备滑动窗口数据"""
        n_samples, total_time, n_grid = data.shape
        horizon = self.config.forecast_horizon
        inputs, targets = [], []

        for i in range(n_samples):
            for t in range(total_time - k - horizon):
                inputs.append(data[i, t:t+k, :])
                targets.append(data[i, t+k:t+k+horizon, :])

        return torch.stack(inputs), torch.stack(targets)

    # ========================================================================
    # GPU优化训练循环
    # ========================================================================

    def _train_epoch_optimized(
        self,
        model: nn.Module,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        use_bdf: bool = True,
        use_causal: bool = True,
        epsilon: float = 0.05
    ) -> float:
        """
        GPU优化的训练epoch

        优化点:
        1. 混合精度 (AMP)
        2. 梯度累积
        3. 预分配张量
        4. 高效内存使用
        """
        model.train()
        total_loss = torch.tensor(0.0, device=self.device)  # 在GPU上累加，避免CPU同步
        n_batches = max(1, len(train_inputs) // self.config.batch_size)

        # AMP上下文
        amp_context = torch.amp.autocast('cuda') if self.use_amp else nullcontext()

        for i in range(n_batches):
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, len(train_inputs))

            # non_blocking传输
            batch_input = train_inputs[start:end].to(self.device, non_blocking=True)
            batch_target = train_targets[start:end].to(self.device, non_blocking=True)

            # 扩展网格 (使用预分配的张量)
            x_grid_exp = self.x_grid_batch.expand(batch_input.shape[0], -1, -1)

            # 混合精度前向传播
            with amp_context:
                predictions = model.predict_multi_step(
                    batch_input, x_grid_exp, self.config.forecast_horizon
                )

                # 计算损失
                mse_loss = F.mse_loss(predictions, batch_target)

                # BDF损失
                if use_bdf:
                    bdf_loss = self._compute_bdf_loss_optimized(predictions)
                    loss = mse_loss + 0.1 * bdf_loss
                else:
                    loss = mse_loss

                # 因果加权
                if use_causal and epsilon > 0:
                    step_losses = torch.mean((predictions - batch_target) ** 2, dim=(0, 2))
                    weights = torch.exp(-epsilon * torch.cumsum(step_losses, dim=0))
                    loss = torch.mean(weights * step_losses)

                # 梯度累积缩放
                loss = loss / self.config.gradient_accumulation

            # 混合精度反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()

                if (i + 1) % self.config.gradient_accumulation == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            else:
                loss.backward()

                if (i + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # 在GPU上累加，避免CPU同步
            total_loss += loss.detach() * self.config.gradient_accumulation

        # 只在最后同步一次
        return (total_loss / n_batches).item()

    def _compute_bdf_loss_optimized(self, pred: torch.Tensor) -> torch.Tensor:
        """GPU优化的BDF损失计算"""
        if pred.shape[1] < 3:
            return torch.tensor(0.0, device=self.device)

        # 使用预计算的BDF系数
        dt_approx = (self.bdf2_coeffs[0] * pred[:, 2:, :] +
                     self.bdf2_coeffs[1] * pred[:, 1:-1, :] +
                     self.bdf2_coeffs[2] * pred[:, :-2, :]) / self.config.dt

        # 空间导数 (向量化计算)
        dx = 2.0 / self.config.n_grid
        u = pred[:, 2:, :]
        u_x = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)
        u_xx = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (dx ** 2)

        # Burgers方程残差
        residual = dt_approx + u * u_x - self.config.nu * u_xx
        return torch.mean(residual ** 2)

    @torch.inference_mode()  # 比no_grad更快
    def _evaluate_model(
        self,
        model: nn.Module,
        val_inputs: torch.Tensor,
        val_targets: torch.Tensor
    ) -> Dict:
        """GPU优化的模型评估"""
        model.eval()

        # 批量评估
        test_input = val_inputs[:5].to(self.device, non_blocking=True)
        test_target = val_targets[:5].to(self.device, non_blocking=True)
        x_grid_test = self.x_grid_batch.expand(5, -1, -1)

        # 混合精度推理
        with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
            predictions = model.predict_multi_step(
                test_input, x_grid_test, self.config.forecast_horizon
            )

        # 计算指标
        l2_error = torch.mean((predictions - test_target) ** 2).item()

        # 向量化计算逐步误差
        step_errors = torch.sqrt(
            torch.mean((predictions - test_target) ** 2, dim=(0, 2))
        ).cpu().tolist()

        return {
            "test_l2_error": l2_error,
            "step_errors": step_errors
        }

    # ========================================================================
    # 消融实验 (GPU优化)
    # ========================================================================

    def run_ablation(self) -> List[Dict]:
        """运行GPU优化的消融实验"""
        print("\n" + "="*70)
        print("GPU-OPTIMIZED ABLATION STUDY")
        print("="*70)

        from improved_pgmno_model import PGMNOV2

        ablation_configs = [
            {"name": "Full_PGMNO", "use_multistep": True, "use_bdf_loss": True,
             "use_causal_weight": True, "k_steps": 2, "epsilon": 0.05},
            {"name": "wo_Multistep", "use_multistep": False, "use_bdf_loss": True,
             "use_causal_weight": True, "k_steps": 1, "epsilon": 0.05},
            {"name": "wo_BDF_Loss", "use_multistep": True, "use_bdf_loss": False,
             "use_causal_weight": True, "k_steps": 2, "epsilon": 0.05},
            {"name": "wo_Causal_Weight", "use_multistep": True, "use_bdf_loss": True,
             "use_causal_weight": False, "k_steps": 2, "epsilon": 0.0},
        ]

        all_results = []

        for exp_config in ablation_configs:
            name = exp_config["name"]
            print(f"\n[Experiment] {name}")

            seed_results = []
            for seed in self.config.seeds:
                # GPU同步计时
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                result = self._run_single_ablation_optimized(exp_config, seed)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start_time

                result['training_time'] = elapsed
                seed_results.append(result)
                print(f"  Seed {seed}: L2={result['test_l2_error']:.4f}, Time={elapsed:.1f}s")

            # 聚合结果 - 使用 .get() 安全访问
            valid_l2_errors = [r.get("test_l2_error") for r in seed_results if r.get("test_l2_error") is not None]
            if valid_l2_errors:
                mean_error = np.mean(valid_l2_errors)
                std_error = np.std(valid_l2_errors)
            else:
                mean_error = float('nan')
                std_error = float('nan')
            mean_time = np.mean([r.get("training_time", 0) for r in seed_results])

            exp_result = {
                "name": name,
                "config": exp_config,
                "seed_results": seed_results,
                "mean_metrics": {
                    "test_l2_error": float(mean_error),
                    "training_time": float(mean_time)
                },
                "std_metrics": {
                    "test_l2_error": float(std_error)
                }
            }
            all_results.append(exp_result)

            print(f"  Result: L2 = {mean_error:.4f} +/- {std_error:.4f}")
            print(f"  Avg Time: {mean_time:.1f}s")

        # 保存结果
        self._save_results(all_results, "ablation_gpu_optimized.json")
        self.results['ablation'] = all_results

        return all_results

    def _run_single_ablation_optimized(self, exp_config: Dict, seed: int) -> Dict:
        """运行单次GPU优化的消融实验"""
        from improved_pgmno_model import PGMNOV2

        torch.manual_seed(seed)
        np.random.seed(seed)

        k = exp_config["k_steps"]

        # 创建模型并移到GPU
        model = PGMNOV2(
            k_steps=k,
            dt=self.config.dt,
            spatial_dim=self.config.n_grid,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)

        # 可选编译
        model = self._maybe_compile_model(model)

        # 生成数据
        train_data, val_data = self.generate_data(seed)
        train_inputs, train_targets = self.prepare_windows(train_data, k)
        val_inputs, val_targets = self.prepare_windows(val_data, k)

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )

        # 训练
        for epoch in range(self.config.n_epochs):
            loss = self._train_epoch_optimized(
                model, train_inputs, train_targets, optimizer,
                use_bdf=exp_config["use_bdf_loss"],
                use_causal=exp_config["use_causal_weight"],
                epsilon=exp_config["epsilon"]
            )

        # 评估
        metrics = self._evaluate_model(model, val_inputs, val_targets)

        return {
            "seed": seed,
            "test_l2_error": metrics.get("test_l2_error", float('nan')),
            "step_errors": metrics.get("step_errors", []),
            "n_params": sum(p.numel() for p in model.parameters())
        }

    # ========================================================================
    # BDF敏感性分析 (GPU优化)
    # ========================================================================

    def run_bdf_sensitivity(self) -> Dict:
        """GPU优化的BDF敏感性分析"""
        print("\n" + "="*70)
        print("GPU-OPTIMIZED BDF SENSITIVITY")
        print("="*70)

        k_values = [1, 2, 3, 4, 5] if not self.config.quick_mode else [1, 2, 3]
        results = {"experiments": []}

        for k in k_values:
            print(f"\n[Testing k={k}]")

            seed_results = []
            for seed in self.config.seeds:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                result = self._run_bdf_experiment_optimized(k, seed)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                result['training_time'] = time.time() - start_time

                seed_results.append(result)

            mean_error = np.mean([r["l2_error"] for r in seed_results])
            mean_time = np.mean([r["training_time"] for r in seed_results])

            results["experiments"].append({
                "k": k,
                "mean_l2_error": float(mean_error),
                "mean_time": float(mean_time),
                "seed_results": seed_results
            })

            print(f"  Result: L2 = {mean_error:.4f}, Time = {mean_time:.1f}s")

        self._save_results(results, "bdf_sensitivity_gpu.json")
        self.results['bdf_sensitivity'] = results

        return results

    def _run_bdf_experiment_optimized(self, k: int, seed: int) -> Dict:
        """运行单次GPU优化的BDF实验"""
        from improved_pgmno_model import PGMNOV2

        torch.manual_seed(seed)

        model = PGMNOV2(
            k_steps=k,
            dt=self.config.dt,
            spatial_dim=self.config.n_grid,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)

        model = self._maybe_compile_model(model)

        train_data, val_data = self.generate_data(seed)
        train_inputs, train_targets = self.prepare_windows(train_data, k)
        val_inputs, val_targets = self.prepare_windows(val_data, k)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.n_epochs):
            self._train_epoch_optimized(model, train_inputs, train_targets, optimizer)

        metrics = self._evaluate_model(model, val_inputs, val_targets)

        return {
            "seed": seed,
            "l2_error": metrics.get("test_l2_error", float('nan')),
            "step_errors": metrics.get("step_errors", [])
        }

    # ========================================================================
    # 计算成本分析 (GPU优化)
    # ========================================================================

    def run_computational_profiling(self) -> Dict:
        """GPU优化的计算成本分析"""
        print("\n" + "="*70)
        print("GPU-OPTIMIZED COMPUTATIONAL PROFILING")
        print("="*70)

        from improved_pgmno_model import PGMNOV2
        from baselines.backbones import TransformerBackbone, FNOBackbone

        results = {"models": {}}

        # PGMNO
        print("\n[Profiling PGMNO]")
        pgmno = PGMNOV2(
            k_steps=2, dt=self.config.dt,
            spatial_dim=self.config.n_grid,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)
        results["models"]["PGMNO"] = self._profile_model_gpu(pgmno, "pgmno")

        # Transformer
        print("[Profiling Transformer]")
        transformer = TransformerBackbone(
            d_model=self.config.hidden_dim,
            n_heads=4,
            n_layers=self.config.num_layers
        ).to(self.device)
        results["models"]["Transformer"] = self._profile_model_gpu(transformer, "transformer")

        # FNO
        print("[Profiling FNO]")
        fno = FNOBackbone(
            in_channels=2, out_channels=1,
            width=self.config.hidden_dim,
            modes=16,
            n_layers=self.config.num_layers
        ).to(self.device)
        results["models"]["FNO"] = self._profile_model_gpu(fno, "fno")

        # 打印对比
        self._print_profiling_comparison(results)

        self._save_results(results, "profiling_gpu.json")
        self.results['profiling'] = results

        return results

    def _profile_model_gpu(self, model: nn.Module, model_type: str) -> Dict:
        """GPU精确计时分析"""
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        # 准备输入
        batch_size = 32
        seq_len = self.config.n_grid

        if model_type == "pgmno":
            x = torch.randn(batch_size, 2, seq_len, device=self.device)
            grid = self.x_grid_batch.expand(batch_size, -1, -1)
        elif model_type == "transformer":
            x = torch.randn(batch_size, seq_len, self.config.hidden_dim, device=self.device)
        else:  # fno
            x = torch.randn(batch_size, seq_len, 2, device=self.device)

        # 预热 (重要!)
        with torch.inference_mode():
            for _ in range(10):
                if model_type == "pgmno":
                    _ = model.predict_multi_step(x, grid, 1)
                else:
                    _ = model(x)

        # GPU同步计时
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

            # 使用CUDA事件精确计时
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            n_runs = 100
            start_event.record()

            with torch.inference_mode():
                for _ in range(n_runs):
                    if model_type == "pgmno":
                        _ = model.predict_multi_step(x, grid, 1)
                    else:
                        _ = model(x)

            end_event.record()
            torch.cuda.synchronize()

            inference_time = start_event.elapsed_time(end_event) / n_runs  # ms

            # 内存分析
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                if model_type == "pgmno":
                    _ = model.predict_multi_step(x, grid, 1)
                else:
                    _ = model(x)
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            # CPU计时
            start = time.time()
            n_runs = 50
            with torch.inference_mode():
                for _ in range(n_runs):
                    if model_type == "pgmno":
                        _ = model.predict_multi_step(x, grid, 1)
                    else:
                        _ = model(x)
            inference_time = (time.time() - start) / n_runs * 1000
            memory_mb = 0

        return {
            "n_params": n_params,
            "n_params_str": f"{n_params/1000:.1f}K" if n_params < 1e6 else f"{n_params/1e6:.1f}M",
            "inference_time_ms": round(inference_time, 3),
            "memory_mb": round(memory_mb, 2),
            "throughput": round(batch_size / inference_time * 1000, 2)
        }

    def _print_profiling_comparison(self, results: Dict):
        """打印性能对比表"""
        print("\n" + "-"*70)
        print(f"{'Model':<15} {'Params':<12} {'Time(ms)':<12} {'Memory(MB)':<12} {'Throughput':<12}")
        print("-"*70)
        for name, data in results["models"].items():
            print(f"{name:<15} {data['n_params_str']:<12} {data['inference_time_ms']:<12} "
                  f"{data['memory_mb']:<12} {data['throughput']:<12}")
        print("-"*70)

    # ========================================================================
    # 工具方法
    # ========================================================================

    def _save_results(self, results: Any, filename: str):
        """保存结果到JSON"""
        output_path = self.output_dir / filename

        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def run_all(self):
        """运行所有GPU优化实验"""
        start_time = time.time()

        print("\n" + "="*70)
        print("GPU-OPTIMIZED PGMNO EXPERIMENTS")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        try:
            self.run_ablation()
        except Exception as e:
            print(f"[ERROR] Ablation failed: {e}")

        try:
            self.run_bdf_sensitivity()
        except Exception as e:
            print(f"[ERROR] BDF sensitivity failed: {e}")

        try:
            self.run_computational_profiling()
        except Exception as e:
            print(f"[ERROR] Profiling failed: {e}")

        elapsed = time.time() - start_time

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)

        return self.results


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Optimized PGMNO Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments with GPU optimization
    python run_gpu_optimized.py --all

    # Enable mixed precision (recommended for Volta+)
    python run_gpu_optimized.py --all --amp

    # Use torch.compile (PyTorch 2.0+)
    python run_gpu_optimized.py --all --compile

    # Specify GPU device
    python run_gpu_optimized.py --all --gpu 0

    # Quick validation
    python run_gpu_optimized.py --all --quick
        """
    )

    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--bdf', action='store_true', help='Run BDF sensitivity')
    parser.add_argument('--profiling', action='store_true', help='Run profiling')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--amp', action='store_true', help='Enable AMP (mixed precision)')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--output', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # 配置
    config = GPUOptimizedConfig(
        gpu_id=args.gpu,
        use_amp=args.amp,
        use_compile=args.compile,
        quick_mode=args.quick,
        output_dir=args.output
    )

    # 运行
    runner = GPUOptimizedRunner(config)

    if args.all:
        runner.run_all()
    else:
        if args.ablation:
            runner.run_ablation()
        if args.bdf:
            runner.run_bdf_sensitivity()
        if args.profiling:
            runner.run_computational_profiling()


if __name__ == "__main__":
    main()
