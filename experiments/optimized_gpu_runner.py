# -*- coding: utf-8 -*-
"""
GPU优化的实验运行器
=====================================

针对低GPU利用率问题的优化版本:
1. 混合精度训练 (FP16)
2. 高效DataLoader (prefetch, pin_memory)
3. 批量并行计算
4. 数据预缓存到GPU
5. 动态batch size调整
6. CUDA kernel优化

目标：提高GPU利用率至80%+，减少训练时间50%+
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
import time
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# CUDA优化的数据生成器
# ============================================================================

class GPUAcceleratedBurgersGenerator:
    """
    GPU加速的Burgers方程数据生成器

    使用CUDA加速FFT和数据生成，大幅提升速度
    """

    def __init__(
        self,
        n_grid=64,
        nt=30,
        dt=0.01,
        nu=0.01/np.pi,
        device='cuda',
        cache_size=1000
    ):
        self.n_grid = n_grid
        self.nt = nt
        self.device = torch.device(device) if device == 'cuda' else torch.device('cpu')
        self.cache_size = cache_size

        # 标量参数 - 保存原始值用于CPU生成，tensor值用于GPU
        self._dt_scalar = dt
        self._nu_scalar = nu
        self.dt = torch.tensor(dt, device=self.device, dtype=torch.float32)
        self.nu = torch.tensor(nu, device=self.device, dtype=torch.float32)

        # 预计算网格坐标
        self.x_grid = torch.linspace(-1, 1, n_grid, device=self.device, dtype=torch.float32)
        self.dx = 2.0 / n_grid

        # FFT波数 (预计算) - 确保在正确的设备上
        k_np = 2 * np.pi * np.fft.fftfreq(n_grid, d=self.dx)
        self.k = torch.tensor(k_np, device=self.device, dtype=torch.float32)
        self.k_sq = self.k ** 2

        # 数据缓存
        self.data_cache = {}

    def generate_batch(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        use_gpu: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量生成数据，支持GPU加速

        Args:
            n_samples: 样本数量
            seed: 随机种子
            use_gpu: 是否使用GPU加速

        Returns:
            data: [n_samples, nt+1, n_grid]
            x_grid: [n_grid]
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        cache_key = f"{n_samples}_{seed}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        if use_gpu and torch.cuda.is_available():
            return self._generate_gpu_batch(n_samples, seed)
        else:
            return self._generate_cpu_batch(n_samples, seed)

    def _generate_gpu_batch(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU加速批量生成"""
        if seed is not None:
            torch.cuda.manual_seed(seed)

        # 批量生成初始条件
        u0_batch = torch.zeros(n_samples, self.n_grid, device=self.device, dtype=torch.float32)

        for i in range(n_samples):
            u0_batch[i] = self._generate_initial_condition_gpu(i)

        # 批量时间演化
        data = torch.zeros(n_samples, self.nt + 1, self.n_grid, device=self.device, dtype=torch.float32)
        data[:, 0, :] = u0_batch

        u_batch = u0_batch.clone()

        # 批量时间步演化 (向量化)
        for t in range(self.nt):
            # 扩散步 (FFT空间)
            u_fft = torch.fft.fft(u_batch, dim=-1)
            u_fft = u_fft * torch.exp(-self.nu * self.k_sq * self.dt)
            u_batch = torch.fft.ifft(u_fft).real

            # 对流步 (Pseudo-spectral)
            # 批量处理所有样本
            u_sq = 0.5 * u_batch ** 2

            # 在频域求导
            u_sq_fft = torch.fft.fft(u_sq, dim=-1)
            u_sq_x = torch.fft.ifft(1j * self.k * u_sq_fft).real

            # 更新
            u_batch = u_batch - self.dt * u_sq_x

            data[:, t + 1, :] = u_batch

        return data, self.x_grid.cpu()

    def _generate_initial_condition_gpu(self, seed_idx: int) -> torch.Tensor:
        """GPU上的初始条件生成"""
        torch.cuda.manual_seed(seed_idx)

        mode = torch.randint(0, 4, (1,)).item()
        x = self.x_grid

        if mode == 0:  # gaussian
            center = torch.rand(1, device=self.device).item() * 1.0 - 0.5
            width = torch.rand(1, device=self.device).item() * 0.2 + 0.1
            amplitude = torch.rand(1, device=self.device).item() * 1.5 + 0.5
            u0 = amplitude * torch.exp(-((x - center) ** 2) / (2 * width ** 2))

        elif mode == 1:  # sine
            freq = torch.randint(1, 5, (1,)).item()
            phase = torch.rand(1, device=self.device).item() * 2 * np.pi
            amplitude = torch.rand(1, device=self.device).item() * 1.0 + 0.5
            u0 = amplitude * torch.sin(freq * np.pi * x + phase)

        elif mode == 2:  # combined
            center = torch.rand(1, device=self.device).item() * 1.0 - 0.5
            u0 = torch.exp(-((x - center) ** 2) / 0.1)
            u0 += 0.5 * torch.sin(np.pi * x)
            u0 += 0.3 * torch.sin(2 * np.pi * x + torch.rand(1, device=self.device).item() * np.pi)

        else:  # random
            n_modes = 5
            u0 = torch.zeros_like(x)
            for k in range(1, n_modes + 1):
                amplitude = torch.randn(1, device=self.device).item() / k
                phase = torch.rand(1, device=self.device).item() * 2 * np.pi
                u0 += amplitude * torch.sin(k * np.pi * x + phase)

        return u0

    def _generate_cpu_batch(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU生成 (后备方案)"""
        from improved_burgers_utils import generate_burgers_data_v2
        data, x_grid = generate_burgers_data_v2(
            n_samples=n_samples,
            n_grid=self.n_grid,
            nt=self.nt,
            dt=self._dt_scalar,  # 使用标量值
            nu=self._nu_scalar,  # 使用标量值
            seed=seed
        )
        return data, x_grid


# ============================================================================
# 高效Dataset和DataLoader
# ============================================================================

class CachedBurgersDataset(Dataset):
    """
    带缓存的高效Dataset

    特性:
    1. 数据预生成并缓存到GPU
    2. 滑动窗口预计算
    3. 非阻塞数据传输
    """

    def __init__(
        self,
        n_samples: int,
        n_grid: int = 64,
        nt: int = 30,
        k_steps: int = 2,
        horizon: int = 10,
        device: str = 'cuda',
        seed: int = 42,
        dt: float = 0.01,
        nu: float = 0.01 / np.pi
    ):
        self.n_samples = n_samples
        self.n_grid = n_grid
        self.nt = nt
        self.k_steps = k_steps
        self.horizon = horizon
        self.device = torch.device(device)

        # 生成数据
        generator = GPUAcceleratedBurgersGenerator(
            n_grid=n_grid,
            nt=nt,
            dt=dt,
            nu=nu,
            device=device
        )
        print(f"[Dataset] Generating {n_samples} samples on {device}...")
        self.data, self.x_grid = generator.generate_batch(n_samples, seed, use_gpu=True)

        # 预计算滑动窗口
        self._precompute_windows()
        print(f"[Dataset] Cached {len(self)} training samples")

    def _precompute_windows(self):
        """预计算所有滑动窗口"""
        windows = []
        for i in range(self.n_samples):
            # 提取所有可能的滑动窗口
            for t in range(self.nt - self.k_steps - self.horizon + 1):
                inp = self.data[i, t:t + self.k_steps, :]
                target = self.data[i, t + self.k_steps:t + self.k_steps + self.horizon, :]
                windows.append((inp, target))

        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        inp, target = self.windows[idx]
        # x_grid: [n_grid] -> [1, n_grid, 1] 用于PGMNO模型
        # DataLoader会自动扩展batch维度
        x_grid_expanded = self.x_grid.unsqueeze(0).unsqueeze(-1)
        return inp, target, x_grid_expanded


class OptimizedDataLoader:
    """
    高效DataLoader封装

    优化特性:
    1. pin_memory加速CPU->GPU传输
    2. prefetch_factor减少等待
    3. 自动batch size优化
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        self.dataset = dataset

        # 自动设置num_workers
        if num_workers is None:
            num_workers = min(4, mp.cpu_count())

        # 自动优化batch size
        if torch.cuda.is_available():
            max_memory = torch.cuda.get_device_properties(0).total_memory
            # 保守估计：每个样本占用约n_grid*4 bytes
            optimal_batch_size = int((max_memory * 0.3) / (dataset.n_grid * 4 * dataset.horizon))
            batch_size = min(batch_size, optimal_batch_size)

        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"[DataLoader] batch_size={batch_size}, num_workers={num_workers}")

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# ============================================================================
# 混合精度训练优化器
# ============================================================================

class AMPTrainer:
    """
    自动混合精度训练器

    特性:
    1. 自动梯度缩放
    2. 动态loss scaling
    3. 梯度累积支持
    4. 内存优化
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.scaler = amp.GradScaler(enabled=self.use_amp)

        if self.use_amp:
            print(f"[AMP] Enabled on {device}")

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        x_grid: torch.Tensor,
        loss_fn: callable
    ) -> Tuple[float, float]:
        """
        执行一个训练步骤

        Returns:
            loss: 损失值
            scale: 当前梯度缩放因子
        """
        # 移动到设备 (non_blocking for async transfer)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        x_grid = x_grid.to(self.device, non_blocking=True)

        # 处理x_grid维度: [batch, 1, n_grid, 1] -> [batch, n_grid, 1]
        if x_grid.dim() == 4:
            x_grid = x_grid.squeeze(1)

        # 训练
        self.model.train()

        if self.use_amp:
            with amp.autocast():
                predictions = self.model.predict_multi_step(inputs, x_grid, targets.shape[1])
                loss = loss_fn(predictions, targets, inputs)

            # 梯度缩放
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if self.gradient_accumulation_steps == 1:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        else:
            predictions = self.model.predict_multi_step(inputs, x_grid, targets.shape[1])
            loss = loss_fn(predictions, targets, inputs)
            loss.backward()

            if self.gradient_accumulation_steps == 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss.item(), self.scaler.get_scale() if self.use_amp else 1.0

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        loss_fn: callable
    ) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)  # 在GPU上累加，避免CPU同步
        n_batches = 0

        for batch_in, batch_target, x_grid in val_loader:
            batch_in = batch_in.to(self.device, non_blocking=True)
            batch_target = batch_target.to(self.device, non_blocking=True)
            x_grid = x_grid.to(self.device, non_blocking=True)

            # 处理x_grid维度: [batch, 1, n_grid, 1] -> [batch, n_grid, 1]
            if x_grid.dim() == 4:
                x_grid = x_grid.squeeze(1)

            if self.use_amp:
                with amp.autocast():
                    predictions = self.model.predict_multi_step(batch_in, x_grid, batch_target.shape[1])
                    loss = loss_fn(predictions, batch_target, batch_in)
            else:
                predictions = self.model.predict_multi_step(batch_in, x_grid, batch_target.shape[1])
                loss = loss_fn(predictions, batch_target, batch_in)

            # 在GPU上累加，避免CPU同步
            total_loss += loss.detach()
            n_batches += 1

        # 只在最后同步一次
        return (total_loss / max(n_batches, 1)).item()


# ============================================================================
# 优化的消融实验运行器
# ============================================================================

@dataclass
class OptimizedConfig:
    """优化的配置"""
    n_epochs: int = 500
    batch_size: int = 64  # 自动调整
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 2
    n_grid: int = 64
    dt: float = 0.01
    nu: float = 0.01 / np.pi
    seeds: List[int] = None
    use_amp: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    gradient_accumulation: int = 1
    device: str = 'auto'

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]


class OptimizedAblationRunner:
    """
    GPU优化的消融实验运行器

    主要优化:
    1. 混合精度训练
    2. 高效DataLoader
    3. 数据预生成和缓存
    4. 动态batch size
    5. 内存优化
    """

    def __init__(self, config: OptimizedConfig):
        self.config = config

        # 设置设备
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        # CUDA设置
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            print(f"[CUDA] Device: {torch.cuda.get_device_name(0)}")
            print(f"[CUDA] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"[CUDA] cuDNN benchmark: {torch.backends.cudnn.benchmark}")

        print(f"[Runner] Device: {self.device}")
        print(f"[Runner] AMP: {config.use_amp}")

    def run_experiment(
        self,
        model_config: Dict,
        seed: int
    ) -> Dict:
        """运行单个实验"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        start_time = time.time()

        # 构建模型 - 使用标量dt和nu
        from improved_pgmno_model import PGMNOV2
        model = PGMNOV2(
            k_steps=2,
            dt=self.config.dt,  # OptimizedConfig中已经是float
            spatial_dim=self.config.n_grid,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        ).to(self.device)

        # 创建数据集
        train_dataset = CachedBurgersDataset(
            n_samples=20,
            n_grid=self.config.n_grid,
            k_steps=2,
            horizon=10,
            device=str(self.device),
            seed=seed,
            dt=self.config.dt,
            nu=self.config.nu
        )

        val_dataset = CachedBurgersDataset(
            n_samples=5,
            n_grid=self.config.n_grid,
            k_steps=2,
            horizon=10,
            device=str(self.device),
            seed=seed + 1000,
            dt=self.config.dt,
            nu=self.config.nu
        )

        # 创建优化后的DataLoader
        train_loader = OptimizedDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=0,  # 数据已缓存，不需要worker
            pin_memory=False
        )

        val_loader = OptimizedDataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=0,
            pin_memory=False
        )

        # 创建训练器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # 简单损失函数
        def loss_fn(pred, target, history):
            return nn.functional.mse_loss(pred, target)

        trainer = AMPTrainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            use_amp=self.config.use_amp,
            gradient_accumulation_steps=self.config.gradient_accumulation
        )

        # 训练循环
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(self.config.n_epochs):
            # 训练
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_in, batch_target, x_grid in train_loader:
                loss, _ = trainer.train_step(batch_in, batch_target, x_grid, loss_fn)
                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(n_batches, 1)

            # 验证
            val_loss = trainer.validate(val_loader, loss_fn)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: train={epoch_loss:.6f}, val={val_loss:.6f}")

        # 测试指标
        test_metrics = self._compute_test_metrics(model, val_loader)

        elapsed = time.time() - start_time

        return {
            "seed": seed,
            "metrics": test_metrics,
            "best_val_loss": best_val_loss,
            "training_time": elapsed,
            "n_params": sum(p.numel() for p in model.parameters())
        }

    def _compute_test_metrics(self, model, val_loader) -> Dict:
        """计算测试指标"""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_in, batch_target, x_grid in val_loader:
                batch_in = batch_in.to(self.device)
                batch_target = batch_target.to(self.device)
                x_grid = x_grid.to(self.device)

                # 处理x_grid维度: [batch, 1, n_grid, 1] -> [batch, n_grid, 1]
                if x_grid.dim() == 4:
                    x_grid = x_grid.squeeze(1)

                if self.config.use_amp:
                    with amp.autocast():
                        predictions = model.predict_multi_step(
                            batch_in, x_grid, batch_target.shape[1]
                        )
                else:
                    predictions = model.predict_multi_step(
                        batch_in, x_grid, batch_target.shape[1]
                    )

                all_preds.append(predictions)
                all_targets.append(batch_target)

                if len(all_preds) >= 2:
                    break

        predictions = torch.cat(all_preds, dim=0)[:32]
        targets = torch.cat(all_targets, dim=0)[:32]

        l2_error = torch.norm(predictions - targets) / torch.norm(targets)

        return {
            "test_l2_error": l2_error.item()
        }


# ============================================================================
# 优化的主运行器
# ============================================================================

def run_optimized_experiments(config: OptimizedConfig) -> List[Dict]:
    """
    运行优化的实验

    主要优化:
    1. GPU数据生成
    2. 混合精度训练
    3. 高效数据加载
    4. 内存优化
    """
    runner = OptimizedAblationRunner(config)
    results = []

    for seed in config.seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*60}")

        result = runner.run_experiment({}, seed)
        results.append(result)

        print(f"\n[Result] Seed {seed}:")
        l2_err = result.get('metrics', {}).get('test_l2_error')
        if l2_err is not None:
            print(f"  L2 Error: {l2_err:.6f}")
        else:
            print(f"  L2 Error: N/A")
        print(f"  Time: {result.get('training_time', 0):.1f}s")

    # 计算统计量 - 过滤掉没有metrics的结果
    l2_errors = [r.get('metrics', {}).get('test_l2_error') for r in results if r.get('metrics', {}).get('test_l2_error') is not None]
    if l2_errors:
        mean_err = np.mean(l2_errors)
        std_err = np.std(l2_errors)
        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Mean L2 Error: {mean_err:.6f} ± {std_err:.6f}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  No valid L2 errors found")
        print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    config = OptimizedConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        device=args.device
    )

    run_optimized_experiments(config)
