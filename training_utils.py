# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import logging
from pathlib import Path
import json
from collections import defaultdict

from improved_burgers_utils import physics_loss_burgers_v2, compute_pde_metrics
from config_manager import PGMNOConfig


class TrainingLogger:
    """Training logger with metrics history."""
    def __init__(self, log_dir: str, experiment_name: str = "pgmno"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = defaultdict(list)
        self.learning_rates = []

        # Time tracking
        self.start_time = time.time()
        self.epoch_times = []

    def _setup_logging(self):
        """Setup logging format."""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  metrics: Optional[Dict] = None, lr: Optional[float] = None):
        """Log epoch information."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if metrics:
            for k, v in metrics.items():
                self.metrics_history[k].append(v)
        if lr is not None:
            self.learning_rates.append(lr)

        # Log to file
        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        if metrics:
            log_msg += f", Metrics = {metrics}"
        if lr is not None:
            log_msg += f", LR = {lr:.2e}"

        self.logger.info(log_msg)

    def log_time(self, epoch: int, epoch_time: float):
        """Log training time."""
        self.epoch_times.append(epoch_time)
        total_time = time.time() - self.start_time
        self.logger.info(f"Epoch {epoch} time: {epoch_time:.2f}s, Total time: {total_time:.2f}s")

    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': dict(self.metrics_history),
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }

        history_file = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def plot_training_curves(self, save: bool = True):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)

        # Evaluation metrics
        if self.metrics_history:
            for i, (metric, values) in enumerate(self.metrics_history.items()):
                row = (i + 2) // 2
                col = (i + 2) % 2
                if row < 2 and col < 2:
                    axes[row, col].plot(values, label=metric)
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel(metric)
                    axes[row, col].set_title(f'Metric: {metric}')
                    axes[row, col].legend()
                    axes[row, col].grid(True)

        plt.tight_layout()

        if save:
            plot_file = self.log_dir / f"{self.experiment_name}_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {plot_file}")

        return fig


class EarlyStopping:
    """Early stopping mechanism."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class WeightedLossCalculator:
    """
    Causal weighted loss calculator.

    Implements Eq.(19): ω_i = exp(-ε Σ_{j=0}^{i-1} L_j)
    """
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def compute_weighted_loss(self, step_losses: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Compute causal weighted loss.

        Args:
            step_losses: list of losses [L_0, L_1, ..., L_{L-1}]
            device: compute device

        Returns:
            weighted_total_loss: causal weighted total Σ ω_i L_i
        """
        import math
        weighted_total_loss = torch.zeros(1, device=device)
        cumulative_loss = 0.0

        for i, loss in enumerate(step_losses):
            # Weight based on cumulative loss from previous steps
            weight = math.exp(-self.epsilon * cumulative_loss)
            weighted_total_loss = weighted_total_loss + weight * loss
            cumulative_loss += loss.item()

        return weighted_total_loss


class ModelValidator:
    """Model validator with multi-step prediction support."""
    def __init__(self, config: PGMNOConfig):
        self.config = config

    def validate_model(self, model: nn.Module, val_loader: DataLoader,
                       device: torch.device, dx: float, nu: float) -> Dict:
        """Validate model performance."""
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_metrics = defaultdict(list)

        with torch.no_grad():
            # Pre-compute base x_grid
            first_batch = next(iter(val_loader))
            _, _, n_grid = first_batch[0].shape
            x_grid_base = torch.linspace(-1, 1, n_grid, device=device)
            del first_batch

            for batch_idx, (batch_in, batch_target) in enumerate(val_loader):
                batch_in = batch_in.to(device)
                batch_target = batch_target.to(device)

                # Multi-step prediction
                batch_size, k, n_grid = batch_in.shape
                predictions = []
                current_history = batch_in.clone()

                # Prepare grid coordinates
                x_grid = x_grid_base.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)

                step_losses = []
                for step in range(min(self.config.training.forecast_horizon, batch_target.size(1))):
                    # Predict next step
                    u_pred, reg_loss = model(current_history, x_grid)
                    u_true = batch_target[:, step, :]

                    # Compute losses
                    mse_loss = nn.functional.mse_loss(u_pred, u_true)
                    # Pass full history for BDF-k support
                    phys_loss = physics_loss_burgers_v2(
                        u_pred, current_history,
                        dt=self.config.training.dt,
                        dx=dx, nu=nu
                    )
                    total_step_loss = mse_loss + self.config.training.phys_weight * phys_loss
                    step_losses.append(total_step_loss)

                    # Compute metrics for first step
                    if step == 0:
                        metrics = compute_pde_metrics(u_pred, u_true,
                                                     self.config.training.dt, dx, nu)
                        for k, v in metrics.items():
                            total_metrics[k].append(v)

                    predictions.append(u_pred.unsqueeze(1))

                    # Update history
                    u_pred_expanded = u_pred.unsqueeze(1)
                    current_history = torch.cat([current_history[:, 1:, :], u_pred_expanded], dim=1)

                # Weighted loss
                weight_calculator = WeightedLossCalculator(self.config.training.epsilon)
                weighted_loss = weight_calculator.compute_weighted_loss(step_losses, device)
                total_loss += weighted_loss

        # Compute average metrics
        avg_metrics = {}
        for k, v in total_metrics.items():
            avg_metrics[f'val_{k}'] = np.mean(v)

        avg_metrics['val_loss'] = (total_loss / len(val_loader)).item()

        return avg_metrics

    def visualize_predictions(self, model: nn.Module, val_loader: DataLoader,
                             device: torch.device, save_path: Optional[str] = None):
        """Visualize prediction results."""
        model.eval()

        with torch.no_grad():
            # Get one batch
            batch_in, batch_target = next(iter(val_loader))
            batch_in = batch_in.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            # Prepare grid coordinates
            batch_size, k, n_grid = batch_in.shape
            x_grid = torch.linspace(-1, 1, n_grid, device=device)
            x_grid = x_grid.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)

            # Predict
            predictions = model.predict_multi_step(
                batch_in, x_grid,
                n_steps=min(self.config.training.forecast_horizon, 10)
            )

            # Visualize first sample
            sample_idx = 0
            x_cpu = x_grid[sample_idx, :, 0].cpu().numpy()

            # Plot
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))

            # History
            for i in range(k):
                axes[0, 0].plot(x_cpu, batch_in[sample_idx, i, :].cpu().numpy(),
                              label=f'History t-{k-i-1}', alpha=0.7)
            axes[0, 0].set_title('History')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Prediction vs ground truth
            n_show = min(predictions.size(1), 5)
            for i in range(n_show):
                ax = axes[(i+1)//3, (i+1)%3]
                pred = predictions[sample_idx, i, :].cpu().numpy()
                true = batch_target[sample_idx, i, :].cpu().numpy()
                ax.plot(x_cpu, pred, label='Prediction', linewidth=2)
                ax.plot(x_cpu, true, label='Ground Truth', linestyle='--')
                ax.set_title(f'Step t+{i+1}')
                ax.legend()
                ax.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

            return fig


class CheckpointManager:
    """Checkpoint manager for saving/loading model states."""
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Optional[object], epoch: int,
                       metrics: Dict, config: PGMNOConfig,
                       is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config.to_dict(),
            'timestamp': time.time()
        }

        # Save latest checkpoint
        checkpoint_file = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_file)

        # Save best checkpoint
        if is_best:
            best_file = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_file)

        # Periodic save
        if (epoch + 1) % config.system.save_frequency == 0:
            epoch_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_file)

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[object] = None):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint.get('metrics', {})


def setup_optimizer_scheduler(model: nn.Module, config: PGMNOConfig):
    """Setup optimizer and learning rate scheduler."""
    # Optimizer
    if config.training.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            **config.training.optimizer_params
        )
    elif config.training.optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            **config.training.optimizer_params
        )
    elif config.training.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay,
            **config.training.optimizer_params
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.training.optimizer_type}")

    # Learning rate scheduler
    scheduler = None
    if config.training.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.n_epochs,
            **config.training.scheduler_params
        )
    elif config.training.scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.n_epochs // 3,
            gamma=0.1,
            **config.training.scheduler_params
        )
    elif config.training.scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            **config.training.scheduler_params
        )

    return optimizer, scheduler