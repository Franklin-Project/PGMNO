# -*- coding: utf-8 -*-
"""
Unified hyperparameter configuration (Response Letter Table A1).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
from pathlib import Path



@dataclass
class SystemConfig:
    """System configuration for checkpointing and logging"""
    save_frequency: int = 10


@dataclass
class BurgersConfig:
    """Burgers equation configuration (Table A1)"""
    # Spatial discretization
    n_grid: int = 128          # Spatial grid points (N)
    x_range: tuple = (-1, 1)   # Spatial domain

    # Temporal discretization
    dt: float = 0.0025           # Time step (Δt) - Reduced for CFL stability
    n_time_steps: int = 100    # Prediction horizon (T)

    # Physical parameters
    nu: float = 0.01 / 3.14159265359  # Viscosity coefficient

    # Data generation
    n_train: int = 1000        # Training samples
    n_val: int = 100           # Validation samples
    n_test: int = 100          # Test samples
    noise_level: float = 0.0   # Data noise level


@dataclass
class ModelConfig:
    """Model architecture configuration (Table A1)"""
    # BDF parameters
    k_steps: int = 2           # BDF order (k)

    # Causal weighting
    epsilon: float = 0.05      # Causal weight parameter (ε)

    # Architecture
    hidden_dim: int = 64       # Hidden dimension
    num_layers: int = 4        # Number of Mamba layers
    ssm_state_dim: int = 64    # SSM state dimension (N_z)

    # Regularization
    dropout: float = 0.1
    use_conservation: bool = True
    use_adaptive_physics: bool = True


@dataclass
class TrainingConfig:
    """Training configuration (Table A1)"""
    # Optimizer
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    weight_decay: float = 1e-4

    # Scheduler
    scheduler_type: str = "cosine"
    eta_min: float = 1e-6      # Minimum learning rate for cosine annealing
    T_max: int = 500           # Total epochs for cosine annealing

    # Training parameters
    batch_size: int = 32
    n_epochs: int = 500

    # Loss weights
    phys_weight: float = 0.1   # Physics loss weight

    # Fixed horizon for all training
    horizon: int = 20

    # Early stopping
    patience: int = 50  # Increased from 20 to allow full training
    min_delta: float = 1e-6


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Random seeds for reproducibility (3 seeds as per Response Letter)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Output directories
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"

    # Device
    device: str = "cuda"


@dataclass
class ReviewerResponseConfig:
    """
    Complete configuration for reviewer response experiments.

    This dataclass aggregates all configurations to ensure consistency
    with Response Letter Table A1.
    """
    burgers: BurgersConfig = field(default_factory=BurgersConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ReviewerResponseConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            burgers=BurgersConfig(**data['burgers']),
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**data['training']),
            experiment=ExperimentConfig(**data['experiment']),
            system=SystemConfig(**data.get('system', {}))
        )


# =============================================================================
# Pre-defined configurations for experiments
# =============================================================================

def get_burgers_baseline_config() -> ReviewerResponseConfig:
    """
    Get baseline Burgers configuration (Table A1).
    This is the main configuration used for all experiments.
    """
    return ReviewerResponseConfig()


def get_ablation_configs() -> Dict[str, ReviewerResponseConfig]:
    """
    Generate 6 ablation experiment configurations.

    Returns:
        Dictionary mapping ablation name to configuration

    Ablation experiments (Response Letter Table 3b):
    1. Full_PGMNO - Complete model (baseline)
    2. wo_Multistep - k=1 (no multi-step context)
    3. wo_BDF_Loss - No physics loss
    4. wo_Causal_Weight - ε=0 (no causal weighting)
    5. Transformer_backbone - Transformer replaces Mamba
    6. FNO_backbone - FNO replaces Mamba
    """
    configs = {}

    # 1. Full PGMNO (baseline)
    configs['Full_PGMNO'] = get_burgers_baseline_config()

    # 2. Without multi-step (k=1)
    wo_multistep = get_burgers_baseline_config()
    wo_multistep.model.k_steps = 1
    configs['wo_Multistep'] = wo_multistep

    # 3. Without BDF loss (phys_weight=0)
    wo_bdf_loss = get_burgers_baseline_config()
    wo_bdf_loss.training.phys_weight = 0.0
    configs['wo_BDF_Loss'] = wo_bdf_loss

    # 4. Without causal weight (ε=0)
    wo_causal = get_burgers_baseline_config()
    wo_causal.model.epsilon = 0.0
    configs['wo_Causal_Weight'] = wo_causal

    # 5. Transformer backbone
    transformer_config = get_burgers_baseline_config()
    # Mark for backbone swap (handled in training script)
    configs['Transformer_backbone'] = transformer_config

    # 6. FNO backbone
    fno_config = get_burgers_baseline_config()
    # Mark for backbone swap (handled in training script)
    configs['FNO_backbone'] = fno_config

    return configs


def get_epsilon_sweep_configs() -> Dict[str, ReviewerResponseConfig]:
    """
    Generate ε (causal weight) sweep configurations.

    Sweep values: 0.0, 0.01, 0.05, 0.1, 0.2
    (Response Letter Table 5a)
    """
    epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.2]
    configs = {}

    for eps in epsilon_values:
        config = get_burgers_baseline_config()
        config.model.epsilon = eps
        configs[f'epsilon_{eps}'] = config

    return configs


def get_bdf_order_sweep_configs() -> Dict[str, ReviewerResponseConfig]:
    """
    Generate BDF order (k) sweep configurations.

    Sweep values: 1, 2, 3, 4, 5
    (Response Letter Table 4a)
    """
    k_values = [1, 2, 3, 4, 5]
    configs = {}

    for k in k_values:
        config = get_burgers_baseline_config()
        config.model.k_steps = k
        configs[f'bdf_k{k}'] = config

    return configs


def get_horizon_sweep_configs() -> Dict[str, ReviewerResponseConfig]:
    """
    Generate prediction horizon sweep configurations.

    Sweep values: T = 10, 50, 100, 200
    (Long-horizon advantage verification)
    """
    horizon_values = [10, 50, 100, 200]
    configs = {}

    for T in horizon_values:
        config = get_burgers_baseline_config()
        config.burgers.n_time_steps = T
        config.training.full_horizon = T
        configs[f'horizon_T{T}'] = config

    return configs


# =============================================================================
# Expected results (for validation)
# =============================================================================

# Expected ablation results (Response Letter Table 3b)
EXPECTED_ABLATION_RESULTS = {
    'Full_PGMNO': {
        'l2_error': 0.00231,
        'std': 0.0008,
        'vs_baseline': 'baseline'
    },
    'wo_Multistep': {
        'l2_error': 0.00341,
        'std': 0.0012,
        'vs_baseline': '+47.6%'
    },
    'wo_BDF_Loss': {
        'l2_error': 0.00298,
        'std': 0.0010,
        'vs_baseline': '+29.0%'
    },
    'wo_Causal_Weight': {
        'l2_error': 0.00267,
        'std': 0.0009,
        'vs_baseline': '+15.6%'
    },
    'Transformer_backbone': {
        'l2_error': 0.00289,
        'std': 0.0012,
        'vs_baseline': '+25.1%'
    },
    'FNO_backbone': {
        'l2_error': 0.00157,
        'std': 0.0001,
        'vs_baseline': '-32.0%'
    }
}

# Expected physics metrics (Response Letter Table 5)
EXPECTED_PHYSICS_METRICS = {
    'PGMNO': {
        'pde_residual': 0.0089,
        'mass_conservation': 0.0023,
        'energy_conservation': 0.0045
    },
    'FNO': {
        'pde_residual': 0.0456,
        'mass_conservation': 0.0089,
        'energy_conservation': 0.0178
    },
    'Transformer': {
        'pde_residual': 0.0789,
        'mass_conservation': 0.0134,
        'energy_conservation': 0.0256
    }
}


# =============================================================================
# Utility functions
# =============================================================================

def print_config_summary(config: ReviewerResponseConfig):
    """Print a formatted summary of the configuration"""
    print("=" * 60)
    print("PGMNO Reviewer Response Configuration")
    print("=" * 60)
    print("\n[Burgers Equation]")
    print(f"  Spatial grid (N):      {config.burgers.n_grid}")
    print(f"  Time step (Δt):        {config.burgers.dt}")
    print(f"  Prediction horizon:    {config.burgers.n_time_steps}")
    print(f"  Training samples:      {config.burgers.n_train}")

    print("\n[Model Architecture]")
    print(f"  BDF order (k):         {config.model.k_steps}")
    print(f"  Causal weight (ε):     {config.model.epsilon}")
    print(f"  Hidden dimension:      {config.model.hidden_dim}")
    print(f"  Mamba layers:          {config.model.num_layers}")
    print(f"  SSM state dim:         {config.model.ssm_state_dim}")

    print("\n[Training]")
    print(f"  Learning rate:         {config.training.learning_rate}")
    print(f"  Batch size:            {config.training.batch_size}")
    print(f"  Epochs:                {config.training.n_epochs}")
    print(f"  Physics weight:        {config.training.phys_weight}")
    print(f"  Optimizer:             {config.training.optimizer_type}")

    print("\n[Experiment]")
    print(f"  Seeds:                 {config.experiment.seeds}")
    print(f"  Device:                {config.experiment.device}")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    config = get_burgers_baseline_config()
    print_config_summary(config)

    print("\n\nAblation Configurations:")
    for name, cfg in get_ablation_configs().items():
        print(f"  - {name}: k={cfg.model.k_steps}, ε={cfg.model.epsilon}, phys_weight={cfg.training.phys_weight}")

    print("\nEpsilon Sweep Configurations:")
    for name, cfg in get_epsilon_sweep_configs().items():
        print(f"  - {name}: ε={cfg.model.epsilon}")

    print("\nBDF Order Sweep Configurations:")
    for name, cfg in get_bdf_order_sweep_configs().items():
        print(f"  - {name}: k={cfg.model.k_steps}")
