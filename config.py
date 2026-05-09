"""
Configuration module for QPINN experiments.
Supports multiple experiment setups with easy ablation studies and benchmarking.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import math


@dataclass
class PDEConfig:
    """PDE-specific parameters (heat equation by default)."""
    alpha: float = 0.1
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    
    def exact_solution(self, x, t):
        """Return exact solution u(x,t)."""
        import torch
        return torch.exp(-self.alpha * math.pi**2 * t) * torch.sin(math.pi * x)


@dataclass
class DataConfig:
    """Training data sampling parameters."""
    n_interior: int = 64      # interior points
    n_initial: int = 64       # initial condition points
    n_boundary: int = 64      # boundary points
    resample_each_epoch: bool = True  # resample points each epoch


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""
    feature_size: int = 4              # feature map output size
    quantum_output_size: int = 4       # quantum layer output size
    hidden_feature: int = 16           # hidden layer for feature map
    hidden_readout: int = 16           # hidden layer for readout
    use_hard_bc: bool = True           # enforce boundary conditions in architecture


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 300
    learning_rate: float = 1e-2
    optimizer_type: str = "Adam"  # "Adam", "SGD", "LBFGS", etc.
    
    # Loss weights
    lambda_pde: float = 1.0
    lambda_consistency: float = 0.1
    lambda_initial: float = 10.0
    lambda_boundary: float = 1.0
    
    # Optional: learning rate schedule
    lr_decay: Optional[str] = None  # None, "exponential", "step", "cosine"
    lr_decay_factor: float = 0.1
    lr_decay_steps: int = 100


@dataclass
class EvaluationConfig:
    """Evaluation grid parameters."""
    nx: int = 60
    nt: int = 60


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs."""
    name: str
    pde: PDEConfig = field(default_factory=PDEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Device and reproducibility
    seed: int = 1234
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "float32"
    
    # Logging
    log_every: int = 25
    save_checkpoint: bool = False
    checkpoint_dir: str = "./checkpoints"


# ============================================================================
# Experiment Presets
# ============================================================================

def baseline_config() -> ExperimentConfig:
    """Baseline configuration matching the original notebook."""
    return ExperimentConfig(
        name="baseline",
        pde=PDEConfig(),
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
    )


def low_data_config() -> ExperimentConfig:
    """Ablation: reduced training data."""
    cfg = baseline_config()
    cfg.name = "low_data"
    cfg.data = DataConfig(
        n_interior=32,
        n_initial=32,
        n_boundary=32,
    )
    return cfg


def high_lambda_ic_config() -> ExperimentConfig:
    """Ablation: increased initial condition weight."""
    cfg = baseline_config()
    cfg.name = "high_lambda_ic"
    cfg.training.lambda_initial = 50.0
    return cfg


def quantum_architecture_config() -> ExperimentConfig:
    """Ablation: larger quantum layer."""
    cfg = baseline_config()
    cfg.name = "quantum_large"
    cfg.model = ModelConfig(
        feature_size=8,
        quantum_output_size=8,
        hidden_feature=32,
        hidden_readout=32,
    )
    return cfg

def feature_size_sweep(f: int) -> ExperimentConfig:
    """Sweep over feature map size."""
    cfg = baseline_config()
    cfg.name = f"feature_{f}"
    cfg.model.feature_size = f
    cfg.model.quantum_output_size = f
    return cfg


def longer_training_config() -> ExperimentConfig:
    """Ablation: longer training with learning rate decay."""
    cfg = baseline_config()
    cfg.name = "longer_training"
    cfg.training.epochs = 1000
    cfg.training.lr_decay = "exponential"
    cfg.training.lr_decay_factor = 0.95
    cfg.training.lr_decay_steps = 50
    return cfg


def low_consistency_config() -> ExperimentConfig:
    """Ablation: reduced consistency loss weight."""
    cfg = baseline_config()
    cfg.name = "low_consistency"
    cfg.training.lambda_consistency = 0.01
    return cfg


def lr_sweep_config(lr: float) -> ExperimentConfig:
    """Learning rate sweep configuration."""
    cfg = baseline_config()
    cfg.name = f"lr_{lr:.0e}"
    cfg.training.learning_rate = lr
    return cfg


def lbfgs_config() -> ExperimentConfig:
    """Use LBFGS optimizer."""
    cfg = baseline_config()
    cfg.name = "lbfgs"
    cfg.training.optimizer_type = "LBFGS"
    cfg.training.epochs = 100  # LBFGS typically converges faster
    return cfg


# ============================================================================
# Config Registry for Easy Access
# ============================================================================

EXPERIMENT_REGISTRY = {
    "baseline": baseline_config,
    "low_data": low_data_config,
    "high_lambda_ic": high_lambda_ic_config,
    "quantum_large": quantum_architecture_config,
    "longer_training": longer_training_config,
    "low_consistency": low_consistency_config,
    "lbfgs": lbfgs_config,
    
    "feature_2": lambda: feature_size_sweep(2),
    "feature_4": lambda: feature_size_sweep(4),
    "feature_8": lambda: feature_size_sweep(8),
    "feature_16": lambda: feature_size_sweep(16),
}


def get_config(experiment_name: str) -> ExperimentConfig:
    """Get configuration by name from registry."""
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
        )
    return EXPERIMENT_REGISTRY[experiment_name]()


def list_experiments() -> list:
    """List all available experiment configurations."""
    return list(EXPERIMENT_REGISTRY.keys())
