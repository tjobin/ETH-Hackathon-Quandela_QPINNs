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
    
    def initial_condition(self, x):
        """Return IC: u(x, 0) = f(x)"""
        import torch
        #return torch.sin(math.pi * x)
        return torch.sin(math.pi * x)
    
    def boundary_condition_left(self, t):
        """Return BC: u(0, t) = g1(t)"""
        import torch
        return torch.zeros_like(t)
    
    def boundary_condition_right(self, t):
        """Return BC: u(1, t) = g2(t)"""
        import torch
        return torch.zeros_like(t)
    

@dataclass
class BurgersEquationConfig(PDEConfig):
    """
    Burgers equation: du/dt + u*du/dx = nu*d²u/dx²
    
    Test case:
    - IC: u(x,0) = sin(pi*x)
    - BC: u(0,t) = 0, u(1,t) = 0
    - Reference: high-accuracy numerical solution
    """
    nu: float = 0.01  # Viscosity coefficient
    
    def exact_solution(self, x, t):
        """
        Reference solution: pure diffusion approximation.
        u(x,t) ≈ sin(pi*x)*exp(-nu*pi²*t)
        
        This ignores the nonlinear u*du/dx term but gives a reasonable reference.
        The actual Burgers solution would need numerical computation.
        """
        import torch
        import math
        
        u = torch.sin(math.pi * x) * torch.exp(-self.nu * (math.pi ** 2) * t)
        
        return u
    
    def initial_condition(self, x):
        """IC: u(x, 0) = sin(pi*x)"""
        import torch
        import math
        return torch.sin(math.pi * x)
    
    def boundary_condition_left(self, t):
        """BC: u(0, t) = 0"""
        import torch
        return torch.zeros_like(t)
    
    def boundary_condition_right(self, t):
        """BC: u(1, t) = 0"""
        import torch
        return torch.zeros_like(t)
 
 
@dataclass
class SchrodingerEquationConfig(PDEConfig):
    """
    Schrodinger equation: i*du/dt + d²u/dx² = 0 (1D time-dependent)
    
    Test case:
    - IC: u(x,0) = cos(x)  [real part of exp(i*x)]
    - BC: u(0,t) = cos(-t), u(1,t) = cos(1-t)
    - Exact solution: u(x,t) = cos(x - t)  [real part of exp(i*(x-t))]
    """
    
    def exact_solution(self, x, t):
        """Exact solution: u(x,t) = cos(x - t)"""
        import torch
        u = torch.cos(x - t)
        return u
    
    def initial_condition(self, x):
        """IC: u(x, 0) = cos(x)"""
        import torch
        return torch.cos(x)
    
    def boundary_condition_left(self, t):
        """BC: u(0, t) = cos(-t) = cos(t)"""
        import torch
        return torch.cos(t)
    
    def boundary_condition_right(self, t):
        """BC: u(1, t) = cos(1-t)"""
        import torch
        return torch.cos(1.0 - t)
 
 
@dataclass
class WaveEquationConfig(PDEConfig):
    """
    1D Wave equation: d²u/dt² = c² * d²u/dx²
    
    Test case (plucked string):
    - IC: u(x,0) = sin(pi*x), du/dt(x,0) = 0
    - BC: u(0,t) = 0, u(1,t) = 0
    - Exact solution: u(x,t) = sin(pi*x)*cos(pi*c*t)
    """
    c: float = 1.0  # Wave speed
    
    def exact_solution(self, x, t):
        """Exact solution: plucked string."""
        import torch
        import math
        u = torch.sin(math.pi * x) * torch.cos(math.pi * self.c * t)
        return u
    
    def initial_condition(self, x):
        """IC: u(x, 0) = sin(pi*x)"""
        import torch
        import math
        return torch.sin(math.pi * x)
    
    def boundary_condition_left(self, t):
        """BC: u(0, t) = 0"""
        import torch
        return torch.zeros_like(t)
    
    def boundary_condition_right(self, t):
        """BC: u(1, t) = 0"""
        import torch
        return torch.zeros_like(t)
 


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
    quantum_type: str = "simple"       # "simple" or "builder"


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

    # Layer freezing options
    freeze_quantum: bool = False
    freeze_feature_map: bool = False
    freeze_readout: bool = False


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

def freeze_quantum_config() -> ExperimentConfig:
    """Only train classical layers, freeze quantum."""
    cfg = baseline_config()
    cfg.name = "freeze_quantum"
    cfg.training.freeze_quantum = True
    return cfg

def freeze_feature_map_config() -> ExperimentConfig:
    """Only train readout, freeze feature map and quantum."""
    cfg = baseline_config()
    cfg.name = "freeze_feature_map"
    cfg.training.freeze_quantum = True
    cfg.training.freeze_feature_map = True
    return cfg

def feature_size_sweep(f: int) -> ExperimentConfig:
    """Sweep over feature map size."""
    cfg = baseline_config()
    cfg.name = f"feature_{f}"
    cfg.model.feature_size = f
    cfg.model.quantum_output_size = f
    return cfg

def burgers_equation_config() -> ExperimentConfig:
    """Burgers equation instead of heat equation."""
    cfg = baseline_config()
    cfg.name = "burgers_equation"
    cfg.pde = BurgersEquationConfig(nu=0.01)
    cfg.training.epochs = 2000  # Might need more epochs
    cfg.training.lambda_pde = 1.0
    cfg.training.lambda_consistency = 0.15
    cfg.training.lambda_initial = 10.0
    return cfg
 
 
def schrodinger_equation_config() -> ExperimentConfig:
    """Schrodinger equation (complex-valued)."""
    cfg = baseline_config()
    cfg.name = "schrodinger_equation"
    cfg.pde = SchrodingerEquationConfig()
    cfg.training.epochs = 500
    # Note: Schrodinger needs special handling for complex numbers
    return cfg
 
 
def wave_equation_config() -> ExperimentConfig:
    """1D Wave equation."""
    cfg = baseline_config()
    cfg.name = "wave_equation"
    cfg.pde = WaveEquationConfig(c=1.0)
    cfg.training.epochs = 2000
    cfg.training.lambda_initial = 15.0  # Higher IC weight
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

def deep_builder_config() -> ExperimentConfig:
    """Configuration for the 4-mode, 3-photon CircuitBuilder with data re-uploading."""
    cfg = baseline_config()
    cfg.name = "deep_quantum_builder"
    cfg.model = ModelConfig(
        # The builder encodes data into exactly 2 modes (modes 0 and 1)
        feature_size=4,              
        # 4 modes with 3 photons yields C(4+3-1, 3) = 20 distinct Fock basis states
        # The probs() measurement outputs a vector of this exact size
        quantum_output_size=4,      
        hidden_feature=16,
        hidden_readout=16,
        quantum_type="builder"       # Triggers the new model class
    )
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
    "deep_builder": deep_builder_config,
    
    "feature_2": lambda: feature_size_sweep(2),
    "feature_4": lambda: feature_size_sweep(4),
    "feature_8": lambda: feature_size_sweep(8),
    "feature_16": lambda: feature_size_sweep(16),

    "freeze_quantum_config": freeze_quantum_config,
    "freeze_feature_map_config": freeze_feature_map_config,

    "burgers_equation": burgers_equation_config,
    "schrodinger_equation": schrodinger_equation_config,
    "wave_equation": wave_equation_config,
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
