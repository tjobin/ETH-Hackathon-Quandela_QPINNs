# QPINN Pipeline - File Reference & Quick Start

## 📁 File Structure

```
qpinn-pipeline/
├── config.py              # Configuration management
├── models.py              # Model implementations
├── losses.py              # Loss functions & data sampling
├── plotter.py             # Visualization utilities
├── main.py                # Main training pipeline
├── benchmark.py           # Benchmark & comparison suite
├── examples.py            # Usage examples
├── requirements.txt       # Dependencies
├── README.md              # Basic usage guide
├── GUIDE.md               # Comprehensive extension guide
└── QUICKSTART.md          # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Single Experiment
```bash
# Baseline configuration
python main.py --experiment baseline

# Any registered experiment
python main.py --experiment low_data
python main.py --experiment high_lambda_ic
python main.py --experiment quantum_large

# List all available experiments
python main.py --list-experiments
```

### Run Ablation Study
```bash
python benchmark.py --experiments baseline low_data high_lambda_ic
```

### Run All Experiments
```bash
python benchmark.py --experiments all
```

### Learning Rate Sweep
```bash
python benchmark.py --lr-sweep --lrs 1e-4 5e-4 1e-3 5e-3 1e-2
```

## 📝 File Descriptions

### config.py
**Purpose**: Centralized configuration management for all experiments

**Key Classes**:
- `PDEConfig`: PDE parameters (heat equation by default)
- `DataConfig`: Training data sampling parameters
- `ModelConfig`: Neural network architecture
- `TrainingConfig`: Optimizer and loss weights
- `EvaluationConfig`: Test grid resolution
- `ExperimentConfig`: Complete experiment configuration

**Key Functions**:
- `baseline_config()`: Reference configuration
- `get_config(name)`: Retrieve experiment by name
- `list_experiments()`: Show available experiments

**Pre-defined Ablations**:
- `baseline_config()`
- `low_data_config()`
- `high_lambda_ic_config()`
- `quantum_architecture_config()`
- `longer_training_config()`
- `low_consistency_config()`
- `lbfgs_config()`

### models.py
**Purpose**: Neural network model implementations

**Key Classes**:
- `MerlinHeatQPINN`: QPINN with MerLin quantum layer
- `ClassicalHeatQPINN`: Classical baseline (no quantum)
- `DeepNetworkQPINN`: Deeper classical network

**Key Functions**:
- `create_model(config, model_type)`: Factory function

**Model Types**:
- `"qpinn"`: MerLin quantum layer
- `"classical"`: Classical baseline
- `"deep_classical"`: Deeper classical network

### losses.py
**Purpose**: Physics-informed loss computation and data sampling

**Key Classes**:
- `DataSampler`: Generates training points (interior, initial, boundary)
- `PhysicsLoss`: Computes PDE residual, consistency, and boundary losses

**Key Methods**:
- `DataSampler.sample_interior()`: Interior points with requires_grad=True
- `DataSampler.sample_initial()`: Initial condition points
- `DataSampler.sample_boundary()`: Boundary condition points
- `PhysicsLoss.pde_residual()`: PDE constraint
- `PhysicsLoss.consistency_residual()`: Spatial derivative consistency
- `PhysicsLoss.initial_condition_loss()`: IC fit
- `PhysicsLoss.boundary_condition_loss()`: BC fit
- `PhysicsLoss.total_loss()`: Weighted sum of all losses

### plotter.py
**Purpose**: Visualization and analysis plotting

**Key Classes**:
- `Plotter`: Handles all visualization tasks

**Key Methods**:
- `plot_training_history()`: Loss curves
- `plot_solution_comparison()`: Exact vs predicted vs error heatmaps
- `plot_solution_slices()`: 1D slices at different times/positions
- `plot_loss_comparison()`: Compare multiple experiments
- `plot_error_metrics()`: Bar charts of metrics

### main.py
**Purpose**: Main training and evaluation pipeline

**Key Classes**:
- `Trainer`: Handles training loop, optimizer, scheduler
- `Evaluator`: Evaluates model on test grid, computes metrics

**Key Functions**:
- `run_experiment(name, model_type, output_dir)`: Complete training → evaluation → plotting

**Features**:
- Multiple optimizer types: Adam, SGD, LBFGS
- Learning rate scheduling: exponential, step, cosine
- Checkpoint saving/loading
- Automatic loss tracking and plotting

### benchmark.py
**Purpose**: Orchestrate multiple experiments for comparison

**Key Classes**:
- `BenchmarkSuite`: Manages batch experiments

**Key Methods**:
- `run_experiments(names, model_types)`: Run multiple configurations
- `lr_sweep(lrs)`: Learning rate sweep
- `generate_comparison_report()`: Extract and compare metrics
- `plot_comparisons()`: Generate comparison plots
- `print_summary()`: Print results table

## 📊 Configuration Quick Reference

### Essential Parameters

```python
config.training.epochs = 300
config.training.learning_rate = 1e-2
config.training.optimizer_type = "Adam"  # or "SGD", "LBFGS"

config.training.lambda_pde = 1.0           # PDE residual weight
config.training.lambda_consistency = 0.1   # Spatial derivative consistency
config.training.lambda_initial = 10.0      # Initial condition weight
config.training.lambda_boundary = 1.0      # Boundary condition weight

config.data.n_interior = 64
config.data.n_initial = 64
config.data.n_boundary = 64

config.model.feature_size = 4
config.model.quantum_output_size = 4
config.model.use_hard_bc = True
```

### Learning Rate Schedules

```python
config.training.lr_decay = None          # No decay
config.training.lr_decay = "exponential" # Exponential decay
config.training.lr_decay = "step"        # Step decay
config.training.lr_decay = "cosine"      # Cosine annealing
config.training.lr_decay_factor = 0.1
config.training.lr_decay_steps = 100
```

## 🎯 Common Use Cases

### 1. Run Baseline
```bash
python main.py --experiment baseline
```

### 2. Compare Models
```bash
python benchmark.py --experiments baseline \
  --model-types qpinn classical deep_classical
```

### 3. Ablation: Data Size
```bash
python benchmark.py --experiments baseline low_data
```

### 4. Ablation: Loss Weights
```bash
python benchmark.py --experiments \
  baseline high_lambda_ic low_consistency
```

### 5. Ablation: Architecture
```bash
python benchmark.py --experiments \
  baseline quantum_large
```

### 6. Training Duration
```bash
python benchmark.py --experiments \
  baseline longer_training
```

### 7. Optimizer Comparison
```bash
python benchmark.py --experiments baseline lbfgs
```

### 8. Learning Rate Sweep
```bash
python benchmark.py --lr-sweep
```

## 📈 Output Organization

Each experiment creates:
```
results/
└── {experiment_name}_{model_type}/
    ├── best_model.pt              # Saved model + config + history
    ├── results.json               # Metrics summary
    ├── training_history.png       # Loss curves
    ├── solution_comparison.png    # Heatmap comparison
    └── solution_slices.png        # 1D profile plots
```

Benchmark suites create:
```
benchmarks/
└── {YYYYMMDD_HHMMSS}/
    ├── {exp1_model1}/             # Individual experiment results
    ├── {exp1_model2}/
    ├── {exp2_model1}/
    ├── ...
    ├── benchmark_report.json      # Consolidated metrics
    ├── loss_comparison.png        # Comparative plots
    └── metrics_comparison.png
```

## 🔧 Customization Examples

### Create New Experiment
```python
# In config.py
def my_test_config() -> ExperimentConfig:
    cfg = baseline_config()
    cfg.name = "my_test"
    cfg.training.epochs = 500
    cfg.training.learning_rate = 5e-3
    return cfg

EXPERIMENT_REGISTRY["my_test"] = my_test_config

# Then run:
# python main.py --experiment my_test
```

### Custom Training Loop
```python
from config import baseline_config
from main import Trainer, Evaluator
from models import create_model

config = baseline_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

model = create_model(config.model).to(device, dtype)
trainer = Trainer(model, config, device, dtype)
history = trainer.train()

evaluator = Evaluator(model, config, device, dtype)
U_pred, U_true, x, t, metrics = evaluator.evaluate()

print(f"RelL2 Error: {metrics['rel_l2']:.4e}")
```

## 📚 Documentation

- **README.md**: Basic usage guide and feature overview
- **GUIDE.md**: Comprehensive extension guide (new models, PDEs, losses)
- **examples.py**: 8 detailed usage examples with code

## 🎓 Learning Path

1. **Start Here**: `README.md` - Understand basic usage
2. **Run Baseline**: `python main.py --experiment baseline`
3. **Browse Examples**: `examples.py` - See different use patterns
4. **Try Ablations**: `python benchmark.py --experiments baseline low_data`
5. **Customize**: `GUIDE.md` - Learn to add new models/experiments
6. **Extend**: Implement your own models, PDEs, or loss functions

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| MerLin import error | Install: `pip install merlinquantum` or use `--model-type classical` |
| CUDA out of memory | Reduce `config.data.n_interior`, `n_initial`, `n_boundary` |
| NaN loss | Reduce learning rate or increase `lambda_initial` |
| Slow convergence | Try LBFGS optimizer or increase `lambda_consistency` |
| Bad solution | Increase epochs or try longer training config |

## 📞 Support

- Check README.md for FAQ
- Review GUIDE.md for advanced customization
- Run examples.py to see usage patterns
- Inspect config.py for all available settings

---

**Version**: 1.0  
**Created for**: QPINN ETH Hackathon Project  
**Tested with**: PyTorch 2.0+, Python 3.8+
