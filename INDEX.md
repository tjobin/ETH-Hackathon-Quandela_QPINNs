# QPINN Analysis Pipeline - Complete Index

## 📚 Documentation Files

### 1. **QUICKSTART.md** (START HERE!)
   - **Purpose**: Quick reference and getting started guide
   - **Contents**:
     - Installation instructions
     - Common command examples
     - File descriptions
     - Configuration quick reference
     - Common use cases
     - Troubleshooting table
   - **Read time**: 10 minutes

### 2. **README.md**
   - **Purpose**: Feature overview and basic usage
   - **Contents**:
     - Project structure
     - Features list
     - Installation
     - Basic usage (single experiments, benchmarks)
     - Configuration guide
     - Model types
     - Advanced usage
   - **Read time**: 15 minutes

### 3. **GUIDE.md** (Comprehensive!)
   - **Purpose**: Deep dive into extending the pipeline
   - **Contents**:
     - Architecture overview
     - Adding new experiments (3 methods)
     - Adding new models (with templates)
     - Adding new PDEs
     - Advanced loss functions
     - Custom training loops
     - Custom plotting
     - Performance optimization
   - **Read time**: 30-45 minutes
   - **Best for**: Developers extending the framework

### 4. **ARCHITECTURE.md**
   - **Purpose**: System design and data flow diagrams
   - **Contents**:
     - System design flowchart
     - Data flow for single experiment
     - Component interactions
     - Training loop flow
     - Loss computation flow
     - Benchmark suite flow
     - Configuration hierarchy
     - Loss function composition
     - Model architecture flow
   - **Visual heavy** - great for understanding system design

## 💻 Python Source Files

### 1. **config.py** (5.9 KB)
   **Core: Configuration Management**
   - `PDEConfig`: PDE parameters and exact solution
   - `DataConfig`: Training data sampling
   - `ModelConfig`: Network architecture
   - `TrainingConfig`: Optimizer and loss weights
   - `EvaluationConfig`: Test grid size
   - `ExperimentConfig`: Top-level configuration
   - Pre-built configs: baseline, low_data, high_lambda_ic, quantum_large, etc.
   - `EXPERIMENT_REGISTRY`: Central registry for all experiments
   - Functions: `get_config()`, `list_experiments()`

### 2. **models.py** (5.4 KB)
   **Core: Neural Network Models**
   - `MerlinHeatQPINN`: QPINN with quantum layer
   - `ClassicalHeatQPINN`: Classical baseline
   - `DeepNetworkQPINN`: Deeper classical variant
   - `create_model()`: Factory function
   - Support for hard boundary condition enforcement
   - Easy to extend with new architectures

### 3. **losses.py** (7.7 KB)
   **Core: Physics Loss & Data Sampling**
   - `DataSampler`: Generates training points
     - `sample_interior()`: Interior points with gradients
     - `sample_initial()`: Initial condition points
     - `sample_boundary()`: Boundary points
   - `PhysicsLoss`: Computes loss terms
     - `gradient()`: Automatic differentiation helper
     - `pde_residual()`: PDE constraint
     - `consistency_residual()`: Spatial derivative matching
     - `initial_condition_loss()`: IC fitting
     - `boundary_condition_loss()`: BC fitting
     - `total_loss()`: Weighted combination

### 4. **main.py** (13 KB)
   **Core: Training Pipeline**
   - `Trainer`: Main training loop
     - Handles optimization, scheduling, checkpointing
     - Support for Adam, SGD, LBFGS
     - Learning rate schedules: exponential, step, cosine
   - `Evaluator`: Model evaluation
     - Computes metrics on test grid
     - Relative L2, MAE, RMSE, max error
   - `run_experiment()`: High-level interface
     - Complete pipeline: train → evaluate → plot
   - CLI with argparse

### 5. **plotter.py** (11 KB)
   **Core: Visualization & Analysis**
   - `Plotter`: Handles all plotting
     - `plot_training_history()`: Loss curves
     - `plot_solution_comparison()`: Heatmaps
     - `plot_solution_slices()`: 1D profiles
     - `plot_loss_comparison()`: Multi-experiment comparison
     - `plot_error_metrics()`: Bar charts
   - Optional saving to disk
   - Customizable figures and titles

### 6. **benchmark.py** (10 KB)
   **Orchestration: Batch Experiments**
   - `BenchmarkSuite`: Manages multiple experiments
     - `run_experiments()`: Batch training
     - `lr_sweep()`: Learning rate sweep
     - `generate_comparison_report()`: Extract metrics
     - `plot_comparisons()`: Comparative plots
     - `save_report()`, `print_summary()`
   - CLI with argparse
   - Timestamped output organization

### 7. **examples.py** (15 KB)
   **Educational: 8 Usage Examples**
   1. Single experiment
   2. Custom configuration
   3. Ablation study
   4. Model comparison
   5. Learning rate sweep
   6. Custom PDE
   7. Benchmark suite
   8. Advanced custom training
   - Runnable examples: `python examples.py 1`
   - Copy-paste ready code

## 📦 Configuration & Dependencies

### requirements.txt
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
merlinquantum>=0.1.0  # Optional
```

## 🎯 Usage Flows

### Single Experiment
```
config.py → models.py → main.Trainer → main.Evaluator → plotter.Plotter
```

### Ablation Study
```
benchmark.py → run_experiment (multiple times) → comparison plots
```

### Custom Training
```
config.py → create custom config → Trainer/Evaluator → plotter
```

## 📊 File Relationships

```
config.py
  ├─ Used by: main.py, models.py, losses.py
  └─ Imports: dataclasses, math, typing

models.py
  ├─ Depends on: config.ModelConfig
  ├─ Used by: main.py
  └─ Imports: torch, config

losses.py
  ├─ Depends on: config.PDEConfig, config.DataConfig
  ├─ Used by: main.py
  └─ Imports: torch, config

main.py
  ├─ Depends on: config, models, losses
  ├─ Uses: plotter (optional)
  └─ Entry point for single experiments

plotter.py
  ├─ No core dependencies
  ├─ Used by: main.py, benchmark.py
  └─ Imports: torch, numpy, matplotlib

benchmark.py
  ├─ Depends on: config, main, plotter
  └─ Entry point for batch experiments
```

## 🚀 Quick Command Reference

```bash
# Single experiments
python main.py --experiment baseline
python main.py --experiment low_data
python main.py --list-experiments

# Ablation studies
python benchmark.py --experiments baseline low_data high_lambda_ic
python benchmark.py --experiments all

# Model comparisons
python benchmark.py --experiments baseline --model-types qpinn classical

# Learning rate sweep
python benchmark.py --lr-sweep --lrs 1e-4 5e-4 1e-3 5e-3 1e-2

# Examples
python examples.py 1  # Single experiment
python examples.py 3  # Ablation study
python examples.py 7  # Benchmark suite
```

## 📋 Checklist: What to Read

**If you want to...**

- **Just run it**: QUICKSTART.md (5 min)
- **Understand basics**: README.md (15 min)
- **Do ablation studies**: QUICKSTART.md + examples.py (20 min)
- **Add a new model**: GUIDE.md section "Adding New Models" (15 min)
- **Add a new PDE**: GUIDE.md section "Adding New PDEs" (10 min)
- **Understand architecture**: ARCHITECTURE.md (15 min)
- **Advanced customization**: GUIDE.md + examples.py (45+ min)
- **Performance optimization**: GUIDE.md section "Performance Optimization" (20 min)

## 💡 Key Features at a Glance

| Feature | File | Example |
|---------|------|---------|
| Config management | config.py | `get_config("baseline")` |
| Model creation | models.py | `create_model(config, "qpinn")` |
| Loss computation | losses.py | `PhysicsLoss.total_loss(...)` |
| Training loop | main.py | `trainer.train()` |
| Evaluation | main.py | `evaluator.evaluate()` |
| Plotting | plotter.py | `plotter.plot_training_history(...)` |
| Benchmarking | benchmark.py | `suite.run_experiments(...)` |
| Experiments | config.py | 7+ pre-built ablations |

## 📈 Experiment Presets

Available in `config.py`:
- `baseline`: Original configuration
- `low_data`: 32 points instead of 64
- `high_lambda_ic`: IC weight = 50
- `quantum_large`: Larger quantum layer
- `longer_training`: 1000 epochs with decay
- `low_consistency`: Consistency weight = 0.01
- `lbfgs`: LBFGS optimizer

**Total Lines of Code**: ~1,300 lines (excluding documentation)

## 🎓 Recommended Reading Order

1. **QUICKSTART.md** (if you're in a hurry)
2. **README.md** (general understanding)
3. **examples.py** (see it in action)
4. **ARCHITECTURE.md** (understand the design)
5. **GUIDE.md** (extend as needed)

## 📝 Notes

- All code is well-commented
- Modular design allows easy extension
- No circular dependencies
- Configuration-driven (change behavior via config, not code)
- PyTorch native (no heavy dependencies)
- GPU/CPU agnostic

---

**Total Documentation**: ~80 KB across 4 markdown files  
**Total Code**: ~50 KB across 7 Python files  
**Examples**: 8 complete, runnable examples  
**Pre-built Experiments**: 7 pre-configured ablations  

Happy experimenting! 🚀
