"""
Main training and evaluation pipeline.
Supports multiple experiments with easy ablation studies.

Usage:
    python main.py --experiment baseline
    python main.py --experiment low_data
    python main.py --list-experiments
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import time

from config import ExperimentConfig, get_config, list_experiments
from models import create_model, MerlinHeatQPINN
from losses import DataSampler, PhysicsLoss
from plotter import Plotter


class Trainer:
    """Handles training loop and checkpointing."""
    
    def __init__(self, model: nn.Module, config: ExperimentConfig,
                 device: torch.device, dtype: torch.dtype,
                 output_dir: Path = None):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype
        self.output_dir = Path(output_dir) if output_dir else Path("./results") / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Freeze layers based on config
        layers_to_freeze = []
        if config.training.freeze_quantum:
            layers_to_freeze.append('quantum')
        if config.training.freeze_feature_map:
            layers_to_freeze.append('feature_map')
        if config.training.freeze_readout:
            layers_to_freeze.append('readout')
        
        if layers_to_freeze:
            print("\n" + "="*70)
            print(f"FREEZING LAYERS: {', '.join(layers_to_freeze)}")
            print("="*70)
            for name, param in model.named_parameters():
                for layer in layers_to_freeze:
                    if layer in name:
                        param.requires_grad = False
                        print(f"  Frozen: {name}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if layers_to_freeze:
            frozen_params = total_params - trainable_params
            print(f"Frozen parameters: {frozen_params:,}")
        print("="*70 + "\n")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler if specified
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function based on PDE type
        pde_name = config.pde.__class__.__name__
        if "Burgers" in pde_name:
            from losses import BurgersPhysicsLoss
            self.physics_loss = BurgersPhysicsLoss(config.pde)
        elif "Wave" in pde_name:
            from losses import WavePhysicsLoss
            self.physics_loss = WavePhysicsLoss(config.pde)
        else:
            # Default to standard heat equation loss
            self.physics_loss = PhysicsLoss(config.pde)
        self.sampler = DataSampler(config.data, config.pde, device, dtype)
        
        # Training history
        self.history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        opt_type = self.config.training.optimizer_type.lower()
        
        if opt_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate
            )
        elif opt_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9
            )
        elif opt_type == "lbfgs":
            return torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                max_iter=20,
                line_search_fn='strong_wolfe'
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler if specified."""
        if self.config.training.lr_decay is None:
            return None
        
        decay_type = self.config.training.lr_decay.lower()
        
        if decay_type == "exponential":
            gamma = self.config.training.lr_decay_factor ** (
                1 / self.config.training.lr_decay_steps
            )
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        
        elif decay_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_factor
            )
        
        elif decay_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        
        else:
            raise ValueError(f"Unknown scheduler: {decay_type}")
    
    def train(self) -> np.ndarray:
        """
        Main training loop.
        
        Returns:
            history: array of shape (epochs, 5) with loss components
        """
        print(f"\n{'='*70}")
        print(f"Training experiment: {self.config.name}")
        print(f"{'='*70}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Optimizer: {self.config.training.optimizer_type}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Sample data
            xt_f = self.sampler.sample_interior()
            xt_i = self.sampler.sample_initial()
            xt_b = self.sampler.sample_boundary()
            
            # Compute loss
            loss, loss_dict = self.physics_loss.total_loss(
                self.model, xt_f, xt_i, xt_b,
                self.config.pde.exact_solution,
                lambda_f=self.config.training.lambda_pde,
                lambda_c=self.config.training.lambda_consistency,
                lambda_i=self.config.training.lambda_initial,
                lambda_b=self.config.training.lambda_boundary,
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            self.history.append([
                loss_dict["total"],
                loss_dict["pde"],
                loss_dict["consistency"],
                loss_dict["initial"],
                loss_dict["boundary"],
            ])
            
            # Logging
            if epoch == 1 or epoch % self.config.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:4d}/{self.config.training.epochs} | "
                    f"Loss: {loss_dict['total']:.4e} | "
                    f"PDE: {loss_dict['pde']:.2e} | "
                    f"Cons: {loss_dict['consistency']:.2e} | "
                    f"IC: {loss_dict['initial']:.2e} | "
                    f"BC: {loss_dict['boundary']:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds")
        
        return np.array(self.history)
    
    def get_history(self) -> np.ndarray:
        """Return training history."""
        return np.array(self.history)
    
    def save_checkpoint(self, name: str = "best_model"):
        """Save model checkpoint."""
        path = self.output_dir / f"{name}.pt"
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)
        print(f"Saved checkpoint: {path}")
        return path
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint: {path}")


class Evaluator:
    """Handles model evaluation on test grid."""
    
    def __init__(self, model: nn.Module, config: ExperimentConfig,
                 device: torch.device, dtype: torch.dtype):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype
    
    def evaluate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                               torch.Tensor, Dict]:
        """
        Evaluate model on regular grid.
        
        Returns:
            U_pred: predicted solution
            U_true: exact solution
            x_grid: x coordinates
            t_grid: t coordinates
            metrics: dictionary of error metrics
        """
        print("\nEvaluating on test grid...")
        
        # Create evaluation grid
        nx, nt = self.config.evaluation.nx, self.config.evaluation.nt
        
        x = torch.linspace(
            self.config.pde.x_min, self.config.pde.x_max, nx,
            device=self.device, dtype=self.dtype
        ).reshape(-1, 1)
        
        t = torch.linspace(
            self.config.pde.t_min, self.config.pde.t_max, nt,
            device=self.device, dtype=self.dtype
        ).reshape(-1, 1)
        
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            U_pred, _ = self.model(xt_grid)
            U_pred = U_pred.reshape(nx, nt).cpu()
            
            # Exact solution
            U_true = self.config.pde.exact_solution(
                xt_grid[:, 0:1], xt_grid[:, 1:2]
            ).reshape(nx, nt).cpu()
        
        # Compute error metrics
        error = U_pred - U_true
        rel_l2 = torch.linalg.norm(error) / torch.linalg.norm(U_true)
        mae = torch.abs(error).mean()
        max_abs_error = torch.abs(error).max()
        rmse = torch.sqrt((error ** 2).mean())
        
        metrics = {
            "rel_l2": float(rel_l2),
            "mae": float(mae),
            "max_abs_error": float(max_abs_error),
            "rmse": float(rmse),
        }
        
        print(f"Relative L2 error: {metrics['rel_l2']:.4e}")
        print(f"MAE: {metrics['mae']:.4e}")
        print(f"Max abs error: {metrics['max_abs_error']:.4e}")
        print(f"RMSE: {metrics['rmse']:.4e}")
        
        return U_pred, U_true, x, t, metrics


def run_experiment(experiment_name: str, model_type: str = "qpinn",
                  output_dir: Optional[str] = None, 
                  freeze_quantum: bool = False) -> Dict:
    """
    Run a complete experiment: train, evaluate, and return results.
    
    Args:
        experiment_name: name of experiment configuration
        model_type: "qpinn", "classical", or "deep_classical"
        output_dir: directory for saving results
    
    Returns:
        results: dictionary with training, evaluation, and metrics info
    """
    # Get configuration
    config = get_config(experiment_name)
    
    # Apply CLI flags to config
    if freeze_quantum:
        config.training.freeze_quantum = True
    
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda"
                         else "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    
    # Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create model
    model = create_model(config.model, model_type=model_type).to(device=device, dtype=dtype)
    
    # Ensure parameters are the right dtype
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    
    print(f"Model: {model}")
    
    # Train
    if output_dir is None:
        output_dir = f"./results/{config.name}_{model_type}"
    
    trainer = Trainer(model, config, device, dtype, output_dir=output_dir)
    history = trainer.train()
    trainer.save_checkpoint()
    
    # Evaluate
    evaluator = Evaluator(model, config, device, dtype)
    U_pred, U_true, x_grid, t_grid, metrics = evaluator.evaluate()
    
    # Plot
    plotter = Plotter(save_dir=output_dir)
    plotter.plot_training_history(history, title=f"Training History - {config.name}",
                                 save_name="training_history.png")
    rel_l2 = plotter.plot_solution_comparison(U_true, U_pred, x_grid, t_grid,
                                             title_prefix=f"{config.name} Solution",
                                             save_name="solution_comparison.png")
    plotter.plot_solution_slices(U_true, U_pred, x_grid, t_grid,
                                save_name="solution_slices.png")
    
    # Save results
    results = {
        "experiment": config.name,
        "model_type": model_type,
        "config": config.__dict__,
        "metrics": metrics,
        "history": history.tolist(),
    }
    
    results_path = Path(output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="QPINN Training and Evaluation")
    parser.add_argument("--experiment", type=str, default="baseline",
                       help="Experiment configuration to run")
    parser.add_argument("--model-type", type=str, default="qpinn",
                       choices=["qpinn", "qpinn_frozen", "classical", "deep_classical"],
                       help="Model type to use")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--freeze-quantum", action="store_true",
                       help="Freeze quantum layer, only train classical layers")
    parser.add_argument("--list-experiments", action="store_true",
                       help="List available experiments and exit")
    
    args = parser.parse_args()
    
    if args.list_experiments:
        print("\nAvailable experiments:")
        for exp_name in list_experiments():
            print(f"  - {exp_name}")
        return
    
    # Run experiment
    results = run_experiment(
        args.experiment,
        model_type=args.model_type,
        output_dir=args.output_dir,
        freeze_quantum=args.freeze_quantum
    )


if __name__ == "__main__":
    main()