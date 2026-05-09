"""
Example usage patterns for the QPINN analysis pipeline.
This script demonstrates different ways to use the framework.
"""

# ============================================================================
# EXAMPLE 1: Running a single experiment programmatically
# ============================================================================

def example_1_single_experiment():
    """Run a single experiment with baseline configuration."""
    
    from main import run_experiment
    
    results = run_experiment(
        experiment_name="baseline",
        model_type="qpinn",
        output_dir="./results/example_1"
    )
    
    print("Experiment Results:")
    print(f"  Relative L2 Error: {results['metrics']['rel_l2']:.4e}")
    print(f"  MAE: {results['metrics']['mae']:.4e}")
    print(f"  RMSE: {results['metrics']['rmse']:.4e}")


# ============================================================================
# EXAMPLE 2: Custom configuration and training
# ============================================================================

def example_2_custom_config():
    """Create and use a custom configuration."""
    
    from config import baseline_config
    from main import Trainer, Evaluator
    from models import create_model
    import torch
    import numpy as np
    
    # Create custom config
    config = baseline_config()
    config.name = "my_custom_experiment"
    config.training.epochs = 500
    config.training.learning_rate = 5e-3
    config.data.n_interior = 128
    config.model.hidden_feature = 32
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create model
    model = create_model(config.model, model_type="qpinn").to(device, dtype)
    
    # Train
    trainer = Trainer(model, config, device, dtype, output_dir="./results/example_2")
    history = trainer.train()
    trainer.save_checkpoint()
    
    # Evaluate
    evaluator = Evaluator(model, config, device, dtype)
    U_pred, U_true, x_grid, t_grid, metrics = evaluator.evaluate()
    
    # Visualize
    from plotter import Plotter
    plotter = Plotter(save_dir="./results/example_2")
    plotter.plot_training_history(history, title="Custom Config Training")
    plotter.plot_solution_comparison(U_true, U_pred, x_grid, t_grid)


# ============================================================================
# EXAMPLE 3: Ablation study with multiple configurations
# ============================================================================

def example_3_ablation_study():
    """Run multiple configurations for ablation study."""
    
    from main import run_experiment
    import json
    from pathlib import Path
    
    experiments = ["baseline", "low_data", "high_lambda_ic", "longer_training"]
    results = {}
    
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"{'='*60}")
        
        result = run_experiment(
            experiment_name=exp_name,
            model_type="qpinn",
            output_dir=f"./results/ablation_{exp_name}"
        )
        
        results[exp_name] = {
            "rel_l2": result["metrics"]["rel_l2"],
            "mae": result["metrics"]["mae"],
            "final_loss": result["history"][-1][0],
        }
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Ablation Study Results")
    print(f"{'='*60}")
    print(f"{'Experiment':<20} {'RelL2':<12} {'MAE':<12} {'Final Loss':<12}")
    print("-" * 60)
    
    for exp_name, metrics in results.items():
        print(f"{exp_name:<20} {metrics['rel_l2']:<12.4e} {metrics['mae']:<12.4e} {metrics['final_loss']:<12.4e}")
    
    # Save results
    results_path = Path("./results/ablation_study.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


# ============================================================================
# EXAMPLE 4: Model comparison (QPINN vs Classical)
# ============================================================================

def example_4_model_comparison():
    """Compare different model types."""
    
    from main import run_experiment
    from plotter import Plotter
    import numpy as np
    from pathlib import Path
    
    model_types = ["qpinn", "classical", "deep_classical"]
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        result = run_experiment(
            experiment_name="baseline",
            model_type=model_type,
            output_dir=f"./results/model_comparison_{model_type}"
        )
        results[model_type] = result
    
    # Compare loss histories
    loss_histories = {
        model_type: np.array(result["history"])
        for model_type, result in results.items()
    }
    
    plotter = Plotter(save_dir="./results/model_comparison")
    plotter.plot_loss_comparison(
        loss_histories,
        title="Model Type Comparison - Loss Convergence",
        save_name="loss_comparison.png"
    )
    
    # Compare metrics
    metrics = {
        model_type: result["metrics"]
        for model_type, result in results.items()
    }
    
    plotter.plot_error_metrics(metrics, save_name="metrics_comparison.png")


# ============================================================================
# EXAMPLE 5: Learning rate sweep
# ============================================================================

def example_5_learning_rate_sweep():
    """Perform a learning rate sweep."""
    
    from config import baseline_config
    from main import run_experiment
    from pathlib import Path
    import json
    
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    sweep_results = {}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        # We'll use the main script with modified command
        # For now, we manually create a custom config
        from config import get_config
        import torch
        import numpy as np
        from models import create_model
        from main import Trainer, Evaluator
        
        config = baseline_config()
        config.name = f"lr_{lr:.0e}"
        config.training.learning_rate = lr
        config.training.epochs = 200  # Shorter for sweep
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        model = create_model(config.model, model_type="qpinn").to(device, dtype)
        trainer = Trainer(model, config, device, dtype, 
                         output_dir=f"./results/lr_sweep_{lr:.0e}")
        history = trainer.train()
        
        evaluator = Evaluator(model, config, device, dtype)
        U_pred, U_true, x_grid, t_grid, metrics = evaluator.evaluate()
        
        sweep_results[f"lr_{lr:.0e}"] = {
            "learning_rate": lr,
            "rel_l2": metrics["rel_l2"],
            "final_loss": history[-1, 0],
            "best_loss": history[:, 0].min(),
        }
    
    # Print and save results
    print(f"\n{'='*70}")
    print("Learning Rate Sweep Results")
    print(f"{'='*70}")
    print(f"{'LR':<12} {'RelL2':<12} {'Final Loss':<12} {'Best Loss':<12}")
    print("-" * 70)
    
    for exp_name, metrics in sweep_results.items():
        print(f"{metrics['learning_rate']:<12.0e} {metrics['rel_l2']:<12.4e} {metrics['final_loss']:<12.4e} {metrics['best_loss']:<12.4e}")
    
    # Save results
    results_path = Path("./results/lr_sweep_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")


# ============================================================================
# EXAMPLE 6: Custom PDE configuration
# ============================================================================

def example_6_custom_pde():
    """Define and train on a custom PDE configuration."""
    
    from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, PDEConfig, EvaluationConfig
    from main import Trainer, Evaluator
    from models import create_model
    import torch
    import numpy as np
    import math
    
    # Create a configuration with modified PDE parameters
    config = ExperimentConfig(
        name="custom_pde",
        pde=PDEConfig(
            alpha=0.2,  # Different diffusion coefficient
            x_min=0.0, x_max=1.0,
            t_min=0.0, t_max=0.5,  # Shorter time domain
        ),
        data=DataConfig(
            n_interior=96,
            n_initial=96,
            n_boundary=96,
        ),
        model=ModelConfig(
            feature_size=6,
            quantum_output_size=6,
        ),
        training=TrainingConfig(
            epochs=400,
            learning_rate=1e-2,
            lambda_pde=1.0,
            lambda_consistency=0.15,
            lambda_initial=15.0,
            lambda_boundary=1.0,
        ),
    )
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    model = create_model(config.model, model_type="qpinn").to(device, dtype)
    trainer = Trainer(model, config, device, dtype, output_dir="./results/custom_pde")
    history = trainer.train()
    
    # Evaluate
    evaluator = Evaluator(model, config, device, dtype)
    U_pred, U_true, x_grid, t_grid, metrics = evaluator.evaluate()
    
    # Visualize
    from plotter import Plotter
    plotter = Plotter(save_dir="./results/custom_pde")
    plotter.plot_training_history(history, title="Custom PDE Configuration")
    plotter.plot_solution_comparison(U_true, U_pred, x_grid, t_grid)


# ============================================================================
# EXAMPLE 7: Batch comparison using benchmark suite
# ============================================================================

def example_7_benchmark_suite():
    """Use the benchmark suite for comprehensive comparison."""
    
    from benchmark import BenchmarkSuite
    
    # Create benchmark suite
    suite = BenchmarkSuite(output_base="./benchmarks/example7")
    
    # Run multiple experiments
    experiments = ["baseline", "low_data", "high_lambda_ic", "quantum_large"]
    model_types = ["qpinn", "classical"]
    
    suite.run_experiments(experiments, model_types=model_types)
    
    # Generate report
    report = suite.generate_comparison_report()
    suite.print_summary(report)
    suite.save_report(report)
    suite.plot_comparisons()


# ============================================================================
# EXAMPLE 8: Advanced - Custom loss and training loop
# ============================================================================

def example_8_advanced_custom_training():
    """Demonstrate advanced custom training with modified loss."""
    
    from config import baseline_config
    from models import create_model
    from losses import DataSampler, PhysicsLoss
    import torch
    import numpy as np
    
    config = baseline_config()
    config.name = "advanced_custom"
    config.training.epochs = 300
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create model, sampler, and loss
    model = create_model(config.model, model_type="qpinn").to(device, dtype)
    sampler = DataSampler(config.data, config.pde, device, dtype)
    physics_loss = PhysicsLoss(config.pde)
    
    # Custom optimizer with gradual weight scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    history = []
    
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Dynamic loss weighting
        progress = epoch / config.training.epochs
        lambda_i = config.training.lambda_initial * (1.0 + progress)  # Increase IC weight
        lambda_f = config.training.lambda_pde * (1.0 - 0.5 * progress)  # Decrease PDE weight
        
        # Sample and compute loss
        xt_f = sampler.sample_interior()
        xt_i = sampler.sample_initial()
        xt_b = sampler.sample_boundary()
        
        loss, loss_dict = physics_loss.total_loss(
            model, xt_f, xt_i, xt_b,
            config.pde.exact_solution,
            lambda_f=lambda_f,
            lambda_c=config.training.lambda_consistency,
            lambda_i=lambda_i,
            lambda_b=config.training.lambda_boundary,
        )
        
        loss.backward()
        optimizer.step()
        
        history.append([loss_dict["total"], loss_dict["pde"], loss_dict["consistency"],
                       loss_dict["initial"], loss_dict["boundary"]])
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}: Loss={loss_dict['total']:.4e}, "
                  f"lambda_i={lambda_i:.2f}, lambda_f={lambda_f:.2f}")
    
    # Plot results
    from plotter import Plotter
    plotter = Plotter(save_dir="./results/advanced_custom")
    plotter.plot_training_history(np.array(history), title="Advanced Custom Training")
    
    print(f"Advanced custom training completed! Results saved to ./results/advanced_custom")


# ============================================================================
# Main: Run all examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Single Experiment", example_1_single_experiment),
        "2": ("Custom Configuration", example_2_custom_config),
        "3": ("Ablation Study", example_3_ablation_study),
        "4": ("Model Comparison", example_4_model_comparison),
        "5": ("Learning Rate Sweep", example_5_learning_rate_sweep),
        "6": ("Custom PDE", example_6_custom_pde),
        "7": ("Benchmark Suite", example_7_benchmark_suite),
        "8": ("Advanced Custom Training", example_8_advanced_custom_training),
    }
    
    print("\n" + "="*70)
    print("QPINN Pipeline Examples")
    print("="*70)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning Example {example_num}: {name}")
            print("="*70)
            func()
        else:
            print(f"Invalid example number: {example_num}")
    else:
        print("\nUsage: python examples.py <example_number>")
        print("Example: python examples.py 1")
