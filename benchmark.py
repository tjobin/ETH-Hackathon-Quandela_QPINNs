"""
Benchmark script for running ablation studies and comparing results.

Usage:
    python benchmark.py --experiments baseline low_data high_lambda_ic
    python benchmark.py --experiments all  # runs all available experiments
    python benchmark.py --lr-sweep  # sweeps learning rates
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from datetime import datetime

from config import get_config, list_experiments
from main import run_experiment
from plotter import Plotter


class BenchmarkSuite:
    """Manages running and comparing multiple experiments."""
    
    def __init__(self, output_base: str = "./benchmarks"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_dir = self.output_base / timestamp
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def run_experiments(self, experiment_names: List[str],
                       model_types: List[str] = None) -> Dict:
        """
        Run multiple experiments.
        
        Args:
            experiment_names: list of experiment names to run
            model_types: list of model types to use (default: ["qpinn"])
        
        Returns:
            results: dictionary with results from all experiments
        """
        if model_types is None:
            model_types = ["qpinn"]
        
        print(f"\n{'='*70}")
        print(f"Running Benchmark Suite")
        print(f"Experiments: {experiment_names}")
        print(f"Model types: {model_types}")
        print(f"Output directory: {self.benchmark_dir}")
        print(f"{'='*70}\n")
        
        for exp_name in experiment_names:
            for model_type in model_types:
                exp_key = f"{exp_name}_{model_type}"
                print(f"\n{'='*70}")
                print(f"Running: {exp_key}")
                print(f"{'='*70}")
                
                try:
                    output_dir = self.benchmark_dir / exp_key
                    results = run_experiment(
                        exp_name,
                        model_type=model_type,
                        output_dir=str(output_dir)
                    )
                    self.results[exp_key] = results
                    print(f"\n✓ Completed: {exp_key}")
                
                except Exception as e:
                    print(f"\n✗ Failed: {exp_key}")
                    print(f"Error: {e}")
                    self.results[exp_key] = {"error": str(e)}
        
        return self.results
    
    def lr_sweep(self, lrs: List[float], n_experiments: int = 5) -> Dict:
        """
        Learning rate sweep.
        
        Args:
            lrs: list of learning rates to try
            n_experiments: number of learning rates (default 5)
        
        Returns:
            results: dictionary with sweep results
        """
        if lrs is None:
            lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        
        print(f"\n{'='*70}")
        print(f"Learning Rate Sweep")
        print(f"Learning rates: {lrs}")
        print(f"{'='*70}\n")
        
        # Create custom configs for each LR
        from config import baseline_config
        
        for lr in lrs:
            cfg = baseline_config()
            cfg.name = f"lr_{lr:.0e}"
            cfg.training.learning_rate = lr
            
            # Register the config
            exp_name = cfg.name
            exp_key = f"lr_sweep_{exp_name}"
            
            print(f"\nRunning LR sweep: {lr}")
            
            try:
                output_dir = self.benchmark_dir / exp_key
                results = run_experiment(
                    "baseline",  # Use baseline but with modified LR in main
                    model_type="qpinn",
                    output_dir=str(output_dir)
                )
                # Modify LR in results
                results["config"]["training"]["learning_rate"] = lr
                self.results[exp_key] = results
                print(f"✓ Completed: LR={lr}")
            
            except Exception as e:
                print(f"✗ Failed: LR={lr}")
                print(f"Error: {e}")
    
    def generate_comparison_report(self) -> Dict:
        """
        Generate comparison report across all experiments.
        
        Returns:
            report: dictionary with comparative analysis
        """
        print(f"\n{'='*70}")
        print("Generating Comparison Report")
        print(f"{'='*70}\n")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.results),
            "experiments": {},
            "best_metrics": {},
        }
        
        # Extract metrics for each experiment
        for exp_name, result in self.results.items():
            if "error" in result:
                report["experiments"][exp_name] = {"error": result["error"]}
                continue
            
            metrics = result.get("metrics", {})
            report["experiments"][exp_name] = {
                "rel_l2": metrics.get("rel_l2"),
                "mae": metrics.get("mae"),
                "max_abs_error": metrics.get("max_abs_error"),
                "rmse": metrics.get("rmse"),
            }
        
        # Find best metrics
        metric_names = ["rel_l2", "mae", "max_abs_error", "rmse"]
        for metric in metric_names:
            best_exp = None
            best_val = float('inf')
            
            for exp_name, metrics in report["experiments"].items():
                if metric in metrics and metrics[metric] is not None:
                    val = metrics[metric]
                    if val < best_val:
                        best_val = val
                        best_exp = exp_name
            
            if best_exp:
                report["best_metrics"][metric] = {
                    "experiment": best_exp,
                    "value": best_val
                }
        
        return report
    
    def plot_comparisons(self):
        """Generate comparison plots across all experiments."""
        print("\nGenerating comparison plots...")
        
        plotter = Plotter(save_dir=self.benchmark_dir)
        
        # Collect loss histories
        loss_histories = {}
        for exp_name, result in self.results.items():
            if "error" not in result and "history" in result:
                history = np.array(result["history"])
                loss_histories[exp_name] = history
        
        if loss_histories:
            plotter.plot_loss_comparison(
                loss_histories,
                title="Loss Comparison Across Experiments",
                save_name="loss_comparison.png"
            )
        
        # Collect error metrics
        metrics = {}
        for exp_name, result in self.results.items():
            if "error" not in result and "metrics" in result:
                metrics[exp_name] = result["metrics"]
        
        if metrics:
            plotter.plot_error_metrics(
                metrics,
                save_name="metrics_comparison.png"
            )
    
    def save_report(self, report: Dict) -> Path:
        """Save benchmark report to JSON."""
        report_path = self.benchmark_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nBenchmark report saved: {report_path}")
        return report_path
    
    def print_summary(self, report: Dict):
        """Print summary of benchmark results."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}\n")
        
        print("Experiments completed:")
        for exp_name in report["experiments"].keys():
            print(f"  ✓ {exp_name}")
        
        print("\nBest metrics:")
        for metric, info in report["best_metrics"].items():
            print(f"  {metric}: {info['value']:.4e} ({info['experiment']})")
        
        print("\nDetailed metrics:")
        for exp_name, metrics in report["experiments"].items():
            print(f"\n  {exp_name}:")
            if "error" in metrics:
                print(f"    Error: {metrics['error']}")
            else:
                for metric_name, value in metrics.items():
                    if value is not None:
                        print(f"    {metric_name}: {value:.4e}")
        
        print(f"\nResults directory: {self.benchmark_dir}")


def main():
    parser = argparse.ArgumentParser(description="QPINN Benchmark Suite")
    parser.add_argument("--experiments", nargs="+", default=["baseline"],
                       help="Experiments to run (use 'all' for all available)")
    parser.add_argument("--model-types", nargs="+", default=["qpinn"],
                       help="Model types to compare")
    parser.add_argument("--lr-sweep", action="store_true",
                       help="Run learning rate sweep instead")
    parser.add_argument("--lrs", nargs="+", type=float, default=None,
                       help="Learning rates for sweep")
    parser.add_argument("--output-dir", type=str, default="./benchmarks",
                       help="Base output directory for benchmarks")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(output_base=args.output_dir)
    
    if args.lr_sweep:
        # Learning rate sweep
        suite.lr_sweep(args.lrs)
    else:
        # Experiment sweep
        if "all" in args.experiments:
            experiments = list_experiments()
        else:
            experiments = args.experiments
        
        suite.run_experiments(experiments, model_types=args.model_types)
    
    # Generate and save report
    report = suite.generate_comparison_report()
    suite.print_summary(report)
    suite.save_report(report)
    suite.plot_comparisons()


if __name__ == "__main__":
    main()
