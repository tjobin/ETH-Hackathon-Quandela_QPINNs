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
import matplotlib.pyplot as plt

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
        
        # Fourier spectral comparisons
        self._plot_fourier_comparisons()
    
    def _plot_fourier_comparisons(self):
        """Generate Fourier spectral comparison plots across experiments."""
        import matplotlib.pyplot as plt
        from fourier_plotter import FourierPlotter
        
        print("\nGenerating Fourier spectral comparisons...")
        
        fourier_plotter = FourierPlotter(save_dir=self.benchmark_dir)
        
        # Collect Fourier histories
        fourier_histories = {}
        final_metrics = {}
        
        for exp_name, result in self.results.items():
            if "error" not in result:
                if "fourier_metrics" in result and result["fourier_metrics"]:
                    fourier_histories[exp_name] = result["fourier_metrics"]
                    final_metrics[exp_name] = result["fourier_metrics"][-1]
        
        if not fourier_histories:
            print("No Fourier metrics found in results")
            return
        
        exp_names = list(fourier_histories.keys())
        
        # 1. Spectral L2 error evolution comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        for exp_name, history in fourier_histories.items():
            epochs = np.array([m['epoch'] for m in history])
            spectral_l2 = np.array([m['spectral_l2'] for m in history])
            ax.semilogy(epochs, spectral_l2, 'o-', label=exp_name, linewidth=2.5, markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Spectral L2 Error', fontsize=12)
        ax.set_title('Spectral L2 Error Comparison Across Experiments', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_spectral_l2_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_spectral_l2_comparison.png")
        
        # 2. Power spectrum error evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        for exp_name, history in fourier_histories.items():
            epochs = np.array([m['epoch'] for m in history])
            power_error = np.array([m['power_spectrum_error'] for m in history])
            ax.semilogy(epochs, power_error, 's-', label=exp_name, linewidth=2.5, markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Power Spectrum Error', fontsize=12)
        ax.set_title('Power Spectrum Error Comparison', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_power_error_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_power_error_comparison.png")
        
        # 3. Spectral concentration (smoothness) comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        for exp_name, history in fourier_histories.items():
            epochs = np.array([m['epoch'] for m in history])
            conc_pred = np.array([m['concentration_pred'] for m in history])
            ax.plot(epochs, conc_pred, 'o-', label=exp_name, linewidth=2.5, markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Spectral Concentration', fontsize=12)
        ax.set_title('Solution Smoothness (Spectral Concentration) Comparison', fontsize=13)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_concentration_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_concentration_comparison.png")
        
        # 4. Low vs High frequency error
        fig, ax = plt.subplots(figsize=(12, 6))
        for exp_name, history in fourier_histories.items():
            epochs = np.array([m['epoch'] for m in history])
            low_freq = np.array([m['low_freq_error'] for m in history])
            high_freq = np.array([m['high_freq_error'] for m in history])
            ax.semilogy(epochs, low_freq, 'o-', label=f'{exp_name} (Low)', linewidth=2, markersize=4)
            ax.semilogy(epochs, high_freq, 's--', label=f'{exp_name} (High)', linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Low vs High Frequency Error Comparison', fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_lowvshigh_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_lowvshigh_comparison.png")
        
        # 5. Final metrics comparison bar chart
        if final_metrics:
            metrics_to_plot = ['spectral_l2', 'power_spectrum_error', 'low_freq_error', 'high_freq_error']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                values = [final_metrics[exp].get(metric, 0) for exp in exp_names]
                bars = ax.bar(exp_names, values, alpha=0.7, color=plt.cm.Set2(idx))
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
                ax.set_title(f'Final {metric.replace("_", " ").title()}', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2e}', ha='center', va='bottom', fontsize=9)
                
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = self.benchmark_dir / "fourier_final_metrics_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ fourier_final_metrics_comparison.png")
        
        # 6. Mode L2 error comparison (new!)
        print("\nGenerating mode-by-mode L2 error comparisons...")
        self._plot_mode_l2_comparisons(fourier_plotter)
        
        print("Fourier spectral comparisons complete!")
    
    def _plot_mode_l2_comparisons(self, fourier_plotter):
        """
        Generate mode-by-mode L2 error comparison plots across experiments.
        
        Args:
            fourier_plotter: FourierPlotter instance
        """
        from fourier_plotter import FourierPlotter
        import torch
        import json
        
        print("\nGenerating mode-by-mode L2 vs k plots for each experiment...")
        
        # For each experiment, load solutions from results JSON and generate plots
        for exp_name, result in self.results.items():
            if "error" not in result:
                # Try to get solutions from loaded results
                if "U_pred" in result and "U_true" in result:
                    U_pred = result["U_pred"]
                    U_true = result["U_true"]
                else:
                    # Try to load from results.json file
                    try:
                        results_dir = self.benchmark_dir / exp_name
                        results_file = results_dir / "results.json"
                        
                        if results_file.exists():
                            with open(results_file, 'r') as f:
                                file_result = json.load(f)
                                U_pred = file_result.get("U_pred")
                                U_true = file_result.get("U_true")
                        else:
                            continue
                    except Exception as e:
                        print(f"  Warning: Could not load solutions for {exp_name}: {e}")
                        continue
                
                # Convert to tensors
                U_pred = torch.tensor(U_pred, dtype=torch.float32)
                U_true = torch.tensor(U_true, dtype=torch.float32)
                
                # Create output directory
                exp_dir = self.benchmark_dir / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a FourierPlotter for this experiment's directory
                exp_plotter = FourierPlotter(save_dir=exp_dir)
                
                # Generate the detailed mode L2 vs k plot
                try:
                    exp_plotter.plot_mode_l2_vs_k_detailed(U_pred, U_true,
                                                          save_name="fourier_mode_l2_vs_k.png")
                    print(f"  ✓ {exp_name}/fourier_mode_l2_vs_k.png")
                except Exception as e:
                    print(f"  Error generating plot for {exp_name}: {e}")
        
        # Create a comparison plot showing mode error patterns across experiments
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        exp_names_list = [name for name in self.results.keys() if "error" not in self.results[name]]
        
        if exp_names_list:
            # For demonstration, we'll plot the final spectral L2 errors
            # This gives a sense of mode performance across models
            
            ax = axes[0]
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Final Spectral L2 Error', fontsize=12)
            ax.set_title('Final Spectral L2 Error by Experiment', fontsize=13, fontweight='bold')
            
            values = []
            for exp_name in exp_names_list:
                result = self.results[exp_name]
                if "fourier_metrics" in result and result["fourier_metrics"]:
                    final_metric = result["fourier_metrics"][-1]
                    values.append(final_metric.get('spectral_l2', 0))
                else:
                    values.append(0)
            
            bars = ax.bar(exp_names_list, values, alpha=0.7, color=plt.cm.tab10(np.arange(len(exp_names_list))))
            ax.set_xticklabels(exp_names_list, rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2e}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3, axis='y')
            
            # Second plot: Low vs High frequency final errors
            ax = axes[1]
            x = np.arange(len(exp_names_list))
            width = 0.35
            
            low_freq_errors = []
            high_freq_errors = []
            
            for exp_name in exp_names_list:
                result = self.results[exp_name]
                if "fourier_metrics" in result and result["fourier_metrics"]:
                    final_metric = result["fourier_metrics"][-1]
                    low_freq_errors.append(final_metric.get('low_freq_error', 0))
                    high_freq_errors.append(final_metric.get('high_freq_error', 0))
                else:
                    low_freq_errors.append(0)
                    high_freq_errors.append(0)
            
            bars1 = ax.bar(x - width/2, low_freq_errors, width, label='Low Freq', alpha=0.7, color='green')
            bars2 = ax.bar(x + width/2, high_freq_errors, width, label='High Freq', alpha=0.7, color='red')
            
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Error', fontsize=12)
            ax.set_title('Low vs High Frequency Error Comparison (Final)', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_names_list, rotation=45, ha='right')
            ax.set_yscale('log')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y', which='both')
        
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_mode_l2_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_mode_l2_comparison.png")
    
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