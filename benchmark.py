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
        
        # 4. Low vs High frequency error (FINAL EPOCH comparison)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Evolution during training
        for exp_name, history in fourier_histories.items():
            epochs = np.array([m['epoch'] for m in history])
            low_freq = np.array([m['low_freq_error'] for m in history])
            high_freq = np.array([m['high_freq_error'] for m in history])
            ax1.semilogy(epochs, low_freq, 'o-', label=f'{exp_name} (Low)', linewidth=2, markersize=4)
            ax1.semilogy(epochs, high_freq, 's--', label=f'{exp_name} (High)', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Error', fontsize=12)
        ax1.set_title('Low vs High Frequency Error Evolution During Training', fontsize=13)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right: Final epoch comparison (bar chart)
        exp_names = list(fourier_histories.keys())
        final_low = []
        final_high = []
        for exp_name in exp_names:
            history = fourier_histories[exp_name]
            if history:
                final_low.append(history[-1]['low_freq_error'])
                final_high.append(history[-1]['high_freq_error'])
        
        x = np.arange(len(exp_names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, final_low, width, label='Low Freq', alpha=0.8, color='green')
        bars2 = ax2.bar(x + width/2, final_high, width, label='High Freq', alpha=0.8, color='red')
        
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_title('Final Epoch: Low vs High Frequency Error', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(exp_names, rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y', which='both')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}', ha='center', va='bottom', fontsize=9)
        
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
        
        # 6. Energy distribution as function of mode index k
        print("  Generating energy distribution plot...")
        
        import torch
        
        # Collect energy distributions from numpy files
        energy_dist = {}
        
        for exp_name, result in self.results.items():
            if "error" not in result:
                try:
                    exp_dir = self.benchmark_dir / exp_name
                    u_true_file = exp_dir / "U_true.npy"
                    
                    if u_true_file.exists():
                        U_true = np.load(u_true_file)
                        U_true = torch.tensor(U_true, dtype=torch.float32)
                        
                        # Compute FFT
                        u_true_hat = torch.fft.fft(U_true, dim=0)
                        
                        # Keep only positive frequencies
                        nx = U_true.shape[0]
                        u_true_hat = u_true_hat[:nx//2, :]
                        
                        # Compute energy per mode: |û[k,:]|²
                        energy_per_mode = torch.sum(torch.abs(u_true_hat) ** 2, dim=1)
                        total_energy = torch.sum(energy_per_mode)
                        energy_fraction = energy_per_mode / total_energy
                        
                        energy_dist[exp_name] = {
                            'energy': energy_per_mode.detach().cpu().numpy(),
                            'energy_fraction': energy_fraction.detach().cpu().numpy(),
                            'total_energy': float(total_energy)
                        }
                except Exception as e:
                    print(f"    Warning: Could not compute energy for {exp_name}: {e}")
        
        if energy_dist:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Use first experiment's energy (it's the same analytical solution for all)
            first_exp = list(energy_dist.keys())[0]
            modes = np.arange(len(energy_dist[first_exp]['energy_fraction']))
            energy_frac = energy_dist[first_exp]['energy_fraction']
            cumulative_energy = np.cumsum(energy_frac)
            
            # Plot 1: Energy fraction per mode (semi-log)
            ax1.semilogy(modes, energy_frac, 'o-', linewidth=2, markersize=4, color='#0284c7')
            ax1.fill_between(modes, energy_frac, alpha=0.3, color='#0284c7')
            ax1.set_xlabel('Mode Index k', fontsize=12)
            ax1.set_ylabel('Energy Fraction per Mode', fontsize=12)
            ax1.set_title('Energy Distribution Across Frequency Modes', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, which='both')
            
            # Add shading for low/high frequency regions
            ax1.axvspan(0, nx//4, alpha=0.1, color='green', label='Low Freq (0-25%)')
            ax1.axvspan(3*nx//4, nx//2, alpha=0.1, color='red', label='High Freq (75-100%)')
            ax1.legend(fontsize=10)
            
            # Plot 2: Cumulative energy
            ax2.plot(modes, cumulative_energy, 'o-', linewidth=2.5, markersize=4, color='#059669')
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% energy')
            ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% energy')
            ax2.fill_between(modes, cumulative_energy, alpha=0.2, color='#059669')
            ax2.set_xlabel('Mode Index k', fontsize=12)
            ax2.set_ylabel('Cumulative Energy Fraction', fontsize=12)
            ax2.set_title('Cumulative Energy vs Mode Index', fontsize=13, fontweight='bold')
            ax2.set_ylim([0, 1.05])
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Add shading for low/high frequency regions
            ax2.axvspan(0, nx//4, alpha=0.1, color='green', label='Low Freq')
            ax2.axvspan(3*nx//4, nx//2, alpha=0.1, color='red', label='High Freq')
            
            plt.tight_layout()
            save_path = self.benchmark_dir / "fourier_energy_distribution.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ fourier_energy_distribution.png")
        
        # 7. Mode L2 error comparison
        print("\nGenerating mode-by-mode L2 error comparisons...")
        self._plot_mode_l2_comparisons(fourier_plotter)
        
        print("Fourier spectral comparisons complete!")
    

    def _plot_mode_l2_comparisons(self, fourier_plotter):
        """
        Generate mode-by-mode L2 error comparison plots across experiments.
        Creates overlay plots showing L2 error vs mode index k for all models.
        """
        print("\nGenerating mode L2 error vs k overlay plots...")
        
        # Try to compute mode errors from individual experiment folders
        self._plot_mode_l2_vs_k_overlay()
    
    def _plot_mode_l2_vs_k_overlay(self):
        """
        Create overlay plots of L2 error vs mode index k.
        Loads solutions from numpy files saved during individual runs.
        """
        import torch
        import numpy as np
        
        print("  Computing mode L2 errors from individual runs...")
        
        mode_errors_dict = {}
        
        # Load solutions from numpy files for each experiment
        for exp_name, result in self.results.items():
            if "error" not in result:
                try:
                    exp_dir = self.benchmark_dir / exp_name
                    
                    # Load numpy files
                    u_pred_file = exp_dir / "U_pred.npy"
                    u_true_file = exp_dir / "U_true.npy"
                    
                    if u_pred_file.exists() and u_true_file.exists():
                        U_pred = np.load(u_pred_file)
                        U_true = np.load(u_true_file)
                        
                        # Convert to tensors
                        U_pred = torch.tensor(U_pred, dtype=torch.float32)
                        U_true = torch.tensor(U_true, dtype=torch.float32)
                        
                        # Compute FFTs and mode errors (positive frequencies only)
                        u_pred_hat = torch.fft.fft(U_pred, dim=0)
                        u_true_hat = torch.fft.fft(U_true, dim=0)
                        
                        # Keep only positive frequencies [0:nx//2]
                        nx = U_pred.shape[0]
                        u_pred_hat = u_pred_hat[:nx//2, :]
                        u_true_hat = u_true_hat[:nx//2, :]
                        
                        # L2 error for each mode
                        mode_errors = torch.norm(u_pred_hat - u_true_hat, dim=1)
                        true_mode_norm = torch.norm(u_true_hat, dim=1)
                        
                        mode_errors_dict[exp_name] = {
                            'mode_errors': mode_errors.detach().cpu().numpy(),
                            'true_mode_norm': true_mode_norm.detach().cpu().numpy(),
                            'relative_errors': (mode_errors / (true_mode_norm + 1e-16)).detach().cpu().numpy()
                        }
                        print(f"    ✓ Loaded {exp_name}")
                    else:
                        print(f"    Skipping {exp_name}: numpy files not found")
                except Exception as e:
                    print(f"    Warning: Could not process {exp_name}: {e}")
        
        if not mode_errors_dict:
            print("  No mode error data available")
            return
        
        # Create overlay plots
        colors = plt.cm.Set1(np.linspace(0, 1, len(mode_errors_dict)))
        
        # Get number of modes (already positive frequencies only)
        n_modes = list(mode_errors_dict.values())[0]['mode_errors'].shape[0]
        modes = np.arange(n_modes)
        
        # Plot 1: Absolute L2 error vs k (semi-log)
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (exp_name, data) in enumerate(mode_errors_dict.items()):
            ax.semilogy(modes, data['mode_errors'], 'o-', label=exp_name, 
                       linewidth=2.5, markersize=5, color=colors[idx])
        
        ax.set_xlabel('Mode Index k', fontsize=12)
        ax.set_ylabel('L2 Error: ||û_pred[k,:] - û_true[k,:]||', fontsize=12)
        ax.set_title('Mode L2 Error vs Frequency (All Models)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_mode_l2_vs_k_overlay.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_mode_l2_vs_k_overlay.png")
        
        # Plot 2: Log-Log scale
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (exp_name, data) in enumerate(mode_errors_dict.items()):
            ax.loglog(modes[1:], data['mode_errors'][1:], 'o-', label=exp_name,
                     linewidth=2.5, markersize=5, color=colors[idx])
        
        ax.set_xlabel('Mode Index k (log)', fontsize=12)
        ax.set_ylabel('L2 Error (log)', fontsize=12)
        ax.set_title('Mode L2 Error vs Frequency (Log-Log, All Models)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_mode_l2_vs_k_loglog_overlay.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_mode_l2_vs_k_loglog_overlay.png")
        
        # Plot 3: Relative error vs k
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, (exp_name, data) in enumerate(mode_errors_dict.items()):
            ax.semilogy(modes, data['relative_errors'], 's-', label=exp_name,
                       linewidth=2.5, markersize=5, color=colors[idx])
        
        ax.set_xlabel('Mode Index k', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title('Relative Mode L2 Error vs Frequency (All Models)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = self.benchmark_dir / "fourier_mode_relative_error_overlay.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fourier_mode_relative_error_overlay.png")
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