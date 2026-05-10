"""
Plotting and visualization utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class Plotter:
    """Handles all plotting and visualization."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frequency_loss_log = {}

    def log_frequency_loss(self, model_type: str, frequency: float,
                           final_loss: float, experiment_name: str = None) -> None:
        """Record a final loss value for a model/frequency pair."""
        self.frequency_loss_log.setdefault(model_type, []).append({
            "frequency": float(frequency),
            "final_loss": float(final_loss),
            "experiment": experiment_name,
        })

    def save_frequency_loss_log(self, save_name: str = "frequency_loss_log.json") -> None:
        """Save the recorded frequency-loss log to JSON."""
        if not self.save_dir or not self.frequency_loss_log:
            return

        import json
        path = self.save_dir / save_name
        with open(path, "w") as f:
            json.dump(self.frequency_loss_log, f, indent=2)
        print(f"Saved: {path}")

    def plot_training_history(self, history: np.ndarray, 
                             figsize: Tuple[int, int] = (10, 6),
                             title: str = "Training History",
                             save_name: Optional[str] = None) -> None:
        """
        Plot training loss history.
        
        Args:
            history: array of shape (epochs, 5) with columns
                     [total, pde, consistency, initial, boundary]
            figsize: figure size
            title: plot title
            save_name: if provided, save figure to this filename
        """
        plt.figure(figsize=figsize)
        
        labels = ["total", "PDE", "consistency", "initial", "boundary"]
        for i, label in enumerate(labels):
            plt.semilogy(history[:, i], label=label, linewidth=2, alpha=0.8)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()
    
    def plot_solution_comparison(self, U_true: torch.Tensor, U_pred: torch.Tensor,
                                x_grid: torch.Tensor = None, 
                                t_grid: torch.Tensor = None,
                                figsize: Tuple[int, int] = (15, 4),
                                title_prefix: str = "Heat Equation Solution",
                                save_name: Optional[str] = None) -> float:
        """
        Plot exact solution, prediction, and error.
        
        Args:
            U_true: exact solution, shape (nx, nt)
            U_pred: predicted solution, shape (nx, nt)
            x_grid: x coordinates for extent (optional)
            t_grid: t coordinates for extent (optional)
            figsize: figure size
            title_prefix: prefix for titles
            save_name: if provided, save figure to this filename
        
        Returns:
            rel_l2: relative L2 error
        """
        # Convert to numpy if needed
        U_true_np = U_true.detach().cpu().numpy() if isinstance(U_true, torch.Tensor) else U_true
        U_pred_np = U_pred.detach().cpu().numpy() if isinstance(U_pred, torch.Tensor) else U_pred
        
        # Compute relative L2 error
        rel_l2 = np.linalg.norm(U_pred_np - U_true_np) / np.linalg.norm(U_true_np)
        
        # Determine extent
        if x_grid is not None and t_grid is not None:
            extent = [float(t_grid[0]), float(t_grid[-1]), 
                     float(x_grid[0]), float(x_grid[-1])]
        else:
            extent = [0, 1, 0, 1]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Exact solution
        im0 = axes[0].imshow(U_true_np, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("x")
        axes[0].set_title(f"{title_prefix} - Exact")
        plt.colorbar(im0, ax=axes[0], label="u")
        
        # Prediction
        im1 = axes[1].imshow(U_pred_np, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("x")
        axes[1].set_title(f"{title_prefix} - Prediction")
        plt.colorbar(im1, ax=axes[1], label="u")
        
        # Error
        error = U_pred_np - U_true_np
        im2 = axes[2].imshow(error, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("x")
        axes[2].set_title(f"Error (RelL2={rel_l2:.2e})")
        plt.colorbar(im2, ax=axes[2], label="error")
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()
        
        return rel_l2
    
    def plot_solution_slices(self, U_true: torch.Tensor, U_pred: torch.Tensor,
                        x_grid: torch.Tensor = None,
                        t_grid: torch.Tensor = None,
                        figsize: Tuple[int, int] = (14, 5),
                        save_name: Optional[str] = None) -> None:
        """..."""
        U_true_np = U_true.detach().cpu().numpy() if isinstance(U_true, torch.Tensor) else U_true
        U_pred_np = U_pred.detach().cpu().numpy() if isinstance(U_pred, torch.Tensor) else U_pred
        
        nx, nt = U_true_np.shape
        
        if x_grid is None:
            x_grid = np.linspace(0, 1, nx)
        else:
            x_grid = x_grid.detach().cpu().numpy() if isinstance(x_grid, torch.Tensor) else x_grid
            x_grid = x_grid.squeeze()  # ADDED: flatten if needed
        
        if t_grid is None:
            t_grid = np.linspace(0, 1, nt)
        else:
            t_grid = t_grid.detach().cpu().numpy() if isinstance(t_grid, torch.Tensor) else t_grid
            t_grid = t_grid.squeeze()  # ADDED: flatten if needed
        
        # Select a few time steps for visualization
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Slices at different times
        ax = axes[0]
        for idx in time_indices:
            t_val = float(t_grid[idx])  # ADDED: convert to float
            ax.plot(x_grid, U_true_np[:, idx], 'o-', label=f't={t_val:.2f} (exact)', linewidth=2)
            ax.plot(x_grid, U_pred_np[:, idx], 's--', label=f't={t_val:.2f} (pred)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u(x, t)", fontsize=12)
        ax.set_title("Solution at different times", fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Slices at different positions
        ax = axes[1]
        x_indices = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]
        for idx in x_indices:
            x_val = float(x_grid[idx])  # ADDED: convert to float
            ax.plot(t_grid, U_true_np[idx, :], 'o-', label=f'x={x_val:.2f} (exact)', linewidth=2)
            ax.plot(t_grid, U_pred_np[idx, :], 's--', label=f'x={x_val:.2f} (pred)', linewidth=2, alpha=0.7)
        

        
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u(x, t)", fontsize=12)
        ax.set_title("Solution at different times", fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Slices at different positions
        ax = axes[1]
        x_indices = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]
        for idx in x_indices:
            x_val = x_grid[idx]
            ax.plot(t_grid, U_true_np[idx, :], 'o-', label=f'x={x_val:.2f} (exact)', linewidth=2)
            ax.plot(t_grid, U_pred_np[idx, :], 's--', label=f'x={x_val:.2f} (pred)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel("u(x, t)", fontsize=12)
        ax.set_title("Solution at different positions", fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()
    
    def plot_loss_comparison(self, loss_histories: Dict[str, np.ndarray],
                            figsize: Tuple[int, int] = (10, 6),
                            title: str = "Loss Comparison",
                            save_name: Optional[str] = None) -> None:
        """
        Plot and compare training histories from multiple experiments.
        
        Args:
            loss_histories: dict mapping experiment name to history array
            figsize: figure size
            title: plot title
            save_name: if provided, save figure to this filename
        """
        plt.figure(figsize=figsize)
        
        for name, history in loss_histories.items():
            plt.semilogy(history[:, 0], label=name, linewidth=2, alpha=0.8)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Total Loss", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()
    
    def plot_error_metrics(self, metrics: Dict[str, Dict[str, float]],
                          figsize: Tuple[int, int] = (10, 6),
                          save_name: Optional[str] = None) -> None:
        """
        Plot error metrics (RelL2, MAE, etc.) across experiments.
        
        Args:
            metrics: dict mapping experiment name to dict of metrics
            figsize: figure size
            save_name: if provided, save figure to this filename
        """
        names = list(metrics.keys())
        
        # Extract all metric types
        metric_types = list(metrics[names[0]].keys()) if names else []

        fig, axes = plt.subplots(1, len(metric_types), figsize=(5 * len(metric_types), 5))
        if len(metric_types) == 1:
            axes = [axes]

        for i, metric_type in enumerate(metric_types):
            values = [metrics[name][metric_type] for name in names]
            axes[i].bar(names, values, alpha=0.7, edgecolor='black')
            axes[i].set_ylabel(metric_type, fontsize=12)
            axes[i].set_title(f"{metric_type} Comparison", fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

            # plt.show()

    def plot_frequency_sweep_losses(self,
                                    sweep_results: Dict[str, List[Dict[str, float]]],
                                    figsize: Tuple[int, int] = (9, 6),
                                    save_name: Optional[str] = None) -> None:
        """
        Plot final training loss after the configured epochs as a function of IC frequency.

        Args:
            sweep_results: mapping from model type to rows with frequency/final_loss
            figsize: figure size
            save_name: if provided, save figure to this filename
        """
        plt.figure(figsize=figsize)

        plotted_any = False
        for model_type, rows in sweep_results.items():
            valid_rows = [
                row for row in rows
                if row.get("final_loss") is not None and row.get("frequency") is not None
            ]
            if not valid_rows:
                continue

            valid_rows = sorted(valid_rows, key=lambda row: row["frequency"])
            frequencies = [row["frequency"] for row in valid_rows]
            final_losses = [row["final_loss"] for row in valid_rows]
            plt.semilogy(
                frequencies,
                final_losses,
                marker="o",
                linewidth=2,
                label=model_type,
            )
            plotted_any = True

        plt.xlabel("Initial condition frequency", fontsize=12)
        plt.ylabel("Final total loss", fontsize=12)
        plt.title("Final Loss After Training vs Frequency", fontsize=14)
        plt.grid(True, alpha=0.3)
        if plotted_any:
            plt.legend(fontsize=10)
        plt.tight_layout()

        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()


    def plot_entropy_history(self, history: np.ndarray,
                             figsize: Tuple[int, int] = (8, 4),
                             save_name: Optional[str] = "entropy_history.png") -> None:
        """
        Plot Von Neumann Entanglement Entropy history.
        Assumes entropy is stored at index 5 of the history array.
        """
        plt.figure(figsize=figsize)

        # History[:, 5] contains the entropy values we appended in Trainer
        plt.plot(history[:, 5], color='purple', linewidth=2, label="Von Neumann Entropy (S)")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Entropy", fontsize=12)
        plt.title("Bipartite Entanglement Entropy over Training", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name and self.save_dir:
            path = self.save_dir / save_name
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        #plt.show()