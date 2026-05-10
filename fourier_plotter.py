"""
Plotting functions for Fourier space metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


class FourierPlotter:
    """Create visualizations for Fourier space metrics."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize plotter.
        
        Args:
            save_dir: Directory to save plots. If None, plots are not saved.
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_power_spectrum(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                           figsize: tuple = (12, 5), 
                           save_name: Optional[str] = None) -> None:
        """
        Plot power spectrum comparison (predicted vs true).
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFTs
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Power spectra (averaged over time)
        power_pred = (torch.abs(u_pred_hat) ** 2).mean(dim=1)
        power_true = (torch.abs(u_true_hat) ** 2).mean(dim=1)
        
        # Frequencies (normalized)
        nx = u_pred.shape[0]
        freqs = np.fft.fftfreq(nx)[:nx//2]
        
        # Convert to numpy
        power_pred_np = power_pred[:nx//2].detach().cpu().numpy()
        power_true_np = power_true[:nx//2].detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Linear scale
        ax1.semilogy(freqs, power_true_np, 'o-', label='True', linewidth=2, markersize=4)
        ax1.semilogy(freqs, power_pred_np, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_xlabel('Normalized Frequency', fontsize=11)
        ax1.set_ylabel('Power Spectrum', fontsize=11)
        ax1.set_title('Power Spectrum Comparison', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Log-log scale
        ax2.loglog(freqs[1:], power_true_np[1:], 'o-', label='True', linewidth=2, markersize=4)
        ax2.loglog(freqs[1:], power_pred_np[1:], 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
        ax2.set_xlabel('Normalized Frequency (log)', fontsize=11)
        ax2.set_ylabel('Power Spectrum (log)', fontsize=11)
        ax2.set_title('Power Spectrum (Log-Log)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_spectral_errors(self, fourier_history: list,
                            figsize: tuple = (14, 5),
                            save_name: Optional[str] = None) -> None:
        """
        Plot evolution of spectral errors during training.
        
        Args:
            fourier_history: List of Fourier metrics dicts from training
            figsize: Figure size
            save_name: Name to save plot
        """
        if not fourier_history:
            print("No Fourier history to plot")
            return
        
        # Extract data
        epochs = np.array([m['epoch'] for m in fourier_history])
        spectral_l2 = np.array([m['spectral_l2'] for m in fourier_history])
        power_error = np.array([m['power_spectrum_error'] for m in fourier_history])
        high_freq = np.array([m['high_freq_error'] for m in fourier_history])
        low_freq = np.array([m['low_freq_error'] for m in fourier_history])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Spectral L2 error
        axes[0, 0].semilogy(epochs, spectral_l2, 'o-', linewidth=2, markersize=5, color='C0')
        axes[0, 0].set_ylabel('Spectral L2 Error', fontsize=11)
        axes[0, 0].set_title('Spectral L2 Error vs Epoch', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power spectrum error
        axes[0, 1].semilogy(epochs, power_error, 'o-', linewidth=2, markersize=5, color='C1')
        axes[0, 1].set_ylabel('Power Spectrum Error', fontsize=11)
        axes[0, 1].set_title('Power Spectrum Error vs Epoch', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency error split
        axes[1, 0].semilogy(epochs, low_freq, 'o-', label='Low Freq', linewidth=2, markersize=5)
        axes[1, 0].semilogy(epochs, high_freq, 's--', label='High Freq', linewidth=2, markersize=5)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Error', fontsize=11)
        axes[1, 0].set_title('Low vs High Frequency Error', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # All errors together
        axes[1, 1].semilogy(epochs, spectral_l2, 'o-', label='Spectral L2', linewidth=2, markersize=4)
        axes[1, 1].semilogy(epochs, power_error, 's-', label='Power', linewidth=2, markersize=4)
        axes[1, 1].semilogy(epochs, low_freq, '^-', label='Low Freq', linewidth=2, markersize=4)
        axes[1, 1].semilogy(epochs, high_freq, 'd-', label='High Freq', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Error', fontsize=11)
        axes[1, 1].set_title('All Spectral Errors', fontsize=12)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_spectral_concentration(self, fourier_history: list,
                                   figsize: tuple = (10, 5),
                                   save_name: Optional[str] = None) -> None:
        """
        Plot spectral concentration (smoothness) during training.
        
        Args:
            fourier_history: List of Fourier metrics dicts from training
            figsize: Figure size
            save_name: Name to save plot
        """
        if not fourier_history:
            print("No Fourier history to plot")
            return
        
        # Extract data
        epochs = np.array([m['epoch'] for m in fourier_history])
        conc_pred = np.array([m['concentration_pred'] for m in fourier_history])
        conc_true = np.array([m['concentration_true'] for m in fourier_history])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(epochs, conc_true, 'o-', label='True Solution', linewidth=2.5, markersize=6)
        ax.plot(epochs, conc_pred, 's--', label='Predicted Solution', linewidth=2.5, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Spectral Concentration (90% energy)', fontsize=12)
        ax.set_title('Solution Smoothness: Spectral Concentration vs Training', fontsize=13)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        ax.text(0.5, 0.05, 'Higher = smoother (more energy in low frequencies)', 
               transform=ax.transAxes, fontsize=10, style='italic', ha='center')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_peak_frequency(self, fourier_history: list,
                           figsize: tuple = (10, 5),
                           save_name: Optional[str] = None) -> None:
        """
        Plot dominant frequency during training.
        
        Args:
            fourier_history: List of Fourier metrics dicts from training
            figsize: Figure size
            save_name: Name to save plot
        """
        if not fourier_history:
            print("No Fourier history to plot")
            return
        
        # Extract data
        epochs = np.array([m['epoch'] for m in fourier_history])
        peak_pred = np.array([m['peak_freq_pred'] for m in fourier_history])
        peak_true = np.array([m['peak_freq_true'] for m in fourier_history])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(epochs, peak_true, 'o-', label='True Solution', linewidth=2.5, markersize=6)
        ax.plot(epochs, peak_pred, 's--', label='Predicted Solution', linewidth=2.5, markersize=6, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Dominant Spatial Frequency (periods)', fontsize=12)
        ax.set_title('Peak Frequency vs Training', fontsize=13)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_mode_l2_errors(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                           figsize: tuple = (12, 6),
                           save_name: Optional[str] = None) -> None:
        """
        Plot L2 error for each Fourier mode (eigenvalue).
        
        Shows L2 distance between each predicted mode and true mode.
        Higher modes tend to have larger errors (oscillations are harder to learn).
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFTs along x-direction
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Compute L2 error for each mode (across time dimension)
        mode_errors = torch.norm(u_pred_hat - u_true_hat, dim=1)
        true_mode_norm = torch.norm(u_true_hat, dim=1)
        
        # Normalized error (relative to magnitude of true mode)
        relative_errors = mode_errors / (true_mode_norm + 1e-16)
        
        nx = u_pred.shape[0]
        modes = np.arange(nx)
        
        # Convert to numpy
        mode_errors_np = mode_errors.detach().cpu().numpy()
        relative_errors_np = relative_errors.detach().cpu().numpy()
        true_mode_norm_np = true_mode_norm.detach().cpu().numpy()
        
        # Create main plot: L2 error vs mode index k
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot L2 error as function of mode k
        ax.semilogy(modes[:nx//2], mode_errors_np[:nx//2], 'o-', linewidth=2.5, 
                   markersize=6, color='C0', label='L2 Error')
        
        ax.set_xlabel('Mode Index k (Frequency)', fontsize=13)
        ax.set_ylabel('L2 Error: ||û_pred[k,:] - û_true[k,:]||', fontsize=13)
        ax.set_title('Mode-by-Mode L2 Error: Error vs Frequency Index', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11)
        
        # Add shaded regions for low/high frequency bands (assuming 1/4 splits)
        n_low = nx // 4
        n_high_start = 3 * nx // 4
        ax.axvspan(0, n_low, alpha=0.1, color='green', label='Low Frequencies')
        ax.axvspan(n_high_start, nx//2, alpha=0.1, color='red', label='High Frequencies')
        ax.legend(fontsize=11, loc='best')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_mode_l2_vs_k_detailed(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                                   figsize: tuple = (14, 10),
                                   save_name: Optional[str] = None) -> None:
        """
        Detailed 4-panel plot of L2 error vs mode index k.
        
        Shows absolute, relative, linear, and log-scale views.
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFTs along x-direction
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Compute L2 error for each mode (across time dimension)
        mode_errors = torch.norm(u_pred_hat - u_true_hat, dim=1)
        true_mode_norm = torch.norm(u_true_hat, dim=1)
        
        # Normalized error (relative to magnitude of true mode)
        relative_errors = mode_errors / (true_mode_norm + 1e-16)
        
        nx = u_pred.shape[0]
        modes = np.arange(nx)
        
        # Convert to numpy
        mode_errors_np = mode_errors.detach().cpu().numpy()
        relative_errors_np = relative_errors.detach().cpu().numpy()
        true_mode_norm_np = true_mode_norm.detach().cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: L2 error vs k (log-log, most important)
        ax = axes[0, 0]
        ax.loglog(modes[1:nx//2], mode_errors_np[1:nx//2], 'o-', linewidth=2.5, 
                 markersize=6, color='C0')
        ax.set_xlabel('Mode Index k (log)', fontsize=12)
        ax.set_ylabel('L2 Error (log)', fontsize=12)
        ax.set_title('L2 Error vs Mode k (Log-Log)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 2: L2 error vs k (log scale)
        ax = axes[0, 1]
        ax.semilogy(modes[:nx//2], mode_errors_np[:nx//2], 'o-', linewidth=2.5,
                   markersize=6, color='C1')
        ax.set_xlabel('Mode Index k', fontsize=12)
        ax.set_ylabel('L2 Error (log)', fontsize=12)
        ax.set_title('L2 Error vs Mode k (Semi-log)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Relative error vs k
        ax = axes[1, 0]
        ax.semilogy(modes[:nx//2], relative_errors_np[:nx//2], 's-', linewidth=2.5,
                   markersize=6, color='C2')
        ax.set_xlabel('Mode Index k', fontsize=12)
        ax.set_ylabel('Relative L2 Error', fontsize=12)
        ax.set_title('Relative Error vs Mode k', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 4: True mode magnitude vs error
        ax = axes[1, 1]
        scatter = ax.scatter(true_mode_norm_np[:nx//2], mode_errors_np[:nx//2],
                            c=modes[:nx//2], cmap='viridis', s=80, alpha=0.7, 
                            edgecolors='black', linewidth=0.5)
        ax.set_xlabel('True Mode Magnitude |û_true[k,:]|', fontsize=12)
        ax.set_ylabel('L2 Error |û_pred[k,:] - û_true[k,:]|', fontsize=12)
        ax.set_title('Error vs True Mode Magnitude', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mode Index k', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_mode_comparison(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                            n_modes: int = 20,
                            figsize: tuple = (14, 10),
                            save_name: Optional[str] = None) -> None:
        """
        Plot individual mode comparisons (predicted vs true).
        
        Shows the first n_modes Fourier modes side by side.
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
            n_modes: Number of modes to plot
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFTs
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Get magnitude and phase
        pred_mag = torch.abs(u_pred_hat[:n_modes])
        true_mag = torch.abs(u_true_hat[:n_modes])
        pred_phase = torch.angle(u_pred_hat[:n_modes])
        true_phase = torch.angle(u_true_hat[:n_modes])
        
        # Convert to numpy
        pred_mag_np = pred_mag.detach().cpu().numpy()
        true_mag_np = true_mag.detach().cpu().numpy()
        pred_phase_np = pred_phase.detach().cpu().numpy()
        true_phase_np = true_phase.detach().cpu().numpy()
        
        n_cols = 4
        n_rows = (n_modes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for mode in range(n_modes):
            ax = axes[mode]
            
            # Plot magnitude spectrum over time
            ax.plot(pred_mag_np[mode], 's--', label='Pred', linewidth=2, markersize=3, alpha=0.7)
            ax.plot(true_mag_np[mode], 'o-', label='True', linewidth=2, markersize=3)
            
            ax.set_title(f'Mode {mode}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            if mode == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        # Hide unused subplots
        for mode in range(n_modes, len(axes)):
            axes[mode].set_visible(False)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_cumulative_energy(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                              figsize: tuple = (10, 6),
                              save_name: Optional[str] = None) -> None:
        """
        Plot cumulative energy (sum of power spectrum) vs mode index.
        
        Shows how quickly energy accumulates across modes.
        Steep curves = energy concentrated in low modes (smooth solutions).
        Flat curves = energy spread across modes (oscillatory solutions).
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFTs
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Power spectra (averaged over time)
        power_pred = (torch.abs(u_pred_hat) ** 2).mean(dim=1)
        power_true = (torch.abs(u_true_hat) ** 2).mean(dim=1)
        
        # Cumulative energy
        cumsum_pred = torch.cumsum(power_pred, dim=0)
        cumsum_true = torch.cumsum(power_true, dim=0)
        
        # Normalize to percentage
        cumsum_pred_norm = cumsum_pred / cumsum_pred[-1] * 100
        cumsum_true_norm = cumsum_true / cumsum_true[-1] * 100
        
        # Convert to numpy
        modes = np.arange(len(cumsum_pred))
        cumsum_pred_np = cumsum_pred_norm.detach().cpu().numpy()
        cumsum_true_np = cumsum_true_norm.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(modes, cumsum_true_np, 'o-', label='True', linewidth=2.5, markersize=3)
        ax.plot(modes, cumsum_pred_np, 's--', label='Predicted', linewidth=2.5, markersize=3, alpha=0.8)
        
        # Add reference lines
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90% energy')
        ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% energy')
        
        ax.set_xlabel('Mode Index', fontsize=12)
        ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
        ax.set_title('Cumulative Energy Distribution: How Fast Does Energy Accumulate?', fontsize=13)
        ax.set_ylim([0, 105])
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_all_spectral(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                         fourier_history: list,
                         figsize: tuple = (16, 12),
                         save_name: Optional[str] = None) -> None:
        """
        Create comprehensive spectral analysis plot.
        
        Args:
            u_pred: Predicted solution
            u_true: True solution
            fourier_history: Training history
            figsize: Figure size
            save_name: Name to save plot
        """
        fig = plt.figure(figsize=figsize)
        
        # Power spectrum (top left)
        ax1 = plt.subplot(2, 3, 1)
        u_pred_hat = torch.fft.fft(u_pred, dim=0)
        u_true_hat = torch.fft.fft(u_true, dim=0)
        power_pred = (torch.abs(u_pred_hat) ** 2).mean(dim=1)
        power_true = (torch.abs(u_true_hat) ** 2).mean(dim=1)
        nx = u_pred.shape[0]
        freqs = np.fft.fftfreq(nx)[:nx//2]
        ax1.semilogy(freqs, power_true[:nx//2].cpu().numpy(), 'o-', label='True', linewidth=2)
        ax1.semilogy(freqs, power_pred[:nx//2].cpu().numpy(), 's--', label='Pred', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Frequency', fontsize=10)
        ax1.set_ylabel('Power', fontsize=10)
        ax1.set_title('Power Spectrum', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Spectral L2 error
        if fourier_history:
            epochs = np.array([m['epoch'] for m in fourier_history])
            spectral_l2 = np.array([m['spectral_l2'] for m in fourier_history])
            ax2 = plt.subplot(2, 3, 2)
            ax2.semilogy(epochs, spectral_l2, 'o-', linewidth=2, color='C0')
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Error', fontsize=10)
            ax2.set_title('Spectral L2 Error', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Concentration
            conc_pred = np.array([m['concentration_pred'] for m in fourier_history])
            conc_true = np.array([m['concentration_true'] for m in fourier_history])
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(epochs, conc_true, 'o-', label='True', linewidth=2)
            ax3.plot(epochs, conc_pred, 's--', label='Pred', linewidth=2, alpha=0.7)
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel('Concentration', fontsize=10)
            ax3.set_title('Spectral Concentration', fontsize=11)
            ax3.set_ylim([0, 1])
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Freq error split
            low_freq = np.array([m['low_freq_error'] for m in fourier_history])
            high_freq = np.array([m['high_freq_error'] for m in fourier_history])
            ax4 = plt.subplot(2, 3, 4)
            ax4.semilogy(epochs, low_freq, 'o-', label='Low', linewidth=2)
            ax4.semilogy(epochs, high_freq, 's-', label='High', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Epoch', fontsize=10)
            ax4.set_ylabel('Error', fontsize=10)
            ax4.set_title('Freq Split Error', fontsize=11)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            # Peak frequency
            peak_pred = np.array([m['peak_freq_pred'] for m in fourier_history])
            peak_true = np.array([m['peak_freq_true'] for m in fourier_history])
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(epochs, peak_true, 'o-', label='True', linewidth=2)
            ax5.plot(epochs, peak_pred, 's--', label='Pred', linewidth=2, alpha=0.7)
            ax5.set_xlabel('Epoch', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.set_title('Peak Frequency', fontsize=11)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            
            # All errors
            power_error = np.array([m['power_spectrum_error'] for m in fourier_history])
            ax6 = plt.subplot(2, 3, 6)
            ax6.semilogy(epochs, spectral_l2, 'o-', label='L2', linewidth=2, markersize=4)
            ax6.semilogy(epochs, power_error, 's-', label='Power', linewidth=2, markersize=4)
            ax6.semilogy(epochs, low_freq, '^-', label='Low', linewidth=2, markersize=4)
            ax6.semilogy(epochs, high_freq, 'd-', label='High', linewidth=2, markersize=4)
            ax6.set_xlabel('Epoch', fontsize=10)
            ax6.set_ylabel('Error', fontsize=10)
            ax6.set_title('All Errors', fontsize=11)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()

    def plot_energy_distribution(self, u_true: torch.Tensor,
                                figsize: tuple = (12, 10),
                                save_name: Optional[str] = None) -> None:
        """
        Plot energy distribution across frequency modes.
        
        Args:
            u_true: True solution, shape (nx, nt)
            figsize: Figure size
            save_name: Name to save plot
        """
        # Compute FFT
        u_true_hat = torch.fft.fft(u_true, dim=0)
        
        # Keep only positive frequencies
        nx = u_true.shape[0]
        u_true_hat = u_true_hat[:nx//2, :]
        
        # Compute energy per mode: |û[k,:]|²
        energy_per_mode = torch.sum(torch.abs(u_true_hat) ** 2, dim=1)
        total_energy = torch.sum(energy_per_mode)
        energy_fraction = energy_per_mode / total_energy
        cumulative_energy = torch.cumsum(energy_fraction, dim=0)
        
        # Convert to numpy
        modes = np.arange(len(energy_fraction))
        energy_frac = energy_fraction.cpu().numpy()
        cumul_energy = cumulative_energy.cpu().numpy()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
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
        ax2.plot(modes, cumul_energy, 'o-', linewidth=2.5, markersize=4, color='#059669')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% energy')
        ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% energy')
        ax2.fill_between(modes, cumul_energy, alpha=0.2, color='#059669')
        ax2.set_xlabel('Mode Index k', fontsize=12)
        ax2.set_ylabel('Cumulative Energy Fraction', fontsize=12)
        ax2.set_title('Cumulative Energy vs Mode Index', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Add shading for low/high frequency regions
        ax2.axvspan(0, nx//4, alpha=0.1, color='green')
        ax2.axvspan(3*nx//4, nx//2, alpha=0.1, color='red')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()