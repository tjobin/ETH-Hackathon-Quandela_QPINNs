"""
Fourier space metrics for PINNs.
Evaluates solution quality in frequency domain.
"""

import torch
import numpy as np
from typing import Dict, Tuple


class FourierMetrics:
    """Compute metrics in Fourier space (frequency domain)."""
    
    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float32,
                 low_freq_fraction: float = 0.5, high_freq_fraction: float = 0.5):
        """
        Initialize Fourier metrics.
        
        Args:
            device: Torch device
            dtype: Torch dtype
            low_freq_fraction: Fraction of modes to consider as "low" frequencies (0.5 = bottom 50%)
            high_freq_fraction: Fraction of modes to consider as "high" frequencies (0.5 = top 50%)
        
        Example:
            - low_freq_fraction=0.25, high_freq_fraction=0.25
              → Low: modes 0-15/60, High: modes 45-60/60
            
            - low_freq_fraction=0.1, high_freq_fraction=0.1
              → Low: modes 0-6/60, High: modes 54-60/60 (tighter split)
            
            - low_freq_fraction=0.5, high_freq_fraction=0.2
              → Low: modes 0-30/60, High: modes 48-60/60 (broader low, narrower high)
        """
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        self.low_freq_fraction = low_freq_fraction
        self.high_freq_fraction = high_freq_fraction
    
    @staticmethod
    def fft_1d(u: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D FFT along the first dimension (x-direction).
        
        Args:
            u: Solution tensor of shape (nx, nt) or (nx,)
        
        Returns:
            u_hat: FFT coefficients (complex)
        """
        if u.dim() == 1:
            u_hat = torch.fft.fft(u)
        else:
            # FFT along x-direction (dim 0)
            u_hat = torch.fft.fft(u, dim=0)
        return u_hat
    
    @staticmethod
    def power_spectrum(u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute power spectrum (magnitude squared of FFT).
        
        Args:
            u_hat: FFT coefficients (complex)
        
        Returns:
            power: Power spectrum |u_hat|²
        """
        return torch.abs(u_hat) ** 2
    
    def compute_mode_energy(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute energy per mode (power spectrum).
        
        Args:
            u_hat: FFT coefficients, shape (n_modes, nt)
        
        Returns:
            energy: Energy per mode, shape (n_modes,)
        """
        # Energy = |û[k,:]|² summed over time
        return torch.sum(torch.abs(u_hat) ** 2, dim=1)
    
    def spectral_error(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> Dict[str, float]:
        """
        Compute spectral error between predicted and true solutions.
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
        
        Returns:
            metrics: Dictionary with spectral error metrics
        """
        # Compute FFTs
        u_pred_hat = self.fft_1d(u_pred)
        u_true_hat = self.fft_1d(u_true)
        
        # Keep only positive frequencies for consistency
        nx = u_pred.shape[0]
        u_pred_hat = u_pred_hat[:nx//2, :]
        u_true_hat = u_true_hat[:nx//2, :]
        
        # Power spectra
        power_pred = self.power_spectrum(u_pred_hat)
        power_true = self.power_spectrum(u_true_hat)
        
        # Spectral error metrics
        # 1. L2 error in frequency domain
        spectral_l2 = torch.norm(u_pred_hat - u_true_hat) / (torch.norm(u_true_hat) + 1e-16)
        
        # 2. Power spectrum error
        power_error = torch.norm(power_pred - power_true) / (torch.norm(power_true) + 1e-16)
        
        # 3. High-frequency error (top 50% of positive frequencies)
        n_high = max(1, int((nx // 2) * self.high_freq_fraction))  # Top 50% of positive frequencies
        # High freq: last n_high modes out of nx//2
        high_start = (nx // 2) - n_high
        high_freq_error = torch.norm(u_pred_hat[high_start:, :] - u_true_hat[high_start:, :])
        
        # 4. Low-frequency error (bottom 50% of positive frequencies)
        n_low = max(1, int((nx // 2) * self.low_freq_fraction))  # Bottom 50% of positive frequencies
        # Low freq: first n_low modes
        low_freq_error = torch.norm(u_pred_hat[:n_low, :] - u_true_hat[:n_low, :])
        
        # 5. Compute mode energies for analysis
        energy_true = self.compute_mode_energy(u_true_hat)
        energy_pred = self.compute_mode_energy(u_pred_hat)
        total_energy_true = torch.sum(energy_true)
        
        # Energy in low and high frequency bands
        energy_low_true = torch.sum(energy_true[:n_low])
        energy_high_true = torch.sum(energy_true[high_start:])
        
        return {
            "spectral_l2": float(spectral_l2),
            "power_spectrum_error": float(power_error),
            "high_freq_error": float(high_freq_error),
            "low_freq_error": float(low_freq_error),
            # Energy information
            "total_energy": float(total_energy_true),
            "low_freq_energy": float(energy_low_true),
            "high_freq_energy": float(energy_high_true),
            "low_freq_energy_fraction": float(energy_low_true / (total_energy_true + 1e-16)),
            "high_freq_energy_fraction": float(energy_high_true / (total_energy_true + 1e-16)),
        }
    
    def peak_frequency(self, u: torch.Tensor) -> float:
        """
        Find the dominant frequency (peak of power spectrum).
        
        Args:
            u: Solution tensor, shape (nx, nt) or (nx,)
        
        Returns:
            peak_freq: Normalized peak frequency (0 to 1)
        """
        u_hat = self.fft_1d(u)
        power = self.power_spectrum(u_hat)
        
        # Flatten if 2D
        if power.dim() > 1:
            power = power.flatten()
        
        # Find index of peak (excluding DC component at 0)
        peak_idx = torch.argmax(power[1:]) + 1
        nx = len(power)
        
        # Normalize to [0, 1]
        peak_freq = float(peak_idx) / nx
        return peak_freq
    
    def spectral_concentration(self, u: torch.Tensor, threshold: float = 0.9) -> float:
        """
        Compute what fraction of energy is in the lowest frequencies.
        
        Measures how much solution is concentrated in low frequencies.
        High value (close to 1) = smooth solution
        Low value = oscillatory solution
        
        Args:
            u: Solution tensor, shape (nx, nt) or (nx,)
            threshold: Fraction of energy to capture (default 0.9 = 90%)
        
        Returns:
            concentration: Fraction of modes needed to capture threshold energy
        """
        u_hat = self.fft_1d(u)
        power = self.power_spectrum(u_hat)
        
        # Flatten to 1D if needed
        if power.dim() > 1:
            power = power.flatten()
        
        # Total energy
        total_energy = torch.sum(power)
        
        # Find how many modes needed for threshold energy
        cumsum = torch.cumsum(power, dim=0)
        target_energy = threshold * total_energy
        
        # Find first index where cumsum >= target_energy
        mask = cumsum >= target_energy
        if mask.any():
            n_modes = int(torch.nonzero(mask, as_tuple=True)[0][0])
        else:
            n_modes = len(power)
        
        nx = len(power)
        concentration = float(n_modes) / max(1, nx)
        
        return concentration
    
    def compute_all_metrics(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> Dict:
        """
        Compute all Fourier metrics at once.
        
        Args:
            u_pred: Predicted solution, shape (nx, nt)
            u_true: True solution, shape (nx, nt)
        
        Returns:
            all_metrics: Dictionary with all Fourier metrics
        """
        metrics = {}
        
        # Spectral errors
        spectral_errs = self.spectral_error(u_pred, u_true)
        metrics.update(spectral_errs)
        
        # Peak frequencies
        metrics["peak_freq_pred"] = self.peak_frequency(u_pred)
        metrics["peak_freq_true"] = self.peak_frequency(u_true)
        
        # Spectral concentration
        metrics["concentration_pred"] = self.spectral_concentration(u_pred)
        metrics["concentration_true"] = self.spectral_concentration(u_true)
        
        return metrics


class FourierTrainingMonitor:
    """Monitor Fourier metrics during training."""
    
    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float32,
                 low_freq_fraction: float = 0.5, high_freq_fraction: float = 0.5):
        """
        Initialize Fourier monitoring.
        
        Args:
            device: Torch device
            dtype: Torch dtype
            low_freq_fraction: Fraction of modes to consider as "low" (0.5 = bottom 50%)
            high_freq_fraction: Fraction of modes to consider as "high" (0.5 = top 50%)
        """
        self.fourier = FourierMetrics(device, dtype, low_freq_fraction, high_freq_fraction)
        self.history = []  # List of dicts, one per evaluation
    
    def evaluate(self, model, config, device: torch.device, 
                 dtype: torch.dtype, epoch: int) -> Dict:
        """
        Evaluate Fourier metrics at current epoch.
        
        Args:
            model: Neural network model
            config: Experiment config with PDE and evaluation settings
            device: Torch device
            dtype: Torch dtype
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of Fourier metrics
        """
        # Create test grid
        nx, nt = 60, 60
        x = torch.linspace(config.pde.x_min, config.pde.x_max, nx, device=device, dtype=dtype)
        t = torch.linspace(config.pde.t_min, config.pde.t_max, nt, device=device, dtype=dtype)
        X, T = torch.meshgrid(x, t, indexing='ij')
        xt_grid = torch.stack([X.flatten(), T.flatten()], dim=1)
        
        # Forward pass (no gradients)
        with torch.no_grad():
            U_pred, _, *_ = model(xt_grid)
            U_pred = U_pred.reshape(nx, nt)
            
            # Exact solution
            x_expand = x.unsqueeze(1).expand(nx, nt)
            t_expand = t.unsqueeze(0).expand(nx, nt)
            U_true = config.pde.exact_solution(x_expand, t_expand)
        
        # Compute Fourier metrics
        metrics = self.fourier.compute_all_metrics(U_pred, U_true)
        metrics["epoch"] = epoch
        
        # Store in history
        self.history.append(metrics)
        
        return metrics
    
    def get_history(self) -> list:
        """Return all stored metrics."""
        return self.history
    
    def summary(self) -> Dict:
        """Get summary of Fourier metrics over training."""
        if not self.history:
            return {}
        
        # Convert to numpy for easier analysis
        keys = list(self.history[0].keys())
        keys.remove("epoch")  # Remove epoch column
        
        summary = {}
        for key in keys:
            values = np.array([m[key] for m in self.history])
            summary[key] = {
                "initial": float(values[0]),
                "final": float(values[-1]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        return summary