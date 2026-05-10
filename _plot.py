import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

def plot_frequency_sweep_entropy(
                    frequencies: List[float] | np.ndarray,
                    final_entropies: List[float],
                    figsize: Tuple[int, int] = (9, 6),
                    save_name: Optional[str] = None,
                    save_dir: Optional[Path] = None
                    ) -> None:
    """
    Plot final entanglement entropy after the configured epochs as a function of IC frequency.

    Args:
        sweep_results: mapping from model type to rows with frequency/final_entropy
        figsize: figure size
        save_name: if provided, save figure to this filename
    """
    plt.figure(figsize=figsize)

    plotted_any = False
    plt.plot(
            frequencies,
            final_entropies,
            marker="s",
            linewidth=2,
            label='QPINN',
        )

    plt.xlabel("Initial condition frequency", fontsize=12)
    plt.ylabel("Final Von Neumann Entropy", fontsize=12)
    plt.title("Final Entanglement Entropy vs Frequency", fontsize=14)
    plt.grid(True, alpha=0.3)
    if plotted_any:
        plt.legend(fontsize=10)
    plt.tight_layout()

    if save_name and save_dir:
        path = save_dir / save_name
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")


if __name__ == "__main__":
    frequencies = np.arange(0.0, 15.00, 0.25)

    # Assuming the data is in a file named 'data.json'
    with open('results/entropy_sweep/20260510_070509/frequency_loss_log.json', 'r') as f:
        data = json.load(f)

    # Extract using a list comprehension
    entropies = [item['final_entropy'] for item in data['qpinn']]

    plot_frequency_sweep_entropy(frequencies, entropies, save_name="frequency_entropy_sweep.png", save_dir=Path("results/entropy_sweep/20260510_070509"))