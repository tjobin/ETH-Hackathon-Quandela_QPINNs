"""
Model definitions for QPINN.
Supports different quantum and classical architectures.
"""

import torch
import torch.nn as nn
from typing import Tuple
from config import ModelConfig
from merlin.core.state_vector import StateVector
from utils import generate_fock_basis




class MerlinHeatQPINN(nn.Module):
    """
    QPINN model using MerLin quantum layer for heat equation.
    Enforces homogeneous Dirichlet BCs through hard constraint: u = x(1-x)*q_u
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature map: classical neural network to map (x, t) to feature space
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size),
        )
        
        # Import MerLin quantum layer
        try:
            import merlin as ML
            self.quantum = ML.QuantumLayer.simple(
                input_size=config.feature_size,
                output_size=config.quantum_output_size,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e
        
        # Readout: classical neural network to extract u and du/dx
        self.readout = nn.Sequential(
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),  # [q_u, ux_hat]
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            # Enforce homogeneous Dirichlet boundary conditions: u(0,t)=u(1,t)=0
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat

class MerlinHeatQPINN_freezeQMweights(nn.Module):
    """
    QPINN model with frozen quantum layer weights.
    Only classical layers (feature_map and readout) are trainable.
    Quantum layer acts as a fixed feature extractor.
    Enforces homogeneous Dirichlet BCs through hard constraint: u = x(1-x)*q_u
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature map: classical neural network to map (x, t) to feature space
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size),
        )
        
        # Import MerLin quantum layer
        try:
            import merlin as ML
            self.quantum = ML.QuantumLayer.simple(
                input_size=config.feature_size,
                output_size=config.quantum_output_size,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e
        
        # Freeze quantum layer weights immediately
        print("Freezing quantum layer weights...")
        for name, param in self.quantum.named_parameters():
            param.requires_grad = False
            print(f"  Frozen: {name}")
        
        # Readout: classical neural network to extract u and du/dx
        self.readout = nn.Sequential(
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),  # [q_u, ux_hat]
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)  # Quantum layer output (fixed, no gradient)
        out = self.readout(q)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            # Enforce homogeneous Dirichlet boundary conditions: u(0,t)=u(1,t)=0
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat


class ClassicalHeatQPINN(nn.Module):
    """
    Classical neural network QPINN (baseline without quantum layer).
    Useful for comparing quantum vs classical performance.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Larger classical network to match quantum layer capacity
        self.network = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size),
            nn.Tanh(),
            nn.Linear(config.feature_size,config.quantum_output_size),
            nn.Tanh(),
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),  # [u, ux_hat]
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        out = self.network(xt)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat


class DeepNetworkQPINN(nn.Module):
    """
    Deeper classical network QPINN for ablation studies.
    """
    
    def __init__(self, config: ModelConfig, depth: int = 3):
        super().__init__()
        self.config = config
        
        # Build deeper network
        layers = [nn.Linear(2, config.hidden_feature), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(config.hidden_feature, config.hidden_feature),
                nn.Tanh(),
            ])
        layers.append(nn.Linear(config.hidden_feature, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        out = self.network(xt)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat


class BuilderMerlinHeatQPINN(nn.Module):
    """
    QPINN model using a custom MerLin CircuitBuilder.
    Incorporates data re-uploading and higher photon counts.
    """
    
    def __init__(self, config_: ModelConfig):
        super().__init__()
        #ARCHITECTURE OF THIS MODEL QUITE RIGID, HARDCODING THE CONFIG
        self.config = ModelConfig(
        # The builder encodes data into exactly 2 modes (modes 0 and 1)
        feature_size=4,              
        # 4 modes with 3 photons yields C(4+3-1, 3) = 20 distinct Fock basis states
        # The probs() measurement outputs a vector of this exact size
        quantum_output_size=20,      
        hidden_feature=16,
        hidden_readout=16,
        quantum_type="builder"       # Triggers the new model class
        )
        config = self.config
        
        # Classical feature map: (x, t) -> 2 features
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size), 
        )
        
        try:
            import merlin as ML
            
            builder = ML.CircuitBuilder(n_modes=4)
            for _ in range(config.feature_size // 2):
                builder.add_entangling_layer(trainable=True, model="mzi")
                builder.add_angle_encoding(modes=[0, 1]) # Allocates input slots 0, 1
            builder.add_entangling_layer(trainable=True, model="mzi")

            pcvl_circuit = builder.to_pcvl_circuit()
            num_modes = pcvl_circuit.m
            critical_path = max(pcvl_circuit.depths())
            
            print(f'Builder circuit has {num_modes} modes and critical path length {critical_path}.')

            self.quantum = ML.QuantumLayer(
                # The circuit expects 4 inputs, so multiply feature_size by the number of uploads (2)
                input_size=config.feature_size,
                builder=builder,
                input_state=StateVector.from_basic_state([1, 1, 1, 0]),
                measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock')
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e
        
        self.readout = nn.Sequential(
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),
        )
        # Precompute the index mapping for the partial trace (Subsystem A: modes 0,1)
        self._setup_partial_trace(photons=3, modes=4, subsys_A_size=2)
        
    def _setup_partial_trace(self, photons: int, modes: int, subsys_A_size: int):
        """Prepares the index map for tracing out Subsystem B."""
        states = generate_fock_basis(photons, modes)
        
        # Split each 4-mode state into A (modes 0,1) and B (modes 2,3)
        A_states = [s[:subsys_A_size] for s in states]
        B_states = [s[subsys_A_size:] for s in states]
        
        # Extract unique states for both subsystems while preserving order
        self.unique_A = list(dict.fromkeys(A_states))
        self.unique_B = list(dict.fromkeys(B_states))
        self.A_dim = len(self.unique_A)
        
        # Create a mapping: state_map[A_idx, B_idx] = flat_state_idx
        # If an (A, B) combination exceeds the photon limit, it remains -1
        state_map = torch.full((self.A_dim, len(self.unique_B)), -1, dtype=torch.long)
        
        for idx, (a, b) in enumerate(zip(A_states, B_states)):
            a_idx = self.unique_A.index(a)
            b_idx = self.unique_B.index(b)
            state_map[a_idx, b_idx] = idx
            
        # Register as a non-trainable buffer so it moves properly with model.to(device)
        self.register_buffer('state_map', state_map)

    def calculate_bipartite_entropy(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Von Neumann entanglement entropy of Subsystem A.
        amplitudes shape: (batch_size, 20) complex
        """
        batch_size = amplitudes.size(0)

        # 1. Initialize the reduced density matrix rho_A
        rho_A = torch.zeros(
            (batch_size, self.A_dim, self.A_dim), 
            dtype=amplitudes.dtype, 
            device=amplitudes.device
        )
        
        # 2. Perform the Partial Trace over Subsystem B
        # rho_A[i,j] = sum_k (c_{i,k} * c_{j,k}^*)
        for i in range(self.A_dim):
            for j in range(self.A_dim):
                # Find indices in subsystem B that are valid for BOTH A_i and A_j
                valid_b = (self.state_map[i] != -1) & (self.state_map[j] != -1)
                idx_ik = self.state_map[i][valid_b]
                idx_jk = self.state_map[j][valid_b]
                
                if len(idx_ik) > 0:
                    rho_A[:, i, j] = torch.sum(
                        amplitudes[:, idx_ik] * torch.conj(amplitudes[:, idx_jk]), 
                        dim=1
                    )
                    
        # 3. Calculate Eigenvalues
        # Using eigvalsh because rho_A is a Hermitian matrix
        eigvals = torch.linalg.eigvalsh(rho_A)
        
        # 4. Calculate Von Neumann Entropy: S = -sum(lambda * ln(lambda))
        # Clamp to avoid log(0) and small negative eigenvalues from float imprecision
        eigvals = torch.clamp(eigvals, min=1e-12, max=1.0)
        entropy = -torch.sum(eigvals * torch.log(eigvals), dim=-1)
        
        return entropy
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        
        # Duplicate the features for data re-uploading
        z_reuploaded = torch.cat([z, z], dim=1)

        # 1. Quantum Pass -> Returns complex amplitudes (batch, 20)
        q_amplitudes = self.quantum(z_reuploaded)
       
        # 2. Derive Entanglement Entropy
        entropy = self.calculate_bipartite_entropy(q_amplitudes)
        
        # 3. Derive probabilities for classical readout: P(x) = |c|^2 -> (batch, 20) real
        q_probs = torch.abs(q_amplitudes) ** 2

        # 4. Classical Readout Pipeline
        out = self.readout(q_probs)
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
            
        return u, ux_hat, entropy.mean()
    

class FiniteDifferenceSolver(nn.Module):
    """
    Classical finite difference solver for the heat equation.
    Solves: du/dt = alpha * d²u/dx²
    
    Uses forward-time central-space (FTCS) explicit scheme.
    No trainable parameters - just solves the PDE directly.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Try to get alpha from config.pde if available, otherwise use default
        if hasattr(config, 'pde') and hasattr(config.pde, 'alpha'):
            self.alpha = config.pde.alpha
        else:
            # Default alpha for heat equation
            self.alpha = 0.1


        self.nx = 128  # spatial grid points
        self.nt = 100  # temporal grid points
        
        # Domain
        self.x_min, self.x_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0
        
        # Grid
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dt = (self.t_max - self.t_min) / (self.nt - 1)
        
        # Stability check for FTCS scheme
        r = self.alpha * self.dt / (self.dx ** 2)
        if r > 0.5:
            print(f"Warning: FTCS stability condition violated (r={r:.3f} > 0.5)")
            print(f"Adjusting dt to ensure stability...")
            self.dt = 0.45 * (self.dx ** 2) / self.alpha
            r = self.alpha * self.dt / (self.dx ** 2)
            print(f"New r={r:.3f}")
        
        self.r = r
        print(f"FiniteDifferenceSolver initialized: alpha={self.alpha}, r={self.r:.4f}")
        
        # Precompute solution on grid
        self._compute_solution()
        
        # Register as buffer (not trainable)
        self.register_buffer('solution_grid', torch.tensor(self.u, dtype=torch.float32))
        self.register_buffer('x_grid', torch.linspace(0, 1, self.nx, dtype=torch.float32))
        self.register_buffer('t_grid', torch.linspace(0, 1, self.nt, dtype=torch.float32))
    
    def _compute_solution(self):
        """Compute solution using FTCS finite difference scheme."""
        import numpy as np
        
        # Initialize solution
        self.u = np.zeros((self.nt, self.nx))
        
        # Initial condition: u(x, 0) = x(1-x)
        x = np.linspace(self.x_min, self.x_max, self.nx)
        self.u[0, :] = np.sin(np.pi * x)
        
        # Boundary conditions: u(0, t) = u(1, t) = 0
        # FTCS scheme
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                self.u[n+1, i] = (
                    self.r * self.u[n, i+1] +
                    (1.0 - 2.0 * self.r) * self.u[n, i] +
                    self.r * self.u[n, i-1]
                )
            # Enforce boundary conditions
            self.u[n+1, 0] = 0.0
            self.u[n+1, -1] = 0.0
        
        print(f"Solution computed: min={self.u.min():.6f}, max={self.u.max():.6f}")
    
    def _interpolate_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate solution at arbitrary (x, t) points using bilinear interpolation.
        
        Args:
            x: x coordinates, shape (batch,)
            t: t coordinates, shape (batch,)
        
        Returns:
            u: interpolated solution values, shape (batch,)
        """
        import torch.nn.functional as F
        
        # Normalize to grid indices
        x_idx = (x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1)
        t_idx = (t - self.t_min) / (self.t_max - self.t_min) * (self.nt - 1)
        
        # Clamp to valid range
        x_idx = torch.clamp(x_idx, 0, self.nx - 1)
        t_idx = torch.clamp(t_idx, 0, self.nt - 1)
        
        # Get surrounding grid points
        x_lo = torch.floor(x_idx).long()
        x_hi = torch.clamp(x_lo + 1, max=self.nx - 1)
        t_lo = torch.floor(t_idx).long()
        t_hi = torch.clamp(t_lo + 1, max=self.nt - 1)
        
        # Interpolation weights
        wx = x_idx - x_lo.float()
        wt = t_idx - t_lo.float()
        
        # Bilinear interpolation
        u_00 = self.solution_grid[t_lo, x_lo]
        u_01 = self.solution_grid[t_lo, x_hi]
        u_10 = self.solution_grid[t_hi, x_lo]
        u_11 = self.solution_grid[t_hi, x_hi]
        
        u = (1-wt) * (1-wx) * u_00 + \
            (1-wt) * wx * u_01 + \
            wt * (1-wx) * u_10 + \
            wt * wx * u_11
        
        return u
    
    def _compute_derivative(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute du/dx using finite differences on the solution grid.
        
        Args:
            x: x coordinates, shape (batch,)
            t: t coordinates, shape (batch,)
        
        Returns:
            ux: x-derivative, shape (batch,)
        """
        # Use central differences
        eps = 1e-3
        x_plus = torch.clamp(x + eps, self.x_min, self.x_max)
        x_minus = torch.clamp(x - eps, self.x_min, self.x_max)
        
        u_plus = self._interpolate_solution(x_plus, t)
        u_minus = self._interpolate_solution(x_minus, t)
        
        ux = (u_plus - u_minus) / (2 * eps)
        
        return ux
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - interpolate solution and its derivative.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux: x-derivative, shape (batch, 1)
        """
        x = xt[:, 0]
        t = xt[:, 1]
        
        # Interpolate solution
        u = self._interpolate_solution(x, t).unsqueeze(-1)
        
        # Compute derivative
        ux = self._compute_derivative(x, t).unsqueeze(-1)
        
        return u, ux
    
    def train(self, mode: bool = True):
        """Override train mode - this solver has no trainable parameters."""
        return self
    
    def eval(self):
        """Override eval mode - this solver has no trainable parameters."""
        return self
    

class ReuploadClassicalHeatQPINN(nn.Module):
    """
    Classical baseline mirroring the re-uploading depth experiment.

    Uses the same ModelConfig.quantum_depth field as ReuploadMerlinHeatQPINN,
    but replaces each additional quantum re-upload block with one trainable
    classical layer. With quantum_depth=1, this matches ClassicalHeatQPINN.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.depth = getattr(config, "quantum_depth", 1)
        if self.depth < 1:
            raise ValueError(f"quantum_depth must be at least 1, got {self.depth}")
        
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size),
            nn.Tanh(),
        )
        
        self.quantum_width_projection = nn.Sequential(
            nn.Linear(config.feature_size, config.quantum_output_size),
            nn.Tanh(),
        )
        
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.quantum_output_size, config.quantum_output_size),
                nn.Tanh(),
            )
            for _ in range(self.depth - 1)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        h = self.feature_map(xt)
        h = self.quantum_width_projection(h)
        
        for layer in self.extra_layers:
            h = layer(h)
        
        out = self.readout(h)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat


class ReuploadMerlinHeatQPINN(nn.Module):
    """
    QPINN with one quantum circuit using sequential data re-uploading blocks.

    Circuit layout:
        U_0(W_0) -> U_enc(x,t) -> U_1(W_1) -> U_enc(x,t) -> U_2(W_2) -> ...

    With quantum_depth=1 this matches MerLin's QuantumLayer.simple layout:
        trainable -> encoding -> trainable

    The circuit is measured once at the end, then a classical readout predicts
    [q_u, ux_hat].
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.quantum_depth = getattr(config, "quantum_depth", 1)
        if self.quantum_depth < 1:
            raise ValueError(f"quantum_depth must be at least 1, got {self.quantum_depth}")
        
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size),
        )
        
        try:
            import merlin as ML
            
            n_modes = config.feature_size + 1
            input_state = [1 if i % 2 == 0 else 0 for i in range(n_modes)]
            
            builder = ML.CircuitBuilder(n_modes=n_modes)
            builder.add_entangling_layer(trainable=True, name="LI_simple")
            for block_idx in range(self.quantum_depth):
                builder.add_angle_encoding(
                    modes=list(range(config.feature_size)),
                    name="input",
                    subset_combinations=False,
                )
                builder.add_entangling_layer(
                    trainable=True,
                    model="mzi",
                    name="RI_simple" if block_idx == 0 else f"reupload_block_{block_idx + 1}",
                )
            
            self.quantum = ML.QuantumLayer(
                input_size=config.feature_size * self.quantum_depth,
                builder=builder,
                input_state=input_state,
                n_photons=sum(input_state),
                measurement_strategy=ML.MeasurementStrategy.probs(
                    computation_space=ML.ComputationSpace.UNBUNCHED
                ),
            )
            
            if self.quantum.output_size != config.quantum_output_size:
                self.quantum_postprocess = ML.ModGrouping(
                    self.quantum.output_size,
                    config.quantum_output_size,
                )
            else:
                self.quantum_postprocess = nn.Identity()
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e
        
        self.readout = nn.Sequential(
            nn.Linear(config.quantum_output_size, config.hidden_readout),
            nn.Tanh(),
            nn.Linear(config.hidden_readout, 2),
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        z_reuploaded = torch.cat([z] * self.quantum_depth, dim=1)
        
        q = self.quantum(z_reuploaded)
        q = self.quantum_postprocess(q)
        out = self.readout(q)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
        if self.config.use_hard_bc:
            u = x * (1.0 - x) * q_u
        else:
            u = q_u
        
        return u, ux_hat


def create_model(config: ModelConfig, model_type: str = "qpinn") -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        config: ModelConfig object
        model_type: "qpinn" (MerLin), "classical", "deep_classical", or "qpinn_frozen"
    
    Returns:
        Initialized model
    """
    if model_type == "qpinn":
        # Check the newly added quantum_type attribute
#        if getattr(config, 'quantum_type', 'simple') == "builder":
        return MerlinHeatQPINN(config)
    elif model_type == "qpinn_builder":
        return BuilderMerlinHeatQPINN(config)
    elif model_type == "qpinn_frozen":
        return MerlinHeatQPINN_freezeQMweights(config)
    elif model_type == "classical":
        return ClassicalHeatQPINN(config)
    elif model_type == "deep_classical":
        return DeepNetworkQPINN(config, depth=3)
    elif model_type == "fd_solver":
        return FiniteDifferenceSolver(config)
    elif model_type == "reupload_qpinn":
        return ReuploadMerlinHeatQPINN(config)
    elif model_type == "reupload_classical":
        return ReuploadClassicalHeatQPINN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    