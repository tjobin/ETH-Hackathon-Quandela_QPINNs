"""
Model definitions for QPINN.
Supports different quantum and classical architectures.
"""

import torch
import torch.nn as nn
import scipy
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
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Classical feature map: (x, t) -> 2 features
        self.feature_map = nn.Sequential(
            nn.Linear(2, config.hidden_feature),
            nn.Tanh(),
            nn.Linear(config.hidden_feature, config.feature_size), 
        )
        
        try:
            import merlin as ML
            
            # 1. Encode all features at the start
            builder = ML.CircuitBuilder(n_modes=config.feature_size)
            builder.add_angle_encoding(modes=None, name="x") # modes=None defaults to all modes

            # 2. Add multiple processing layers (depth)
            depth = config.quantum_depth # or config.quantum_depth
            for _ in range(depth):
                builder.add_entangling_layer(trainable=True, model="mzi")

            input_state = [1 for _ in range(config.n_photons)] + [0 for _ in range(config.feature_size - config.n_photons)]

            pcvl_circuit = builder.to_pcvl_circuit()
            num_modes = pcvl_circuit.m
            critical_path = max(pcvl_circuit.depths())
            
            print(f'Builder circuit has {num_modes} modes and critical path length {critical_path}.')

            self.quantum = ML.QuantumLayer(
                # The circuit expects 4 inputs, so multiply feature_size by the number of uploads (2)
                input_size=config.feature_size,
                builder=builder,
                input_state=StateVector.from_basic_state(input_state),
                measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock')
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e
        
        self.readout = nn.Sequential(
            nn.Linear(int(scipy.special.comb(config.feature_size+config.n_photons-1, config.n_photons)), config.hidden_readout),
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
        if getattr(config, 'quantum_type', 'simple') == "builder":
            return BuilderMerlinHeatQPINN(config)
        return MerlinHeatQPINN(config)
    elif model_type == "qpinn_frozen":
        return MerlinHeatQPINN_freezeQMweights(config)
    elif model_type == "classical":
        return ClassicalHeatQPINN(config)
    elif model_type == "deep_classical":
        return DeepNetworkQPINN(config, depth=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")