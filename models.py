"""
Model definitions for QPINN.
Supports different quantum and classical architectures.
"""

import torch
import torch.nn as nn
from typing import Tuple
from config import ModelConfig


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
            nn.Linear(config.feature_size, config.hidden_readout),
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
            
            builder = ML.CircuitBuilder(n_modes=4)
            builder.add_entangling_layer(trainable=True, model="mzi")
            builder.add_angle_encoding(modes=[0, 1]) # Allocates input slots 0, 1
            builder.add_entangling_layer(trainable=True, model="mzi")
            builder.add_angle_encoding(modes=[0, 1]) # Allocates input slots 2, 3
            builder.add_entangling_layer(trainable=True, model="mzi")

            pcvl_circuit = builder.to_pcvl_circuit()
            num_modes = pcvl_circuit.m
            critical_path = max(pcvl_circuit.depths())
            
            print(f'Builder circuit has {num_modes} modes and critical path length {critical_path}.')

            self.quantum = ML.QuantumLayer(
                # The circuit expects 4 inputs, so multiply feature_size by the number of uploads (2)
                input_size=config.feature_size,
                builder=builder,
                input_state=[1, 1, 1, 0],
                measurement_strategy=ML.MeasurementStrategy.probs()
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
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        
        # Duplicate the features for data re-uploading
        # Maps (batch, 2) -> (batch, 4)
        z_reuploaded = torch.cat([z, z], dim=1)
        
        q = self.quantum(z_reuploaded)
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
        model_type: "qpinn" (MerLin), "classical", or "deep_classical"
    
    Returns:
        Initialized model
    """
    if model_type == "qpinn":
        # Check the newly added quantum_type attribute
        if getattr(config, 'quantum_type', 'simple') == "builder":
            return BuilderMerlinHeatQPINN(config)
        return MerlinHeatQPINN(config)
    elif model_type == "classical":
        return ClassicalHeatQPINN(config)
    elif model_type == "deep_classical":
        return DeepNetworkQPINN(config, depth=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
