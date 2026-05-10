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
    elif model_type == "reupload_qpinn":
        return ReuploadMerlinHeatQPINN(config)
    elif model_type == "reupload_classical":
        return ReuploadClassicalHeatQPINN(config)
    elif model_type == "classical":
        return ClassicalHeatQPINN(config)
    elif model_type == "deep_classical":
        return DeepNetworkQPINN(config, depth=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
