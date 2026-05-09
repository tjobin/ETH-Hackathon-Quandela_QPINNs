"""
Data sampling and loss computation utilities.
"""

import torch
import torch.nn as nn
from config import PDEConfig, DataConfig


class DataSampler:
    """Handles sampling of interior, initial, and boundary points."""
    
    def __init__(self, config: DataConfig, pde_config: PDEConfig, 
                 device: torch.device, dtype: torch.dtype):
        self.config = config
        self.pde = pde_config
        self.device = device
        self.dtype = dtype
    
    def sample_interior(self, n: int = None) -> torch.Tensor:
        """
        Sample interior points where PDE is enforced.
        
        Returns:
            xt: tensor of shape (n, 2) with columns [x, t], requires_grad=True
        """
        if n is None:
            n = self.config.n_interior
        
        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        
        # Scale to domain
        x = self.pde.x_min + x * (self.pde.x_max - self.pde.x_min)
        t = self.pde.t_min + t * (self.pde.t_max - self.pde.t_min)
        
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        return xt
    
    def sample_initial(self, n: int = None) -> torch.Tensor:
        """
        Sample initial condition points (t=0).
        
        Returns:
            xt: tensor of shape (n, 2) with columns [x, t]
        """
        if n is None:
            n = self.config.n_initial
        
        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        x = self.pde.x_min + x * (self.pde.x_max - self.pde.x_min)
        
        t = torch.full_like(x, self.pde.t_min)
        xt = torch.cat([x, t], dim=1)
        return xt
    
    def sample_boundary(self, n: int = None) -> torch.Tensor:
        """
        Sample boundary condition points (x=0 and x=1).
        
        Returns:
            xt: tensor of shape (n, 2) with columns [x, t]
        """
        if n is None:
            n = self.config.n_boundary
        
        t = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t = self.pde.t_min + t * (self.pde.t_max - self.pde.t_min)
        
        # Split between x=0 and x=1
        half = n // 2
        x0 = torch.full((half, 1), self.pde.x_min, device=self.device, dtype=self.dtype)
        x1 = torch.full((n - half, 1), self.pde.x_max, device=self.device, dtype=self.dtype)
        x = torch.cat([x0, x1], dim=0)
        
        # Concatenate [x, t] along dimension 1 (not 0!)
        t_boundary = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t_boundary = self.pde.t_min + t_boundary * (self.pde.t_max - self.pde.t_min)
        
        xt = torch.cat([x, t_boundary], dim=1)  # CHANGED: dim=1 instead of dim=0
        return xt


class PhysicsLoss:
    """Computes physics-informed loss terms."""
    
    def __init__(self, pde_config: PDEConfig):
        self.pde = pde_config
        self.mse = nn.MSELoss()
    
    @staticmethod
    def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient dy/dx using automatic differentiation.
        
        Args:
            y: output tensor
            x: input tensor with requires_grad=True
        
        Returns:
            grad: gradient tensor of same shape as x
        """
        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
    
    def pde_residual(self, model, xt_interior: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: du/dt - alpha * d²u/dx²
        
        Heat equation: du/dt = alpha * d²u/dx²
        We use: du/dt - alpha * d(ux_hat)/dx = 0
        where ux_hat is the network's approximation of du/dx.
        
        Args:
            model: neural network model
            xt_interior: interior points with requires_grad=True
        
        Returns:
            residual: PDE residual tensor
        """
        u, ux_hat, _ = model(xt_interior)
        
        # Compute time and spatial derivatives
        grad_u = self.gradient(u, xt_interior)
        u_t = grad_u[:, 1:2]
        
        grad_ux_hat = self.gradient(ux_hat, xt_interior)
        ux_hat_x = grad_ux_hat[:, 0:1]
        
        # PDE residual
        residual = u_t - self.pde.alpha * ux_hat_x
        return residual
    
    def consistency_residual(self, model, xt_interior: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency residual: du/dx - ux_hat
        
        Ensures that the network's spatial derivative approximation matches
        the actual spatial derivative of the solution.
        
        Args:
            model: neural network model
            xt_interior: interior points with requires_grad=True
        
        Returns:
            residual: consistency residual tensor
        """
        u, ux_hat, _ = model(xt_interior)
        
        grad_u = self.gradient(u, xt_interior)
        u_x = grad_u[:, 0:1]
        
        residual = u_x - ux_hat
        return residual
    
    def entropy(self, model, xt_interior: torch.Tensor) -> torch.Tensor:
        """
        Compute entanglement entropy from the quantum layer's output.
        
        Args:
            model: neural network model
            xt_interior: interior points with requires_grad=True
        
        Returns:
            entropy: scalar tensor representing the average bipartite entanglement entropy
        """
        _, _, entanglement_entropy = model(xt_interior)
        return entanglement_entropy.mean()
    
    def initial_condition_loss(self, model, xt_initial: torch.Tensor,
                               exact_fn) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            model: neural network model
            xt_initial: initial condition points
            exact_fn: function u_exact(x, t) that evaluates exact solution
        
        Returns:
            loss: MSE loss at initial time
        """
        u_pred, _, _ = model(xt_initial)
        x = xt_initial[:, 0:1]
        t = xt_initial[:, 1:2]
        u_exact = exact_fn(x, t)
        
        return self.mse(u_pred, u_exact)
    
    def boundary_condition_loss(self, model, xt_boundary: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.
        Note: if hard BCs are enforced in the model, this should be near zero.
        
        Args:
            model: neural network model
            xt_boundary: boundary condition points
        
        Returns:
            loss: MSE loss at boundaries (should be near zero)
        """
        u_pred, _, _ = model(xt_boundary)
        return self.mse(u_pred, torch.zeros_like(u_pred))
    
    def total_loss(self, model, xt_f: torch.Tensor, xt_i: torch.Tensor,
                   xt_b: torch.Tensor, exact_fn, 
                   lambda_f: float = 1.0, lambda_c: float = 0.1,
                   lambda_i: float = 10.0, lambda_b: float = 1.0) -> tuple:
        """
        Compute total weighted loss.
        
        Args:
            model: neural network model
            xt_f: interior points
            xt_i: initial condition points
            xt_b: boundary condition points
            exact_fn: exact solution function
            lambda_f: weight for PDE residual
            lambda_c: weight for consistency residual
            lambda_i: weight for initial condition
            lambda_b: weight for boundary condition
        
        Returns:
            (total_loss, loss_dict): tuple of total loss and dictionary with component losses
        """
        # PDE and consistency losses
        r_f = self.pde_residual(model, xt_f)
        r_c = self.consistency_residual(model, xt_f)
        
        loss_f = self.mse(r_f, torch.zeros_like(r_f))
        loss_c = self.mse(r_c, torch.zeros_like(r_c))
        
        # Initial condition loss
        loss_i = self.initial_condition_loss(model, xt_i, exact_fn)
        
        # Boundary condition loss
        loss_b = self.boundary_condition_loss(model, xt_b)
        
        # Total loss
        total_loss = (lambda_f * loss_f + lambda_c * loss_c + 
                     lambda_i * loss_i + lambda_b * loss_b)
        
        batch_entropy = self.entropy(model, xt_f)
        
        loss_dict = {
            "total": total_loss.item(),
            "pde": loss_f.item(),
            "consistency": loss_c.item(),
            "initial": loss_i.item(),
            "boundary": loss_b.item(),
            "entropy": batch_entropy.item()
        }
        
        return total_loss, loss_dict
