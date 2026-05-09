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
        
        xt = torch.cat([x, t_boundary], dim=1)
        return xt


class PhysicsLoss:
    """Computes physics-informed loss terms for heat equation."""
    
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
        u, ux_hat, *_ = model(xt_interior)
        
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
        u, ux_hat, *_ = model(xt_interior)
        
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
        _, _, *entanglement_entropy = model(xt_interior)
        if not entanglement_entropy:
            return torch.tensor(0.0, device=xt_interior.device, dtype=xt_interior.dtype)
        return torch.tensor(entanglement_entropy).mean()
    
    def initial_condition_loss(self, model, xt_initial: torch.Tensor,
                               exact_fn=None) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Uses the PDE's initial_condition() method to enforce u(x, 0) = f(x).
        
        Args:
            model: neural network model
            xt_initial: initial condition points with t=0
            exact_fn: (optional) function u_exact(x, t) that evaluates exact solution
        
        Returns:
            loss: MSE loss at initial time
        """
        u_pred, _, *_ = model(xt_initial)
        x = xt_initial[:, 0:1]
        
        # Use the PDE's initial_condition method
        u_ic = self.pde.initial_condition(x)
        
        return self.mse(u_pred, u_ic)
    
    def boundary_condition_loss(self, model, xt_boundary: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Uses the PDE's boundary_condition_left() and boundary_condition_right() methods.
        
        Args:
            model: neural network model
            xt_boundary: boundary condition points (x=0 and x=1)
        
        Returns:
            loss: MSE loss at boundaries
        """
        u_pred, _, *_ = model(xt_boundary)
        
        x = xt_boundary[:, 0:1]
        t = xt_boundary[:, 1:2]
        
        # Split into left (x=0) and right (x=1) boundaries
        # Left boundary: x ≈ 0
        mask_left = x < 0.5
        if mask_left.any():
            u_bc_left = self.pde.boundary_condition_left(t[mask_left])
            loss_left = self.mse(u_pred[mask_left], u_bc_left)
        else:
            loss_left = torch.tensor(0.0, device=u_pred.device, dtype=u_pred.dtype)
        
        # Right boundary: x ≈ 1
        mask_right = x >= 0.5
        if mask_right.any():
            u_bc_right = self.pde.boundary_condition_right(t[mask_right])
            loss_right = self.mse(u_pred[mask_right], u_bc_right)
        else:
            loss_right = torch.tensor(0.0, device=u_pred.device, dtype=u_pred.dtype)
        
        return loss_left + loss_right
    
    def total_loss(self, model, xt_f: torch.Tensor, xt_i: torch.Tensor,
                   xt_b: torch.Tensor, exact_fn=None, 
                   lambda_f: float = 1.0, lambda_c: float = 0.1,
                   lambda_i: float = 10.0, lambda_b: float = 1.0) -> tuple:
        """
        Compute total weighted loss.
        
        Args:
            model: neural network model
            xt_f: interior points
            xt_i: initial condition points
            xt_b: boundary condition points
            exact_fn: (optional, deprecated) exact solution function
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
        
        # Initial condition loss (uses PDE's initial_condition method)
        loss_i = self.initial_condition_loss(model, xt_i, exact_fn)
        
        # Boundary condition loss (uses PDE's BC methods)
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


class BurgersPhysicsLoss(PhysicsLoss):
    """Loss for Burgers equation: du/dt + u*du/dx = nu*d²u/dx²"""
    
    def pde_residual(self, model, xt_interior: torch.Tensor) -> torch.Tensor:
        """
        Compute Burgers PDE residual.
        du/dt + u*du/dx - nu*d²u/dx² = 0
        """
        u, ux_hat, *_ = model(xt_interior)
        
        # Get derivatives
        grad_u = self.gradient(u, xt_interior)
        u_t = grad_u[:, 1:2]
        u_x = grad_u[:, 0:1]
        
        grad_ux = self.gradient(ux_hat, xt_interior)
        ux_xx = grad_ux[:, 0:1]
        
        # Burgers equation: du/dt + u*du/dx - nu*d²u/dx² = 0
        nu = getattr(self.pde, 'nu', 0.01)
        residual = u_t + u * u_x - nu * ux_xx
        return residual


class WavePhysicsLoss(PhysicsLoss):
    """Loss for Wave equation: d²u/dt² = c² * d²u/dx²"""
    
    def pde_residual(self, model, xt_interior: torch.Tensor) -> torch.Tensor:
        """
        Compute Wave PDE residual.
        d²u/dt² - c² * d²u/dx² = 0
        """
        u, ux_hat, *_ = model(xt_interior)
        
        # Get first derivatives
        grad_u = self.gradient(u, xt_interior)
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]
        
        # Get second derivatives
        grad_ut = self.gradient(u_t, xt_interior)
        u_tt = grad_ut[:, 1:2]
        
        grad_ux = self.gradient(ux_hat, xt_interior)
        u_xx = grad_ux[:, 0:1]
        
        # Wave equation: d²u/dt² - c² * d²u/dx² = 0
        c = getattr(self.pde, 'c', 1.0)
        residual = u_tt - (c ** 2) * u_xx
        return residual