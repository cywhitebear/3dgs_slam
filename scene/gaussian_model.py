"""
Gaussian Model definition for 3D Gaussian Splatting.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Import SH utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sh_utils import RGB2SH


class GaussianModel(nn.Module):
    """
    Represents a set of 3D Gaussians with learnable parameters.
    """
    
    def __init__(self, num_points):
        """
        Initialize Gaussian model.
        
        Args:
            num_points: Number of Gaussian points
        """
        super().__init__()
        
        # Learnable parameters
        self._xyz = nn.Parameter(torch.zeros((num_points, 3), dtype=torch.float32))
        self._features_dc = nn.Parameter(torch.zeros((num_points, 1, 3), dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.zeros((num_points, 15, 3), dtype=torch.float32))
        
        # Log scale and rotation
        self._scaling = nn.Parameter(torch.zeros((num_points, 3), dtype=torch.float32))
        self._rotation = nn.Parameter(torch.zeros((num_points, 4), dtype=torch.float32))
        
        # Opacity (logit space)
        self._opacity = nn.Parameter(torch.zeros((num_points, 1), dtype=torch.float32))
        
        self.num_gaussians = num_points
    
    def initialize_from_pointcloud(self, positions, colors, init_scale=0.01, init_opacity=0.5):
        """
        Initialize Gaussians from point cloud.
        
        Args:
            positions: (N, 3) numpy array or torch tensor of 3D positions
            colors: (N, 3) numpy array or torch tensor of RGB colors in [0, 1]
            init_scale: Initial scale for Gaussians (smaller = finer detail)
            init_opacity: Initial opacity [0, 1] (will be converted to logit space internally)
        """
        # Convert to torch tensors if needed
        if isinstance(positions, torch.Tensor):
            positions_t = positions
        else:
            positions_t = torch.from_numpy(positions).float()
        
        if isinstance(colors, torch.Tensor):
            colors_t = colors
        else:
            colors_t = torch.from_numpy(colors).float()
        
        # Ensure colors are in [0, 1]
        colors_t = torch.clamp(colors_t, 0, 1)
        
        with torch.no_grad():
            # Positions
            self._xyz.data = positions_t
            
            # Colors (DC component in SH space)
            # CRITICAL: Apply RGB2SH transformation to convert RGB to SH DC space
            # This is required for correct color interpretation by viewers
            # DC = (RGB - 0.5) / 0.28209479 (SH basis normalization constant)
            colors_sh = RGB2SH(colors_t)  # Transform RGB → SH DC space
            self._features_dc.data = colors_sh.unsqueeze(1)
            
            # Initialize scales (small fixed size, learnable)
            # Use log scale: scale = exp(log_scale)
            # Smaller values = finer details
            # Try init_scale=0.01 for small, 0.05 for medium, 0.1 for large
            self._scaling.data = torch.ones((len(positions_t), 3), dtype=torch.float32, 
                                            device=positions_t.device) * np.log(init_scale)
            
            # Initialize rotation (identity quaternion: [0, 0, 0, 1])
            self._rotation.data = torch.zeros((len(positions_t), 4), dtype=torch.float32,
                                             device=positions_t.device)
            self._rotation.data[:, 3] = 1.0
            
            # Initialize opacity (logit space)
            # Convert desired opacity [0, 1] to logit space for optimization
            # logit(x) = log(x / (1-x))
            # Clamp to avoid log(0) or log(inf)
            init_opacity = np.clip(init_opacity, 0.001, 0.999)
            logit_opacity = np.log(init_opacity / (1 - init_opacity))
            self._opacity.data = torch.ones((len(positions_t), 1), dtype=torch.float32,
                                           device=positions_t.device) * logit_opacity
    
    def get_xyz(self):
        """Get position parameters."""
        return self._xyz
    
    def get_scaling(self):
        """Get scaling parameters (exponentiated)."""
        return torch.exp(self._scaling)
    
    def get_rotation(self):
        """Get rotation parameters (normalized quaternion)."""
        rot = self._rotation
        return rot / (torch.norm(rot, dim=1, keepdim=True) + 1e-8)
    
    def get_opacity(self):
        """Get opacity parameters (sigmoid applied)."""
        return torch.sigmoid(self._opacity)
    
    def get_colors_dc(self):
        """Get DC color component."""
        return self._features_dc
    
    def get_colors_rest(self):
        """Get higher-order SH color components."""
        return self._features_rest
    
    # Utility functions for parameter activation
    # These convert from learned parameter space to actual values
    scaling_activation = torch.exp
    scaling_inverse_activation = torch.log
    opacity_activation = torch.sigmoid
    rotation_activation = lambda x: nn.functional.normalize(x, dim=1)
    
    def replace_parameters(self, new_params):
        """
        Replace parameters with new tensors after densification (split/clone/prune).
        
        This method safely updates all nn.Parameter attributes with new tensors
        returned by strategy.refine(). It keeps them as learnable parameters
        so gradients flow correctly and they remain registered with the module.
        
        Args:
            new_params (dict): Dictionary with keys like 'means', 'scales', 'quats',
                             'opacities', 'features_dc', 'features_rest'
                             These tensors come from strategy.refine()
        
        Returns:
            None (updates in-place)
        """
        with torch.no_grad():
            # Map strategy parameter names to our internal names
            param_mapping = {
                'means': '_xyz',
                'scales': '_scaling',
                'quats': '_rotation',
                'opacities': '_opacity',
                'features_dc': '_features_dc',
                'features_rest': '_features_rest',
            }
            
            # Update each parameter, ensuring it remains an nn.Parameter
            for strategy_name, internal_name in param_mapping.items():
                if strategy_name in new_params:
                    new_tensor = new_params[strategy_name]
                    # Replace the parameter (wraps tensor as nn.Parameter automatically)
                    setattr(self, internal_name, nn.Parameter(new_tensor))
        
        # Update gaussian count to reflect new population
        self.num_gaussians = new_params['means'].shape[0]
    
    def __repr__(self):
        return f"GaussianModel(num_gaussians={self.num_gaussians})"
