"""
Gaussian Model definition for 3D Gaussian Splatting.
"""

import numpy as np
import torch
import torch.nn as nn


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
    
    def initialize_from_pointcloud(self, positions, colors):
        """
        Initialize Gaussians from point cloud.
        
        Args:
            positions: (N, 3) numpy array or torch tensor of 3D positions
            colors: (N, 3) numpy array or torch tensor of RGB colors in [0, 1]
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
        
        with torch.no_grad():
            # Positions
            self._xyz.data = positions_t
            
            # Colors (convert to SH representation, storing DC component)
            # DC component is the mean color
            self._features_dc.data = colors_t.unsqueeze(1)
            
            # Initialize scales (small fixed size)
            # Use log scale: scale = exp(log_scale)
            init_scale = 0.1
            self._scaling.data = torch.ones((len(positions_t), 3), dtype=torch.float32, 
                                            device=positions_t.device) * np.log(init_scale)
            
            # Initialize rotation (identity quaternion: [0, 0, 0, 1])
            self._rotation.data = torch.zeros((len(positions_t), 4), dtype=torch.float32,
                                             device=positions_t.device)
            self._rotation.data[:, 3] = 1.0
            
            # Initialize opacity (logit space, 0.5 -> logit 0)
            self._opacity.data = torch.zeros((len(positions_t), 1), dtype=torch.float32,
                                            device=positions_t.device)
    
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
    
    def __repr__(self):
        return f"GaussianModel(num_gaussians={self.num_gaussians})"
