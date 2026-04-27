"""
3D Gaussian Splatting Training Script.

Complete 3DGS training pipeline:
1. INITIALIZATION: All point cloud points → Learnable 3D Gaussians
   - Position: from point cloud coordinates
   - Color: from point cloud RGB
   - Scale: fixed initial size (0.1), learnable
   - Rotation: identity quaternion, learnable
   - Opacity: 0.5 (logit space), learnable

2. RENDERING (per batch):
   - Project 3D Gaussians to 2D using camera pose (world-to-camera) and intrinsics
   - Rasterize 2D Gaussians to image via gsplat differentiable splatting
   - Output: differentiable image with gradient flow

3. LOSS & OPTIMIZATION:
   - L1 loss between rendered image and ground truth
   - Backward pass: compute gradients for all Gaussian parameters
   - Adam optimizer: update position, color, scale, rotation, opacity
   - Learning rate decay: exponential decay per epoch

4. DENSITY CONTROL:
   - Prune low-opacity Gaussians (< 0.005) that don't contribute
   - Reduces parameters while maintaining quality

5. OUTPUT:
   - Checkpoints: Full model state with all parameters
   - PLY file: Point cloud with positions, colors, and metadata
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.utils.data as data_utils

import cv2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from scene.gaussian_model import GaussianModel
from utils.general_utils import mkdir_p, inverse_sigmoid


class Trainer:
    """Handles 3DGS training."""
    
    def __init__(self, data_dir, output_dir, config, force_cpu=False):
        """
        Initialize trainer.
        
        Args:
            data_dir: Path to dataset
            output_dir: Output directory for logs and checkpoints
            config: Config dictionary
            force_cpu: Force CPU even if CUDA available
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))
        
        self.config = config
        
        # Device selection
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Trainer] Using device: {self.device}")
        
        # Load dataset
        print("[Trainer] Loading dataset...")
        self.dataset = Dataset(data_dir)
        
        # Initialize Gaussians from point cloud
        print("[Trainer] Initializing Gaussians from point cloud...")
        self.gaussians = GaussianModel(len(self.dataset.points_map))
        self.gaussians = self.gaussians.to(self.device)
        
        # Initialize from point cloud
        points, colors = self.dataset.get_pointcloud()
        points = points.to(self.device)
        colors = colors.to(self.device)
        
        # Initialize Gaussians with configurable scale and opacity
        # Smaller init_scale = finer details, but takes longer to train
        # Suggested values: 0.01 (very small), 0.05 (small), 0.1 (medium)
        # init_opacity: 0-1 range for initial opacity (0.1=barely visible, 0.5=half, 0.9=mostly opaque)
        init_scale = self.config.get('init_scale', 0.01)
        init_opacity = self.config.get('init_opacity', 0.5)
        self.gaussians.initialize_from_pointcloud(points, colors, init_scale=init_scale, init_opacity=init_opacity)
        
        print(f"[Trainer] Initialized {self.gaussians.num_gaussians} Gaussians")
        
        # Setup optimizer
        self.optimizer = Adam([
            {'params': [self.gaussians._xyz], 'lr': config['lr_xyz']},
            {'params': [self.gaussians._features_dc], 'lr': config['lr_color']},
            {'params': [self.gaussians._features_rest], 'lr': config['lr_color']},
            {'params': [self.gaussians._opacity], 'lr': config['lr_opacity']},
            {'params': [self.gaussians._scaling], 'lr': config['lr_scaling']},
            {'params': [self.gaussians._rotation], 'lr': config['lr_rotation']},
        ], lr=0.0)
        
        # Setup scheduler
        self.scheduler = ExponentialLR(self.optimizer, gamma=config['lr_decay'])
        
        # Tensorboard
        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        
        # Get poses and intrinsics
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)  # (N, 4, 4)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)  # (3, 3)
        
        print(f"[Trainer] Loaded {len(self.poses_c2w)} camera poses")
        
        # Frame indices for training
        self.train_frame_indices = np.arange(len(self.dataset))
        
        self.iteration = 0
    
    def render_image(self, pose_c2w, K, image_width, image_height):
        """
        Render image using gsplat rasterization.
        
        Args:
            pose_c2w: (4, 4) camera to world matrix
            K: (3, 3) intrinsic matrix
            image_width: Image width
            image_height: Image height
        
        Returns:
            rendered_image: (3, H, W) RGB image
        """
        from gsplat.rendering import rasterization
        
        # Get Gaussian parameters
        means = self.gaussians.get_xyz()  # (N, 3) - world space
        quats = self.gaussians.get_rotation()  # (N, 4) - normalized quaternions
        scales = self.gaussians.get_scaling()  # (N, 3) - exponentiated scales
        opacities = self.gaussians.get_opacity().squeeze(-1)  # (N,) - sigmoid applied
        colors = self.gaussians.get_colors_dc()  # (N, 1, 3) - DC SH component
        
        # Convert camera pose: c2w -> w2c (world to camera)
        # gsplat expects viewmats as world-to-camera transformation
        pose_w2c = torch.linalg.inv(pose_c2w)
        
        # Prepare for gsplat (batch size = 1)
        viewmat = pose_w2c.unsqueeze(0)  # (1, 4, 4)
        K_batch = K.unsqueeze(0)  # (1, 3, 3)
        
        # Render using gsplat
        try:
            image, alpha, info = rasterization(
                means=means,                    # (N, 3) in world space
                quats=quats,                    # (N, 4) normalized quaternions
                scales=scales,                  # (N, 3) positive scales
                opacities=opacities,            # (N,) in [0, 1]
                colors=colors.squeeze(1),       # (N, 3) RGB values
                viewmats=viewmat,               # (1, 4, 4) world-to-camera
                Ks=K_batch,                     # (1, 3, 3) intrinsics
                width=image_width,
                height=image_height,
                near_plane=0.01,
                far_plane=100.0,
                radius_clip=0.0,
            )
            # image shape: (1, H, W, 3)
            image = image.squeeze(0).permute(2, 0, 1)  # (3, H, W)
            
            return image
        except Exception as e:
            print(f"[WARNING] Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.ones((3, image_height, image_width), device=self.device)
    
    def train_step(self, frame_indices):
        """
        Run one training step.
        
        Args:
            frame_indices: Indices of frames to use
        
        Returns:
            loss_value: Loss value
        """
        self.optimizer.zero_grad()
        
        total_loss = 0
        num_frames = len(frame_indices)
        
        for frame_idx in frame_indices:
            # Get target image
            target_image = self.dataset.get_image(frame_idx)  # (3, H, W)
            target_image = target_image.to(self.device)
            
            # Get camera pose
            pose_c2w = self.poses_c2w[frame_idx]
            
            # Render image
            rendered_image = self.render_image(
                pose_c2w, self.K,
                self.dataset.image_width,
                self.dataset.image_height
            )
            
            # Compute loss
            loss = F.l1_loss(rendered_image, target_image)
            total_loss += loss
        
        total_loss = total_loss / num_frames
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Density control: prune low opacity Gaussians
        self.prune_low_opacity_gaussians(opacity_threshold=0.005)
        
        return total_loss.item()
    
    def prune_low_opacity_gaussians(self, opacity_threshold=0.005):
        """
        Remove Gaussians with opacity below threshold.
        These Gaussians contribute negligibly to the final image.
        
        Args:
            opacity_threshold: Opacity threshold for pruning
        """
        with torch.no_grad():
            opacities = self.gaussians.get_opacity().squeeze(-1)  # (N,)
            
            # Find Gaussians to keep
            mask = opacities > opacity_threshold
            num_to_remove = (~mask).sum().item()
            
            if num_to_remove > 0:
                # Keep only high-opacity Gaussians
                self.gaussians._xyz.data = self.gaussians._xyz.data[mask]
                self.gaussians._features_dc.data = self.gaussians._features_dc.data[mask]
                self.gaussians._features_rest.data = self.gaussians._features_rest.data[mask]
                self.gaussians._scaling.data = self.gaussians._scaling.data[mask]
                self.gaussians._rotation.data = self.gaussians._rotation.data[mask]
                self.gaussians._opacity.data = self.gaussians._opacity.data[mask]
                
                # Update optimizer to remove pruned parameters
                # Reset optimizer state for new parameter set
                self.optimizer = Adam([
                    {'params': [self.gaussians._xyz], 'lr': self.config['lr_xyz']},
                    {'params': [self.gaussians._features_dc], 'lr': self.config['lr_color']},
                    {'params': [self.gaussians._features_rest], 'lr': self.config['lr_color']},
                    {'params': [self.gaussians._opacity], 'lr': self.config['lr_opacity']},
                    {'params': [self.gaussians._scaling], 'lr': self.config['lr_scaling']},
                    {'params': [self.gaussians._rotation], 'lr': self.config['lr_rotation']},
                ], lr=0.0)
                
                self.gaussians.num_gaussians = len(self.gaussians._xyz)
                
                if self.iteration % 100 == 0:
                    print(f"[Density Control] Pruned {num_to_remove} low-opacity Gaussians at iteration {self.iteration}")

    
    def train(self, num_epochs, batch_size=4):
        """
        Run training loop.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"\n[Trainer] Starting training for {num_epochs} epochs...")
        print(f"[Trainer] Dataset: {len(self.dataset)} frames")
        print(f"[Trainer] Batch size: {batch_size}\n")
        
        num_batches = (len(self.train_frame_indices) + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            # Shuffle frame order
            indices = np.random.permutation(self.train_frame_indices)
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            
            for batch_idx in pbar:
                # Get batch of frame indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Training step
                loss = self.train_step(batch_indices)
                epoch_loss += loss
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                
                # Tensorboard logging
                self.tb_writer.add_scalar('Loss/train', loss, self.iteration)
                self.iteration += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch + 1)
            
            # Learning rate decay
            self.scheduler.step()
    
    def save_checkpoint(self, epoch):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'iteration': self.iteration,
            'model_state_dict': self.gaussians.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"[Trainer] Saved checkpoint to {checkpoint_path}")
    
    def save_gaussians_as_ply(self, filename="gaussians.ply"):
        """
        Save trained Gaussians as binary PLY in standard 3DGS format.
        Compatible with 3DGS viewers (e.g., https://www.3dgsviewers.com/)
        
        IMPORTANT: Parameter spaces in the PLY file:
        - xyz: World coordinates (direct values)
        - scales: LOG-SPACE (stored as _scaling = log(actual_scale))
        - opacities: LOGIT-SPACE (stored as _opacity = log(opacity/(1-opacity)))
        - rotations: Normalized quaternions
        - colors_dc: Direct RGB values [0, 1]
        - colors_rest: SH coefficients (can be negative)
        
        These are the LEARNED parameters, not the transformed ones.
        The viewer will apply exp() and sigmoid() to recover actual values.
        """
        import struct
        
        # Extract Gaussian parameters
        xyz = self.gaussians.get_xyz().detach().cpu().numpy()                    # (N, 3)
        colors_dc = self.gaussians.get_colors_dc().squeeze(1).detach().cpu().numpy()   # (N, 3)
        colors_rest = self.gaussians.get_colors_rest().detach().cpu().numpy()    # (N, 15, 3)
        scales = self.gaussians._scaling.detach().cpu().numpy()                  # (N, 3) - log-scales!
        opacities = self.gaussians._opacity.squeeze(-1).detach().cpu().numpy()   # (N,) - logit-space opacity!
        rotations = self.gaussians.get_rotation().detach().cpu().numpy()         # (N, 4) - normalized quaternions
        
        num_points = len(xyz)
        
        # NOTE: DC and rest are in SPHERICAL HARMONICS space, NOT RGB!
        # They can be negative and outside [0, 1] range. DO NOT CLIP!
        # The viewer will apply SH2RGB transformation to recover proper colors.
        # Only positions and opacities are bounded:
        # - positions: unlimited (world coordinates)
        # - opacities: stored in logit space (unlimited)
        
        # Compute normals from rotation (use z-axis of rotation matrix)
        # For now, use simplified normals based on rotation
        normals = np.zeros((num_points, 3), dtype=np.float32)
        # Simple approach: use rotation to compute a normal direction
        for i in range(num_points):
            # Extract rotation matrix from quaternion
            q = rotations[i]  # [qx, qy, qz, qw]
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            
            # Compute z-axis of rotation matrix: 2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx^2 + qy^2)
            nz = np.array([
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2)
            ], dtype=np.float32)
            normals[i] = nz / (np.linalg.norm(nz) + 1e-8)
        
        # Flatten SH rest coefficients (15 components × 3 channels = 45 values)
        colors_rest_flat = colors_rest.reshape(num_points, -1)  # (N, 45)
        
        ply_path = self.output_dir / filename
        
        # Write binary PLY
        with open(ply_path, 'wb') as f:
            # Write header
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
            # Add f_rest properties (45 values: f_rest_0 to f_rest_44)
            for i in range(45):
                header += f"property float f_rest_{i}\n"
            
            header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            f.write(header.encode('utf-8'))
            
            # Write binary data
            for i in range(num_points):
                # Position
                f.write(struct.pack('fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))
                
                # Normal
                f.write(struct.pack('fff', normals[i, 0], normals[i, 1], normals[i, 2]))
                
                # DC color
                f.write(struct.pack('fff', colors_dc[i, 0], colors_dc[i, 1], colors_dc[i, 2]))
                
                # Rest of SH components (45 floats)
                for j in range(45):
                    f.write(struct.pack('f', colors_rest_flat[i, j]))
                
                # Opacity
                f.write(struct.pack('f', opacities[i]))
                
                # Scale
                f.write(struct.pack('fff', scales[i, 0], scales[i, 1], scales[i, 2]))
                
                # Rotation quaternion
                f.write(struct.pack('ffff', rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
        
        print(f"[Trainer] Saved Gaussians to {ply_path} in 3DGS format")
        print(f"[Trainer]   - Points: {num_points}")
        print(f"[Trainer]   - Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
        print(f"[Trainer]   - Scale range: [{scales.min():.3f}, {scales.max():.3f}]")
        print(f"[Trainer] This PLY is compatible with 3DGS viewers")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU even if CUDA available")
    
    args = parser.parse_args()
    
    # Default config
    config = {
        'lr_xyz': 0.00016,
        'lr_color': 0.0025,
        'lr_opacity': 0.05,
        'lr_scaling': 0.005,
        'lr_rotation': 0.001,
        'lr_decay': 0.9999,
        'checkpoint_interval': 10,
        'init_scale': 0.01,           # Initial Gaussian size in meters (0.01=1cm, 0.05=5cm, 0.1=10cm)
        'init_opacity': 0.5,          # Initial opacity (0-1, passed through sigmoid in logit space)
    }
    
    # Create trainer and run
    trainer = Trainer(args.data_dir, args.output_dir, config, force_cpu=args.cpu)
    trainer.train(args.epochs, args.batch_size)
    
    print("[Trainer] Training complete!")


if __name__ == "__main__":
    main()
