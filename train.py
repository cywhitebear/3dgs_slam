"""
3D Gaussian Splatting Training Script.

Initializes Gaussians from point cloud and trains with gsplat.
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
        self.gaussians.initialize_from_pointcloud(points, colors)
        
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
        means = self.gaussians.get_xyz()  # (N, 3)
        quats = self.gaussians.get_rotation()  # (N, 4)
        scales = self.gaussians.get_scaling()  # (N, 3)
        opacities = self.gaussians.get_opacity().squeeze(-1)  # (N,) - squeeze to 1D
        colors = self.gaussians.get_colors_dc()  # (N, 1, 3)
        
        # Prepare for gsplat
        viewmat = pose_c2w.unsqueeze(0)  # (1, 4, 4)
        K_batch = K.unsqueeze(0)  # (1, 3, 3)
        
        # Render using gsplat
        try:
            image, alpha, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,  # Now (N,) as expected by gsplat
                colors=colors.squeeze(1),  # (N, 3)
                viewmats=viewmat,
                Ks=K_batch,
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
        
        return total_loss.item()
    
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
        """Save trained Gaussians as PLY point cloud."""
        try:
            import open3d as o3d
        except ImportError:
            print("[Trainer] open3d not installed, skipping PLY save")
            return
        
        # Extract Gaussian parameters
        xyz = self.gaussians.get_xyz().detach().cpu().numpy()
        colors = self.gaussians.get_colors_dc().squeeze(1).detach().cpu().numpy()
        
        # Clamp colors to valid range
        colors = np.clip(colors, 0, 1)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save
        ply_path = self.output_dir / filename
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"[Trainer] Saved Gaussians to {ply_path}")


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
    }
    
    # Create trainer and run
    trainer = Trainer(args.data_dir, args.output_dir, config, force_cpu=args.cpu)
    trainer.train(args.epochs, args.batch_size)
    
    print("[Trainer] Training complete!")


if __name__ == "__main__":
    main()
