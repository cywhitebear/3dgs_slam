"""
3D Gaussian Splatting Trainer - Rewritten following gsplat official example.
Uses step_pre_backward / step_post_backward pattern for proper densification.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import gsplat
from gsplat import DefaultStrategy
from gsplat.rendering import rasterization

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene.dataset_readers import Dataset
from utils.general_utils import mkdir_p
from utils.sh_utils import RGB2SH


def create_gaussians_with_optimizers(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scale: float = 0.2,
    init_opacity: float = 0.5,
    sh_degree: int = 0,  # 0 means only DC component (RGB colors)
    means_lr: float = 1.6e-4,
    scale_lr: float = 5e-3,
    opacity_lr: float = 5e-2,
    quat_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    device: str = "cuda",
) -> tuple:
    """
    Initialize Gaussians from point cloud with fixed scale.
    Returns: (splats ParameterDict, optimizers dict)
    """
    # Ensure tensors are on device
    points = points.to(device).float()
    rgbs = rgbs.to(device).float()
    rgbs = torch.clamp(rgbs, 0, 1)
    
    N = points.shape[0]
    
    # Initialize means (positions)
    means = points  # [N, 3]
    
    # Initialize scales - fixed size in log space
    scales = torch.ones((N, 3), device=device) * np.log(init_scale)  # [N, 3]
    
    # Initialize rotations (identity quaternions: [0, 0, 0, 1])
    quats = torch.zeros((N, 4), device=device)
    quats[:, 3] = 1.0  # [N, 4]
    
    # Initialize opacities (logit space)
    init_opacity = np.clip(init_opacity, 0.001, 0.999)
    logit_opacity = np.log(init_opacity / (1 - init_opacity))
    opacities = torch.ones((N, 1), device=device) * logit_opacity  # [N, 1]
    
    # Initialize colors (DC SH component for RGB)
    # Convert RGB to SH DC space
    colors_sh = RGB2SH(rgbs)  # [N, 3]
    sh0 = colors_sh.unsqueeze(1)  # [N, 1, 3]
    
    # Create ParameterDict (mimics gsplat's approach)
    splats = nn.ParameterDict({
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
    })
    
    # Create optimizers for each parameter with different learning rates
    optimizers = {
        "means": Adam([{"params": splats["means"], "lr": means_lr}], eps=1e-15),
        "scales": Adam([{"params": splats["scales"], "lr": scale_lr}], eps=1e-15),
        "quats": Adam([{"params": splats["quats"], "lr": quat_lr}], eps=1e-15),
        "opacities": Adam([{"params": splats["opacities"], "lr": opacity_lr}], eps=1e-15),
        "sh0": Adam([{"params": splats["sh0"], "lr": sh0_lr}], eps=1e-15),
    }
    
    return splats, optimizers


class Trainer:
    """3DGS Trainer following gsplat official pattern."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: dict,
        force_cpu: bool = False,
    ):
        """Initialize trainer."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        mkdir_p(str(self.output_dir))
        
        self.config = config
        self.device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Trainer] Device: {self.device}")
        
        # Load dataset
        print("[Trainer] Loading dataset...")
        self.dataset = Dataset(data_dir)
        points, colors = self.dataset.get_pointcloud()
        self.poses_c2w = self.dataset.get_poses_torch().to(self.device)
        self.K = self.dataset.get_intrinsics_torch().to(self.device)
        
        print(f"[Trainer] Points: {len(points)}, Cameras: {len(self.poses_c2w)}")
        
        # Create Gaussians and optimizers
        init_scale = config.get('init_scale', 0.2)
        init_opacity = config.get('init_opacity', 0.5)
        self.splats, self.optimizers = create_gaussians_with_optimizers(
            points=points,
            rgbs=colors,
            init_scale=init_scale,
            init_opacity=init_opacity,
            means_lr=config['lr_xyz'],
            scale_lr=config['lr_scaling'],
            opacity_lr=config['lr_opacity'],
            quat_lr=config['lr_rotation'],
            device=str(self.device),
        )
        
        print(f"[Trainer] Initialized {len(self.splats['means'])} Gaussians")
        
        # Setup learning rate scheduler (only for means which has schedule)
        self.scheduler = ExponentialLR(self.optimizers["means"], gamma=config['lr_decay'])
        
        # Setup strategy
        self.strategy = DefaultStrategy(
            prune_opa=config.get('prune_opa', 0.005),
            grow_grad2d=config.get('grow_grad2d', 0.0002),
            grow_scale3d=config.get('grow_scale3d', 0.01),
            grow_scale2d=config.get('grow_scale2d', 0.05),
            prune_scale3d=config.get('prune_scale3d', 0.1),
            prune_scale2d=config.get('prune_scale2d', 0.15),
            refine_start_iter=config.get('refine_start_iter', 500),
            refine_stop_iter=config.get('refine_stop_iter', 15000),
            refine_every=config.get('refine_every', 100),
            reset_every=config.get('reset_every', 3000),
            verbose=True,
        )
        
        # Initialize strategy state
        self.strategy_state = self.strategy.initialize_state()
        print(f"[Trainer] Strategy: densification {self.strategy.refine_start_iter}-{self.strategy.refine_stop_iter} iters, every {self.strategy.refine_every}")
        
        # Tensorboard
        self.tb_writer = SummaryWriter(str(self.output_dir / "runs"))
        self.iteration = 0
    
    def rasterize_splats(self, camtoworld: torch.Tensor, K: torch.Tensor, width: int, height: int):
        """Render a single image."""
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)  # [N]
        sh0 = self.splats["sh0"]  # [N, 1, 3]
        
        # Combine colors
        colors = sh0  # [N, 1, 3]
        
        # World-to-camera transformation
        viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1, 4, 4]
        K_batch = K.unsqueeze(0)  # [1, 3, 3]
        
        # Render
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors.squeeze(1),  # [N, 3]
            viewmats=viewmat,
            Ks=K_batch,
            width=width,
            height=height,
            packed=True,
            near_plane=0.01,
            far_plane=100.0,
        )
        
        # render_colors: [1, H, W, 3]
        image = render_colors.squeeze(0).permute(2, 0, 1)  # [3, H, W]
        self.last_info = info
        
        return image
    
    def train_step(self, frame_indices: np.ndarray) -> float:
        """Run one training step."""
        # Zero gradients for all optimizers
        for opt in self.optimizers.values():
            opt.zero_grad()
        
        total_loss = 0.0
        num_frames = len(frame_indices)
        
        # Render all frames in batch
        for frame_idx in frame_indices:
            target_image = self.dataset.get_image(frame_idx)  # [3, H, W]
            target_image = target_image.to(self.device)
            
            pose_c2w = self.poses_c2w[frame_idx]
            
            # Render
            rendered = self.rasterize_splats(
                pose_c2w,
                self.K,
                self.dataset.image_width,
                self.dataset.image_height,
            )
            
            # Loss
            loss = F.l1_loss(rendered, target_image)
            total_loss += loss
        
        total_loss = total_loss / num_frames
        
        # Pre-backward step
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
        )
        
        # Backward
        total_loss.backward()
        
        # Optimizer steps for all parameters
        for opt in self.optimizers.values():
            opt.step()
        
        # Post-backward step (handles split/clone/prune densification)
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.iteration,
            info=self.last_info,
            packed=True,
        )
        
        return total_loss.item()
    
    def train(self, num_epochs: int, batch_size: int = 4):
        """Run training loop."""
        print(f"\n[Trainer] Starting training: {num_epochs} epochs, batch_size={batch_size}\n")
        
        num_batches = (len(self.dataset) + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(self.dataset))
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0
            
            for batch_idx in pbar:
                # Get batch
                start = batch_idx * batch_size
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # Train step
                loss = self.train_step(batch_indices)
                epoch_loss += loss
                
                # Log
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                self.tb_writer.add_scalar('loss/train', loss, self.iteration)
                self.tb_writer.add_scalar('gs_count', len(self.splats["means"]), self.iteration)
                
                self.iteration += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(epoch + 1)
            
            # LR schedule
            self.scheduler.step()
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint."""
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'iteration': self.iteration,
            'splats': {k: v.data for k, v in self.splats.items()},
        }, ckpt_path)
        print(f"[Trainer] Saved checkpoint: {ckpt_path}")
    
    def save_ply(self, filename: str = "gaussians.ply"):
        """Export as PLY file."""
        import struct
        
        means = self.splats["means"].detach().cpu().numpy()
        scales = self.splats["scales"].detach().cpu().numpy()
        quats = self.splats["quats"].detach().cpu().numpy()
        opacities = self.splats["opacities"].squeeze(-1).detach().cpu().numpy()
        sh0 = self.splats["sh0"].squeeze(1).detach().cpu().numpy()
        
        num_points = len(means)
        
        ply_path = self.output_dir / filename
        
        with open(ply_path, 'wb') as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {num_points}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float scale_0\n")
            f.write(b"property float scale_1\n")
            f.write(b"property float scale_2\n")
            f.write(b"property float rot_0\n")
            f.write(b"property float rot_1\n")
            f.write(b"property float rot_2\n")
            f.write(b"property float rot_3\n")
            f.write(b"property float opacity\n")
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            f.write(b"end_header\n")
            
            # Data
            for i in range(num_points):
                f.write(struct.pack('f', means[i, 0]))
                f.write(struct.pack('f', means[i, 1]))
                f.write(struct.pack('f', means[i, 2]))
                f.write(struct.pack('f', scales[i, 0]))
                f.write(struct.pack('f', scales[i, 1]))
                f.write(struct.pack('f', scales[i, 2]))
                f.write(struct.pack('f', quats[i, 0]))
                f.write(struct.pack('f', quats[i, 1]))
                f.write(struct.pack('f', quats[i, 2]))
                f.write(struct.pack('f', quats[i, 3]))
                f.write(struct.pack('f', opacities[i]))
                f.write(struct.pack('f', sh0[i, 0]))
                f.write(struct.pack('f', sh0[i, 1]))
                f.write(struct.pack('f', sh0[i, 2]))
        
        print(f"[Trainer] Saved PLY: {ply_path}")


if __name__ == "__main__":
    # Config
    config = {
        'lr_xyz': 0.00016,
        'lr_color': 0.0025,
        'lr_opacity': 0.05,
        'lr_scaling': 0.005,
        'lr_rotation': 0.001,
        'lr_decay': 0.9999,
        'checkpoint_interval': 5,
        'init_scale': 0.2,  # Fixed size in meters
        'init_opacity': 0.5,
        'refine_start_iter': 500,  # Start densification at iteration 500
        'refine_stop_iter': 15000,  # Stop densification at iteration 15000
        'refine_every': 100,  # Densify every 100 iterations
    }
    
    data_dir = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/itri58_colored_pcd"
    output_dir = "output_v2"
    
    trainer = Trainer(data_dir, output_dir, config)
    trainer.train(num_epochs=10, batch_size=8)
    trainer.save_ply("gaussian_reconstruction_v2.ply")
    print("\nTraining complete!")
