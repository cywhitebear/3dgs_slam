"""
Quick test training script with a small subset of data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Trainer


def main():
    data_dir = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/itri58_colored_pcd"
    output_dir = "output_test"
    
    # Use default config with smaller training
    config = {
        'lr_xyz': 0.00016,
        'lr_color': 0.0025,
        'lr_opacity': 0.05,
        'lr_scaling': 0.005,
        'lr_rotation': 0.001,
        'lr_decay': 0.9999,
        'checkpoint_interval': 5,
    }
    
    print("=" * 60)
    print("3D Gaussian Splatting - Test Training")
    print("=" * 60)
    
    # Create trainer with GPU if available
    trainer = Trainer(data_dir, output_dir, config, force_cpu=False)
    
    # Run quick test training
    print("\n[Test] Running 5 epochs with batch_size=8...")
    trainer.train(num_epochs=5, batch_size=8)
    
    # Save trained Gaussians as PLY
    trainer.save_gaussians_as_ply("gaussian_reconstruction.ply")
    
    print("\n[Test] Training complete!")
    print(f"[Test] Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
