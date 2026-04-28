"""
Dataset reader for loading camera poses, images, intrinsics, and point cloud.
"""

import os
import json
import numpy as np
import cv2
import torch
import open3d as o3d
from pathlib import Path


class Dataset:
    """
    Loads and manages dataset (images, poses, intrinsics, point cloud).
    """
    
    def __init__(self, data_dir):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing dataset
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "itri58_image"
        self.poses_file = self.data_dir / "camera_gt_pose.txt"
        self.intrinsics_file = self.data_dir / "camera_intrinsics.json"
        self.pointcloud_file = self.data_dir / "itri58_full_color_map.pcd"
        
        # Load data
        self._load_poses()
        self._load_intrinsics()
        self._load_image_list()
        self._load_pointcloud()
    
    def _load_poses(self):
        """Load camera poses from file."""
        self.poses = []
        with open(self.poses_file, 'r') as f:
            for line in f:
                pose_values = [float(x) for x in line.strip().split(',')]
                pose_matrix = np.array(pose_values, dtype=np.float32).reshape(4, 4)
                self.poses.append(pose_matrix)
        print(f"[Dataset] Loaded {len(self.poses)} camera poses")
    
    def _load_intrinsics(self):
        """Load camera intrinsics from JSON file."""
        with open(self.intrinsics_file, 'r') as f:
            data = json.load(f)
        
        # Extract intrinsics from camera_intrinsics.json format
        fx = data['fx']
        fy = data['fy']
        cx = data['cx']
        cy = data['cy']
        self.image_width = data['width']
        self.image_height = data['height']
        
        # Build intrinsic matrix
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        print(f"[Dataset] Image size: {self.image_width}x{self.image_height}")
        print(f"[Dataset] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    def _load_image_list(self):
        """Load list of image filenames."""
        self.image_filenames = sorted(os.listdir(self.image_dir))
        print(f"[Dataset] Found {len(self.image_filenames)} images")
    
    def _load_pointcloud(self):
        """Load point cloud map."""
        pcd = o3d.io.read_point_cloud(str(self.pointcloud_file))
        self.points_map = np.asarray(pcd.points, dtype=np.float32)
        self.colors_map = np.asarray(pcd.colors, dtype=np.float32)
        print(f"[Dataset] Loaded point cloud with {len(self.points_map)} points")
    
    def get_poses_torch(self):
        """Get poses as torch tensors (c2w matrices)."""
        poses_list = []
        for pose in self.poses:
            # Poses are already in c2w format (camera-to-world)
            poses_list.append(torch.from_numpy(pose).float())
        return torch.stack(poses_list)  # (N, 4, 4)
    
    def get_intrinsics_torch(self):
        """Get intrinsics as torch tensor."""
        return torch.from_numpy(self.K).float()  # (3, 3)
    
    def get_image(self, frame_idx):
        """
        Load image for a given frame.
        
        Args:
            frame_idx: Frame index
        
        Returns:
            Image as torch tensor (3, H, W) with values in [0, 1]
        """
        if frame_idx >= len(self.image_filenames):
            raise IndexError(f"Frame {frame_idx} out of range")
        
        image_path = self.image_dir / self.image_filenames[frame_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Convert to torch tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    
    def get_image_batch(self, frame_indices):
        """
        Load multiple images.
        
        Args:
            frame_indices: List of frame indices
        
        Returns:
            Stack of images (N, 3, H, W)
        """
        images = [self.get_image(i) for i in frame_indices]
        return torch.stack(images)
    
    def get_pointcloud(self):
        """
        Get point cloud map.
        
        Returns:
            points: (N, 3) torch tensor
            colors: (N, 3) torch tensor
        """
        points = torch.from_numpy(self.points_map).float()
        colors = torch.from_numpy(self.colors_map).float()
        return points, colors
    
    def __len__(self):
        """Get number of frames."""
        return len(self.poses)
    
    def __repr__(self):
        return f"Dataset(frames={len(self)}, points={len(self.points_map)}, size={self.image_width}x{self.image_height})"
