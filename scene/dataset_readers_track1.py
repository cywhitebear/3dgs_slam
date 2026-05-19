"""
Dataset reader for Track1_gs dataset.
Handles new intrinsics format (K matrix) and CSV-based camera poses with timestamps.
"""

import os
import json
import numpy as np
import cv2
import torch
import open3d as o3d
import pandas as pd
from pathlib import Path


class Dataset:
    """
    Loads and manages Track1_gs dataset (images, poses, intrinsics, point cloud).
    """
    
    def __init__(self, data_dir):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing dataset
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "image"
        self.poses_file = self.data_dir / "camera_pose.csv"
        self.intrinsics_file = self.data_dir / "camera_intrinsics.json"
        self.pointcloud_file = self.data_dir / "Track1_test.pcd"
        self.mask_dir = self.data_dir / "sky_masks"
        
        # Load data
        self._load_intrinsics()
        self._load_poses()
        self._load_image_list()
        self._load_pointcloud()
    
    def _load_intrinsics(self):
        """Load camera intrinsics from JSON file (new format)."""
        with open(self.intrinsics_file, 'r') as f:
            data = json.load(f)
        
        # Extract intrinsics from K matrix (9 elements: row-major)
        K_flat = data['K']
        K_matrix = np.array(K_flat, dtype=np.float32).reshape(3, 3)
        
        # Extract parameters
        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]
        
        self.image_width = data['width']
        self.image_height = data['height']
        
        # Store intrinsic matrix
        self.K = K_matrix
        
        print(f"[Dataset] Image size: {self.image_width}x{self.image_height}")
        print(f"[Dataset] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"[Dataset] Distortion model: {data.get('distortion_model', 'unknown')}")
    
    def _load_poses(self):
        """Load camera poses from CSV file."""
        # Read CSV: timestamp, m00, m01, ..., m33
        df = pd.read_csv(self.poses_file)
        
        self.timestamps = df['timestamp'].values
        self.poses = []
        
        # Extract 4x4 matrices from columns m00-m33
        matrix_cols = [f'm{i}{j}' for i in range(4) for j in range(4)]
        
        for idx, row in df.iterrows():
            pose_values = [row[col] for col in matrix_cols]
            pose_matrix = np.array(pose_values, dtype=np.float32).reshape(4, 4)
            self.poses.append(pose_matrix)
        
        print(f"[Dataset] Loaded {len(self.poses)} camera poses from CSV")
    
    def _load_image_list(self):
        """Load list of image filenames and match with poses."""
        image_filenames = sorted(os.listdir(self.image_dir))
        
        # Convert timestamps to strings for matching
        timestamp_strs = [str(ts) for ts in self.timestamps]
        
        # Filter images that have corresponding poses
        self.image_filenames = []
        self.valid_pose_indices = []
        
        for i, ts_str in enumerate(timestamp_strs):
            image_path = self.image_dir / f"{ts_str}.jpg"
            if image_path.exists():
                self.image_filenames.append(f"{ts_str}.jpg")
                self.valid_pose_indices.append(i)
        
        print(f"[Dataset] Found {len(self.image_filenames)} matched images with poses")
    
    def _load_pointcloud(self):
        """Load point cloud map."""
        if not self.pointcloud_file.exists():
            print(f"[Dataset] Warning: Point cloud file not found at {self.pointcloud_file}")
            self.points_map = np.array([], dtype=np.float32).reshape(0, 3)
            self.colors_map = np.array([], dtype=np.float32).reshape(0, 3)
            return
        
        pcd = o3d.io.read_point_cloud(str(self.pointcloud_file))
        self.points_map = np.asarray(pcd.points, dtype=np.float32)
        self.colors_map = np.asarray(pcd.colors, dtype=np.float32)
        
        # Clamp colors to [0, 1] range if needed
        if self.colors_map.max() > 1.0:
            self.colors_map = self.colors_map / 255.0
        
        print(f"[Dataset] Loaded point cloud with {len(self.points_map)} points")
    
    def get_poses_torch(self):
        """Get poses as torch tensors (c2w matrices)."""
        poses_list = []
        for idx in self.valid_pose_indices:
            pose = self.poses[idx]
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
            frame_idx: Frame index (0-based, into the filtered image list)
        
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

    def get_mask(self, frame_idx):
        """
        Load a single binary sky mask.
        
        Args:
            frame_idx: Frame index (0-based)
        
        Returns:
            Mask as torch tensor (H, W), or None if mask doesn't exist
        """
        if not self.mask_dir.exists():
            return None
        
        timestamp_str = str(self.timestamps[self.valid_pose_indices[frame_idx]])
        mask_path = self.mask_dir / f"{timestamp_str}.png"
        
        if not mask_path.exists():
            return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(mask)  # (H, W)

    def get_lidar_depth(self, frame_idx, device="cuda"):
        """
        Project the global point cloud into a camera frame to create a sparse depth map.
        
        Returns:
            depth_map: (H, W) torch tensor with depth values in meters
        """
        # 1. Get raw data from the class attributes
        points = torch.from_numpy(self.points_map).float().to(device) # (N, 3)
        pose_c2w = torch.from_numpy(self.poses[self.valid_pose_indices[frame_idx]]).float().to(device) # (4, 4)
        K = torch.from_numpy(self.K).float().to(device) # (3, 3)
        
        # 2. Transform points from World space to Camera space
        # P_cam = w2c * P_world
        pose_w2c = torch.linalg.inv(pose_c2w)
        
        # Add homogeneous coordinate for matrix multiplication
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=-1)
        p_cam = (pose_w2c @ points_h.T).T # (N, 4)
        
        # 3. Filter points
        # Only keep points in front of the camera (positive Z)
        z = p_cam[:, 2]
        mask = z > 0.1 # Near plane clipping
        
        p_cam = p_cam[mask]
        z = z[mask]
        
        # 4. Project to 2D pixel coordinates
        # [u, v, 1] = K * [x/z, y/z, 1]
        p_pix = (K @ (p_cam[:, :3] / z.unsqueeze(-1)).T).T # (N, 3)
        u = p_pix[:, 0].long()
        v = p_pix[:, 1].long()
        
        # 5. Filter points within image boundaries
        valid_mask = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
        
        # 6. Create sparse depth map with Z-buffer logic
        # Initialize with zeros
        depth_map = torch.zeros((self.image_height, self.image_width), device=device)
        
        # Sort by depth descending so that when we index_put, the closest points (last written) remain
        indices = torch.argsort(z, descending=True)
        u, v, z = u[indices], v[indices], z[indices]
        
        depth_map[v, u] = z
        
        return depth_map

    def __len__(self):
        """Get number of frames."""
        return len(self.image_filenames)
    
    def __repr__(self):
        return f"Dataset(frames={len(self)}, points={len(self.points_map)}, size={self.image_width}x{self.image_height})"
