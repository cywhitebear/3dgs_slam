"""
Dataset reader for Track3 dataset with Depth Anything 3.
Loads new pointcloud file and image/pose data.
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
    def __init__(self, data_dir, pointcloud_file="Track3_lidar_da3_merge.pcd"):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing dataset
            pointcloud_file: Name of pointcloud file to load (relative to data_dir)
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "image_all"
        self.poses_file = self.data_dir / "camera_poses_all.csv"
        self.intrinsics_file = self.data_dir / "camera_intrinsics.json"
        self.pointcloud_file = self.data_dir / pointcloud_file
        self.sky_mask_dir = self.data_dir / "sky_masks"

        # Load data
        self._load_intrinsics()
        self._load_poses()
        self._load_image_list()
        self._load_pointcloud()

    def _load_intrinsics(self):
        """Load camera intrinsics from JSON file."""
        with open(self.intrinsics_file, 'r') as f:
            data = json.load(f)

        K_flat = data['K']
        K_matrix = np.array(K_flat, dtype=np.float32).reshape(3, 3)

        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]

        self.image_width = data['width']
        self.image_height = data['height']
        self.K = K_matrix

        print(f"[Dataset] Image size: {self.image_width}x{self.image_height}")
        print(f"[Dataset] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    def _load_poses(self):
        """Load camera poses from CSV file."""
        df = pd.read_csv(self.poses_file)

        self.timestamps = df['timestamp'].values
        self.poses = []

        matrix_cols = [f'm{i}{j}' for i in range(4) for j in range(4)]

        for idx, row in df.iterrows():
            pose_values = [row[col] for col in matrix_cols]
            pose_matrix = np.array(pose_values, dtype=np.float32).reshape(4, 4)
            self.poses.append(pose_matrix)

        print(f"[Dataset] Loaded {len(self.poses)} camera poses from CSV")

    def _load_image_list(self):
        """Load list of image filenames and match with poses."""
        image_filenames = sorted(os.listdir(self.image_dir))

        timestamp_strs = [str(ts) for ts in self.timestamps]

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

        if self.colors_map.max() > 1.0:
            self.colors_map = self.colors_map / 255.0

        print(f"[Dataset] Loaded point cloud with {len(self.points_map)} points from {self.pointcloud_file.name}")

    def get_poses_torch(self):
        """Get poses as torch tensors (c2w matrices)."""
        poses_list = []
        for idx in self.valid_pose_indices:
            pose = self.poses[idx]
            poses_list.append(torch.from_numpy(pose).float())
        return torch.stack(poses_list)

    def get_intrinsics_torch(self):
        """Get intrinsics as torch tensor."""
        return torch.from_numpy(self.K).float()

    def get_image(self, frame_idx):
        """Load image for a given frame."""
        if frame_idx >= len(self.image_filenames):
            raise IndexError(f"Frame {frame_idx} out of range")

        image_path = self.image_dir / self.image_filenames[frame_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def get_image_batch(self, frame_indices):
        """Load multiple images."""
        images = [self.get_image(i) for i in frame_indices]
        return torch.stack(images)

    def get_pointcloud(self):
        """Get point cloud map."""
        points = torch.from_numpy(self.points_map).float()
        colors = torch.from_numpy(self.colors_map).float()
        return points, colors

    def get_sky_mask(self, frame_idx):
        """Load sky mask for initialization (255 for sky, 0 for other)."""
        if not self.sky_mask_dir.exists():
            return None

        timestamp_str = str(self.timestamps[self.valid_pose_indices[frame_idx]])
        mask_path = self.sky_mask_dir / f"{timestamp_str}.png"

        if not mask_path.exists():
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(mask)

    def get_lidar_depth(self, frame_idx, device="cuda"):
        """Project point cloud into camera frame to create sparse depth map."""
        points = torch.from_numpy(self.points_map).float().to(device)
        pose_c2w = torch.from_numpy(self.poses[self.valid_pose_indices[frame_idx]]).float().to(device)
        K = torch.from_numpy(self.K).float().to(device)

        pose_w2c = torch.linalg.inv(pose_c2w)

        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=-1)
        p_cam = (pose_w2c @ points_h.T).T

        z = p_cam[:, 2]
        mask = z > 0.1

        p_cam = p_cam[mask]
        z = z[mask]

        p_pix = (K @ (p_cam[:, :3] / z.unsqueeze(-1)).T).T
        u = p_pix[:, 0].long()
        v = p_pix[:, 1].long()

        valid_mask = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]

        depth_map = torch.zeros((self.image_height, self.image_width), device=device)

        indices = torch.argsort(z, descending=True)
        u, v, z = u[indices], v[indices], z[indices]

        depth_map[v, u] = z

        return depth_map

    def __len__(self):
        """Get number of frames."""
        return len(self.image_filenames)

    def __repr__(self):
        return f"Dataset(frames={len(self)}, points={len(self.points_map)}, size={self.image_width}x{self.image_height})"
