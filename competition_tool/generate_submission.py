import os
import struct
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from gsplat.rendering import rasterization

class StandaloneRenderer:
    """Renderer for RGB Novel View Synthesis using rectified intrinsics."""
    
    def __init__(self, ply_path, intrinsics, device="cuda"):
        self.device = torch.device(device)
        self.K = intrinsics.to(self.device)
        self.splats = self.load_ply(ply_path)
        
    def load_ply(self, path):
        """Loads the 14-float binary PLY format used in the midterm project[cite: 3]."""
        print(f"Loading splats from {path}...")
        with open(path, 'rb') as f:
            header = ""
            while "end_header" not in header:
                line = f.readline().decode('ascii')
                header += line
                if "element vertex" in line:
                    num_points = int(line.split()[-1])

            # point_format: 3 means, 3 scales, 4 quats, 1 opacity, 3 sh0[cite: 3]
            point_format = '14f'
            size = struct.calcsize(point_format)
            
            data = []
            for _ in range(num_points):
                data.append(struct.unpack(point_format, f.read(size)))
            
            data = torch.tensor(data, device=self.device)
            
        return nn.ParameterDict({
            "means": nn.Parameter(data[:, 0:3]),
            "scales": nn.Parameter(data[:, 3:6]),
            "quats": nn.Parameter(data[:, 6:10]),
            "opacities": nn.Parameter(data[:, 10:11]),
            "sh0": nn.Parameter(data[:, 11:14].unsqueeze(1)),
        })

    def render_rgb(self, camtoworld, width, height):
        """Standard GS rendering using SH DC components[cite: 3]."""
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).squeeze(-1)
        
        # Convert SH DC to RGB using standard constant C0[cite: 3]
        C0 = 0.28209479177387814
        rgb = self.splats["sh0"].squeeze(1) * C0 + 0.5
        
        viewmat = torch.linalg.inv(camtoworld)
        
        # Rasterization (packed=False for consistent 2D grids)[cite: 2]
        features, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=rgb,
            viewmats=viewmat.unsqueeze(0),
            Ks=self.K.unsqueeze(0),
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1000.0,
        )

        image_rgb = (features[0] + (1.0 - alphas[0]) * 0.0).permute(2, 0, 1)
        return image_rgb

def load_intrinsics_from_json(json_path):
    """
    Parses fx, fy, cx, cy from the P matrix for rectified pinhole rendering.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract from P matrix (3x4 projection) for rectified coordinate system
    # P = [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0]
    p = data['P']
    fx = p[0]
    fy = p[5]
    cx = p[2]
    cy = p[6]
    
    # Get dimensions directly from JSON
    width = data['width']
    height = data['height']
        
    k_tensor = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).float()
    
    return k_tensor, width, height

def main():
    # --- PATH CONFIGURATION ---
    PLY_FILE = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/output_track3/gaussian_reconstruction_track3_mask25.ply"
    K_JSON = "/media/ee904/DATA1/Track3/camera_intrinsics.json"
    POSE_LIST_FILE = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/track3/track3_test_pose_list.txt"
    FRAME_LIST_FILE = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/track3/track3_test_frame_list.txt"
    OUTPUT_DIR = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/track3/submission_mask25"
    # --------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load official frame timestamps and poses[cite: 4, 5]
    with open(FRAME_LIST_FILE, 'r') as f:
        frame_ids = [line.strip().split('.')[0] for line in f if line.strip()]
    raw_poses = np.loadtxt(POSE_LIST_FILE)
    test_poses = [torch.from_numpy(p.reshape(4, 4)).float() for p in raw_poses]

    # 2. Load camera parameters and dimensions from JSON[cite: 6]
    print(f"Loading camera parameters from {K_JSON}...")
    intrinsics, width, height = load_intrinsics_from_json(K_JSON)
    
    # 3. Initialize Renderer
    renderer = StandaloneRenderer(PLY_FILE, intrinsics)

    print(f"Rendering {len(frame_ids)} views at {width}x{height}...")
    for i, frame_id in enumerate(tqdm(frame_ids)):
        current_pose = test_poses[i].to(renderer.device)
        rgb = renderer.render_rgb(current_pose, width, height)
        
        # Convert to 8-bit BGR for OpenCV
        rgb_np = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{frame_id}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

    print(f"\nDone! Official renders saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()