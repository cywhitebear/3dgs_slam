"""
Render comparison video: 2x2 grid with render, GT, DA3 depth, DA3 confidence.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat.rendering import rasterization
from scene.dataset_readers_track3_da3 import Dataset

C0 = 0.28209479177387814


def load_ply(ply_path, device="cuda"):
    """Load Gaussian Splatting PLY (14 float properties per vertex)."""
    with open(ply_path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip()
        assert fmt == b"format binary_little_endian 1.0", fmt
        num_points = 0
        properties = []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                num_points = int(line.split()[-1])
            elif line.startswith(b"property"):
                properties.append(line.split()[-1].decode())
            elif line == b"end_header":
                break

        data = np.frombuffer(
            f.read(num_points * len(properties) * 4), dtype=np.float32
        ).reshape(num_points, len(properties))

    cols = {name: i for i, name in enumerate(properties)}
    t = lambda *names: torch.from_numpy(
        np.ascontiguousarray(data[:, [cols[n] for n in names]])
    ).float().to(device)

    means = t("x", "y", "z")
    scales = torch.exp(t("scale_0", "scale_1", "scale_2"))
    quats = t("rot_0", "rot_1", "rot_2", "rot_3")
    opacities = torch.sigmoid(t("opacity").squeeze(-1))
    sh0 = t("f_dc_0", "f_dc_1", "f_dc_2")
    rgb = torch.clamp(sh0 * C0 + 0.5, 0.0, 1.0)

    print(f"[load_ply] {num_points} gaussians from {ply_path}")
    return means, quats, scales, opacities, rgb


@torch.no_grad()
def render_frame(means, quats, scales, opacities, rgb, camtoworld, K, width, height):
    """Render single frame."""
    viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)
    image_rgb, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=rgb,
        viewmats=viewmat,
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=1000.0,
    )
    img = torch.clamp(image_rgb[0], 0.0, 1.0)
    return (img.cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3] RGB


def visualize_depth(depth):
    """Colormap depth for visualization (viridis)."""
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_vis = (plt.cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)


def visualize_confidence(conf):
    """Colormap confidence for visualization (hot)."""
    conf_norm = np.clip(conf / (conf.max() + 1e-6), 0, 1)
    conf_vis = (plt.cm.hot(conf_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", default="output_track3/track3_da3_1.ply")
    parser.add_argument("--data_dir", default="/media/ee904/DATA1/sdc_reconstruction/Track3")
    parser.add_argument("--pointcloud_file", default="Track3_lidar_da3_merge.pcd")
    parser.add_argument("--output", default="render_da3_comparison.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    means, quats, scales, opacities, rgb = load_ply(args.ply, device)

    dataset = Dataset(args.data_dir, pointcloud_file=args.pointcloud_file)
    poses_c2w = dataset.get_poses_torch().to(device)
    K = dataset.get_intrinsics_torch().to(device)
    W, H = dataset.image_width, dataset.image_height

    frame_indices = list(range(0, len(dataset), args.stride))
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    # 2x2 grid: width=2*W, height=2*H
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W * 2, H * 2)
    )

    for idx in tqdm(frame_indices, desc="Rendering"):
        # Render
        render = render_frame(
            means, quats, scales, opacities, rgb, poses_c2w[idx], K, W, H
        )

        # Ground truth
        gt = dataset.get_image(idx)  # [3, H, W] in [0, 1] RGB
        gt = (gt.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

        # DA3 depth and confidence
        da3_depth, da3_conf = dataset.get_da3_depth(idx, device=device)

        if da3_depth is not None:
            da3_depth_np = da3_depth.cpu().numpy()
            da3_conf_np = da3_conf.cpu().numpy()
            depth_vis = visualize_depth(da3_depth_np)
            conf_vis = visualize_confidence(da3_conf_np)
        else:
            # Placeholder if DA3 missing
            depth_vis = np.zeros((H, W, 3), dtype=np.uint8)
            conf_vis = np.zeros((H, W, 3), dtype=np.uint8)

        # Convert render RGB to BGR
        render_bgr = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)

        # Combine 2x2: top-left=render, top-right=gt, bottom-left=depth, bottom-right=conf
        top = np.hstack([render_bgr, gt])
        bottom = np.hstack([depth_vis, conf_vis])
        combined = np.vstack([top, bottom])

        writer.write(combined)

    writer.release()
    print(f"[main] Saved video: {args.output} ({len(frame_indices)} frames, {W*2}x{H*2})")


if __name__ == "__main__":
    main()
