"""
Render frames from trained 3DGS model and calculate PSNR against ground truth.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path

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


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (0-255 uint8)."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", default="output_track3/track3_da3_1.ply")
    parser.add_argument("--data_dir", default="/media/ee904/DATA1/sdc_reconstruction/Track3")
    parser.add_argument("--pointcloud_file", default="Track3_lidar_da3_merge.pcd")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all")
    parser.add_argument("--output_csv", default=None, help="Save per-frame PSNR to CSV")
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

    psnr_values = []

    print(f"[render_psnr] Rendering {len(frame_indices)} frames...")
    for idx in tqdm(frame_indices, desc="PSNR"):
        render = render_frame(
            means, quats, scales, opacities, rgb, poses_c2w[idx], K, W, H
        )

        # Load GT image
        gt = dataset.get_image(idx)  # [3, H, W] in [0, 1] RGB
        gt = (gt.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Convert render from RGB to BGR for cv2 comparison (if needed)
        # Here we keep both in RGB
        psnr = calculate_psnr(render, gt)
        psnr_values.append(psnr)

    psnr_values = np.array(psnr_values)

    print(f"\n[render_psnr] PSNR Results (stride={args.stride}):")
    print(f"  Mean PSNR:   {psnr_values.mean():.2f} dB")
    print(f"  Median PSNR: {np.median(psnr_values):.2f} dB")
    print(f"  Std PSNR:    {psnr_values.std():.2f} dB")
    print(f"  Min PSNR:    {psnr_values.min():.2f} dB")
    print(f"  Max PSNR:    {psnr_values.max():.2f} dB")

    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'psnr'])
            for fidx, psnr in zip(frame_indices, psnr_values):
                writer.writerow([fidx, psnr])
        print(f"\n[render_psnr] Saved per-frame PSNR to {args.output_csv}")


if __name__ == "__main__":
    main()
