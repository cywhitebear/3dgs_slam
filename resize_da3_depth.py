"""
Resize DA3 predicted depth from 378x504 to 720x960 (image size).
Saves resized depth back to .npz files.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

def resize_da3_depths(da3_dir, output_dir, target_size=(960, 720)):
    """
    Resize all .npz depth predictions to target image size.

    Args:
        da3_dir: Input dir with .npz files (378x504 depth)
        output_dir: Output dir for resized .npz files
        target_size: (width, height) = (960, 720)
    """
    da3_path = Path(da3_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(da3_path.glob("*.npz"))
    print(f"[resize] Found {len(npz_files)} .npz files")

    for npz_file in tqdm(npz_files, desc="Resizing"):
        data = np.load(npz_file)

        depth = data["depth"]  # (378, 504)
        conf = data["conf"]    # (378, 504)

        # Resize using bilinear interpolation (preserve metric values)
        depth_resized = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(conf, target_size, interpolation=cv2.INTER_LINEAR)
        image_resized = cv2.resize(data["image"], target_size, interpolation=cv2.INTER_LINEAR)

        # Adjust intrinsics to match resized dimensions
        K = data["intrinsics"]
        h_orig, w_orig = depth.shape
        h_new, w_new = target_size[1], target_size[0]
        scale_x = w_new / w_orig
        scale_y = h_new / h_orig
        K_resized = K.copy()
        K_resized[0, 0] *= scale_x  # fx
        K_resized[1, 1] *= scale_y  # fy
        K_resized[0, 2] *= scale_x  # cx
        K_resized[1, 2] *= scale_y  # cy

        # Save resized
        out_file = out_path / npz_file.name
        np.savez(
            out_file,
            depth=depth_resized,
            conf=conf_resized,
            image=image_resized,
            intrinsics=K_resized,
        )

    print(f"[resize] Saved {len(npz_files)} resized files to {out_path}")

if __name__ == "__main__":
    da3_dir = "/media/ee904/DATA1/sdc_reconstruction/Track3/da3_Track3_results"
    output_dir = "/media/ee904/DATA1/sdc_reconstruction/Track3/da3_depth_all"

    resize_da3_depths(da3_dir, output_dir, target_size=(960, 720))
    print("Done!")
