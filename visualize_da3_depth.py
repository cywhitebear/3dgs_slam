"""
Visualize DA3 predicted depth + confidence + image side-by-side.
Saves visualization PNGs and displays depth range stats.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_da3_frame(npz_path, output_dir=None):
    """Load and visualize one DA3 .npz frame."""
    data = np.load(npz_path)

    depth = data['depth']      # (H, W)
    conf = data['conf']        # (H, W)
    image = data['image']      # (H, W, 3)
    K = data['intrinsics']     # (3, 3)

    print(f"\n[viz] Frame: {npz_path.name}")
    print(f"  Image size: {image.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  K:\n{K}")
    print(f"\n  Depth stats:")
    print(f"    min={depth.min():.2f}m, max={depth.max():.2f}m, mean={depth.mean():.2f}m")
    print(f"    median={np.median(depth):.2f}m, std={depth.std():.2f}m")
    print(f"\n  Confidence stats:")
    print(f"    min={conf.min():.4f}, max={conf.max():.4f}, mean={conf.mean():.4f}")

    # Normalize depth for visualization (avoid extremes distorting colormap)
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_vis = (plt.cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    # Normalize confidence for visualization
    conf_norm = np.clip(conf / conf.max(), 0, 1)
    conf_vis = (plt.cm.hot(conf_norm)[:, :, :3] * 255).astype(np.uint8)

    # Convert image to 0-255 if needed
    if image.max() <= 1.0:
        image_vis = (image * 255).astype(np.uint8)
    else:
        image_vis = image.astype(np.uint8)

    # RGB to BGR for cv2
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
    conf_vis = cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR)

    # Resize for display (all same height)
    h = image_vis.shape[0]
    w = image_vis.shape[1]
    depth_vis = cv2.resize(depth_vis, (w, h))
    conf_vis = cv2.resize(conf_vis, (w, h))

    # Combine horizontally: image | depth | conf
    combined = np.hstack([image_vis, depth_vis, conf_vis])

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{npz_path.stem}_viz.png"
        cv2.imwrite(str(out_file), combined)
        print(f"\n  Saved: {out_file}")

    # Display
    cv2.imshow("Image | Depth | Confidence", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", help="Path to .npz file")
    parser.add_argument("--dir", default="/media/ee904/DATA1/sdc_reconstruction/Track3/da3_depth_all",
                        help="DA3 depth dir (will pick first file if --npz not given)")
    parser.add_argument("--output", help="Output dir for PNG (optional)")
    args = parser.parse_args()

    if args.npz:
        npz_file = Path(args.npz)
    else:
        npz_files = sorted(Path(args.dir).glob("*.npz"))
        if not npz_files:
            print(f"No .npz files in {args.dir}")
            sys.exit(1)
        npz_file = npz_files[0]
        print(f"Using first file: {npz_file.name}")

    visualize_da3_frame(npz_file, args.output)
