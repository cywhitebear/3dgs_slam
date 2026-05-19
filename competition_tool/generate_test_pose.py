import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def load_and_interpolate():
    # --- PATHS (UNCHANGED) ---
    IMG_DIR = "/media/ee904/DATA1/Track3/image"
    GT_POSE_FILE = "/media/ee904/DATA1/Track3/camera_poses.csv"
    TEST_LIST_FILE = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/track3/track3_test_frame_list.txt"
    OUTPUT_FILE = "/media/ee904/DATA1/ITRI_58/2025-03-10-10-48-26-b58-lidar-camera-ptp/3dgs_project/track3/track3_test_pose_list.txt"

    # 1. Load CSV - Use int64 for the timestamp column to avoid precision loss
    df_pose = pd.read_csv(GT_POSE_FILE)
    
    # Force timestamps to int64 immediately
    gt_timestamps_int = df_pose.iloc[:, 0].values.astype(np.int64)
    raw_matrices = df_pose.iloc[:, 1:].values 

    all_pos = []
    all_quat = []
    
    # 2. Decompose flat 16-element rows into Position and Rotation
    for i in range(len(raw_matrices)):
        matrix = raw_matrices[i].reshape(4, 4)
        all_pos.append(matrix[:3, 3])
        rot_mat = matrix[:3, :3]
        all_quat.append(R.from_matrix(rot_mat).as_quat())

    all_pos = np.array(all_pos)
    all_quat = np.array(all_quat)

    # 3. Setup Interpolation with Normalized Integer Time[cite: 1, 4]
    t0_int = gt_timestamps_int[0]
    # The difference (norm_times) is small enough to be a safe float64
    norm_times = (gt_timestamps_int - t0_int).astype(np.float64)
    
    pos_interp = interp1d(norm_times, all_pos, axis=0, kind='linear', fill_value="extrapolate")
    rotations = R.from_quat(all_quat)
    slerp = Slerp(norm_times, rotations)

    # 4. Process Test Frame List
    if not os.path.exists(TEST_LIST_FILE):
        print(f"Error: {TEST_LIST_FILE} not found.")
        return

    with open(TEST_LIST_FILE, 'r') as f:
        test_frames = [line.strip().split('.')[0] for line in f if line.strip()]
    
    interpolated_results = []
    
    print(f"Interpolating poses for {len(test_frames)} test frames...")
    for frame_id in test_frames:
        # Calculate t_target using integer subtraction FIRST to preserve precision
        # Then convert the result to float
        t_target = float(int(frame_id) - t0_int)
        
        # --- FIX: ROBUST BOUNDARY CHECKING ---
        # Slerp crashes if t_target is even 0.0000001 outside the range.
        # We clip t_target to the range of norm_times.
        t_target_safe = np.clip(t_target, norm_times[0], norm_times[-1])
        
        # Estimate Position and Rotation
        p_interp = pos_interp(t_target_safe)
        r_interp = slerp(t_target_safe).as_matrix()
        
        # Reconstruct 4x4 Matrix
        new_matrix = np.eye(4)
        new_matrix[:3, :3] = r_interp
        new_matrix[:3, 3] = p_interp
        
        interpolated_results.append(new_matrix.flatten())

    # 5. Save as flat 16-float lines
    np.savetxt(OUTPUT_FILE, interpolated_results, fmt='%.18e')
    print(f"Successfully saved test poses to {OUTPUT_FILE}")

if __name__ == "__main__":
    load_and_interpolate()