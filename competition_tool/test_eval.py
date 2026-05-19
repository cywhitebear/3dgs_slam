import pandas as pd
import numpy as np
import cv2
import os

# --- MOCK LOCAL ENVIRONMENT ---
# Path to your solution.csv generated previously
SOLUTION_CSV = "solution.csv" 

# Directory where you copied the 30 .jpg ground truth images
GT_DIR = "./itri58_gt_images" 

# Directory where you have test/student images (simulate a zip extraction)
# SUB_DIR = "./itri58_gt_images" 
SUB_DIR = "./test_submission" # Simulate a student submission with the same GT images (for testing)
# -------------------------------

def calculate_psnr(img1, img2):
    """Calculates PSNR between two images."""
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def load_image_with_extensions(directory, frame_id, preferred_ext=None):
    """
    Simulates the robust loader in the Kaggle custom metric.
    """
    extensions = ['.png', '.jpg', '.jpeg']
    if preferred_ext:
        ext_list = [preferred_ext] + [e for e in extensions if e != preferred_ext]
    else:
        ext_list = extensions

    for ext in ext_list:
        path = os.path.join(directory, f"{frame_id}{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    return None

def run_dry_test():
    # 1. Load the solution file
    if not os.path.exists(SOLUTION_CSV):
        print(f"Error: {SOLUTION_CSV} not found!")
        return
    
    sol_df = pd.read_csv(SOLUTION_CSV)
    psnr_scores = []

    print(f"Evaluating {len(sol_df)} rows from {SOLUTION_CSV}...")

    # 2. Replicate Kaggle Custom Metric loop
    for frame_id in sol_df['Id'].values:
        # Load Professor's GT (Specified as .jpg)
        img_gt = load_image_with_extensions(GT_DIR, frame_id, preferred_ext='.jpg')
        
        # Load Student Submission (May be .png or .jpg)
        img_sub = load_image_with_extensions(SUB_DIR, frame_id)

        if img_gt is None:
            print(f"Professor Error: Could not find GT image for {frame_id} in {GT_DIR}")
            continue

        if img_sub is None:
            print(f"Student Penalty: Missing frame {frame_id} in {SUB_DIR} -> 0.0 PSNR")
            psnr_scores.append(0.0)
            continue

        # Resolution enforcement (1440x928)
        if img_gt.shape != img_sub.shape:
            print(f"Resizing mismatch for {frame_id}: {img_sub.shape} -> {img_gt.shape}")
            img_sub = cv2.resize(img_sub, (img_gt.shape[1], img_gt.shape[0]))

        score = calculate_psnr(img_gt, img_sub)
        psnr_scores.append(score)
        print(f"Frame {frame_id}: {score:.4f} dB")

    # 3. Final Summary
    if psnr_scores:
        final_avg = np.mean(psnr_scores)
        print("\n" + "="*30)
        print(f"SIMULATED LEADERBOARD SCORE: {final_avg:.4f} dB")
        print("="*30)
    else:
        print("Error: No scores were calculated. Check your image paths.")

if __name__ == "__main__":
    run_dry_test()