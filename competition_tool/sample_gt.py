import os
import shutil

def copy_even_sampled_images(compressed_dir, train_dir, glitch_file, dest_dir, num_samples=30):
    """
    Samples exactly 30 frames evenly across a specific timeline window, 
    ensuring glitch frames and training frames are excluded before sampling.
    """
    # Updated target range boundaries
    START_TS = 1777884729516471269
    END_TS = 1777885248816711303

    # 1. Identify all compressed source files
    all_compressed_files = {f for f in os.listdir(compressed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    all_compressed_ids = {os.path.splitext(f)[0]: f for f in all_compressed_files}
    
    # 2. Get blacklist: IDs in training set
    train_ids = {os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    
    # 3. Get blacklist: Glitch frames [cite: 3]
    glitch_ids = set()
    if os.path.exists(glitch_file):
        with open(glitch_file, 'r') as f:
            glitch_ids = {line.strip() for line in f if line.strip()}
    
    # 4. Filter valid candidates within the specified range
    # We build the entire pool of valid frames FIRST to ensure we can pick enough
    candidate_ids = []
    for fid in all_compressed_ids.keys():
        fid_int = int(fid)
        # Apply range filter and exclude blacklisted frames
        if START_TS <= fid_int <= END_TS:
            if fid not in train_ids and fid not in glitch_ids:
                candidate_ids.append(fid)
    
    # Sort candidates chronologically to ensure even temporal spacing
    candidate_ids.sort(key=int)
    
    total_candidates = len(candidate_ids)
    if total_candidates < num_samples:
        raise ValueError(f"Only {total_candidates} candidates available in the range after filtering glitches. "
                         f"Required: {num_samples}. Please widen the START_TS/END_TS window.")

    # 5. Calculate Even Indices to guarantee exactly 30 samples
    selected_ids = []
    for i in range(num_samples):
        # Linear spacing ensures we represent the full duration of the window
        idx = int(i * (total_candidates - 1) / (num_samples - 1))
        selected_ids.append(candidate_ids[idx])
    
    # 6. Create destination folder and copy files
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"Pool size: {total_candidates} frames. Sampling {num_samples} frames at even intervals...")
    for fid in selected_ids:
        filename = all_compressed_ids[fid]
        shutil.copy2(os.path.join(compressed_dir, filename), os.path.join(dest_dir, filename))
    
    # 7. Save reference list for the competition
    with open("track3_test_frame_list.txt", "w") as f:
        for fid in selected_ids:
            f.write(f"{fid}.jpg\n")
            
    print(f"Done. {num_samples} images copied to: {dest_dir}")

if __name__ == "__main__":
    # Maintaining your directory settings
    CONFIG = {
        "compressed_dir": "/media/ee904/DATA1/Track3/image_all",
        "train_dir": "/media/ee904/DATA1/Track3/image",
        "glitch_file": "/media/ee904/DATA1/Track3/glitch_frames.txt",
        "dest_dir": "/media/ee904/DATA1/Track3/track3_test_gt_images"
    }
    
    copy_even_sampled_images(**CONFIG, num_samples=30)