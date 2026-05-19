import os

def check_duplicates_and_save_list():
    # --- PATH SETTINGS ---
    test_gt_dir = "/media/ee904/DATA1/Track3/track3_test_gt_images"
    train_img_dir = "/media/ee904/DATA1/Track3/image"
    output_list_path = "/media/ee904/DATA1/Track3/track3_test_frame_list.txt"

    # 1. Get the list of all files in the test GT directory
    if not os.path.exists(test_gt_dir):
        print(f"Error: Test GT directory not found: {test_gt_dir}")
        return

    test_gt_files = sorted([f for f in os.listdir(test_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    test_gt_ids = {os.path.splitext(f)[0] for f in test_gt_files}

    # 2. Get the set of IDs in the training image directory for fast lookup
    if not os.path.exists(train_img_dir):
        print(f"Error: Training image directory not found: {train_img_dir}")
        return

    train_ids = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

    # 3. Check for duplicates (intersection of sets)
    duplicates = test_gt_ids.intersection(train_ids)

    if duplicates:
        print(f"ALERT: Found {len(duplicates)} duplicate IDs between Test and Train sets!")
        print(f"Duplicate IDs: {sorted(list(duplicates))[:10]}...") 
    else:
        print("Success: No duplicates found. Test set is distinct from Training set.")

    # 4. Write the list of test images to the text file
    # We use the actual filenames (including extensions) to match your previous workflows
    print(f"Writing {len(test_gt_files)} filenames to {output_list_path}...")
    
    with open(output_list_path, 'w') as f:
        for filename in test_gt_files:
            f.write(f"{filename}\n")

    print("Done!")

if __name__ == "__main__":
    check_duplicates_and_save_list()