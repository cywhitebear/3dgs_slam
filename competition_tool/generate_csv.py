import pandas as pd

def generate_kaggle_csvs(frame_list_path):
    # 1. Load the official 30 test frame timestamps
    with open(frame_list_path, 'r') as f:
        # Strip '.jpg' and whitespace to get the raw timestamp ID[cite: 1]
        frame_ids = [line.strip().split('.')[0] for line in f if line.strip()]

    # 2. Create sample_submission.csv
    # This is what students download and include in their submission.zip
    sample_sub = pd.DataFrame({
        'Id': frame_ids,
        'Predicted': [0.0] * len(frame_ids)  # Dummy placeholder values
    })
    sample_sub.to_csv('submission.csv', index=False)

    # 3. Create solution.csv
    # This is your hidden "answer key" mapping the IDs to the leaderboard
    solution = pd.DataFrame({
        'Id': frame_ids,
        'Expected': [0.0] * len(frame_ids),  # Dummy placeholder ignored by your script
        'Usage': ['Public'] * len(frame_ids)
    })
    solution.to_csv('solution.csv', index=False)

    print(f"Generated {len(frame_ids)} rows in solution.csv and submission.csv")

if __name__ == "__main__":
    # Path to your official frame list[cite: 1]
    FRAME_LIST = "/media/ee904/DATA1/Track3/track3_test_frame_list.txt"
    generate_kaggle_csvs(FRAME_LIST)