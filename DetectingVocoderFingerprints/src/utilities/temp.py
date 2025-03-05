import pandas as pd
import os

def find_unique_filenames_from_folder(folder_path, csv_file1, csv_file2):
    """
    Compares filenames in a folder with filenames from two CSV files (containing absolute paths),
    and returns filenames that are not shared by both.

    Args:
        folder_path (str): Path to the folder containing the files.
        csv_file1 (str): Path to the first CSV file.
        csv_file2 (str): Path to the second CSV file.

    Returns:
        set: Filenames that are unique to one source (either the folder or the concatenated CSVs).
    """
    # Extract filenames from the folder
    filenames_from_folder = set(os.listdir(folder_path))  # Gets only filenames
    print(len(filenames_from_folder))

    # Load both CSVs, concatenate them into one DataFrame, and extract filenames
    df1 = pd.read_csv(csv_file1, header=None)
    df2 = pd.read_csv(csv_file2, header=None)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Extract filenames from absolute paths
    filenames_from_csvs = set(os.path.basename(path) for path in combined_df[0])

    # Find unique filenames
    unique_filenames = filenames_from_folder.symmetric_difference(filenames_from_csvs)  # Find differences

    return unique_filenames

# Example Usage
folder_path = "/USERSPACE/DATASETS/WaveFake/ljspeech_hnsf"
csv_file1 = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_seed_files/40/train/ljspeech_hnsf_train.csv"
csv_file2 = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_seed_files/40/test/ljspeech_hnsf_test.csv"

unique_filenames = find_unique_filenames_from_folder(folder_path, csv_file1, csv_file2)
print("Unique filenames (not shared by both sources):")
for filename in unique_filenames:
    print(filename)

