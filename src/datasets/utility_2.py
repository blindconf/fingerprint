
import os
from src.training.invariables import DATASETS
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import torch.nn.functional as torch_nn_func
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from torch.utils.data import Sampler


def create_train_validate_test_csvs(
    audio_dir,
    dest_dir,
    classification_type,
    seed,
    same_label=False,
    starting_label=1):

    dataset_csv_dir_path = os.path.join(dest_dir, f'csv_files')

    # Creaet csv of the target dataset if not existing
    if not os.path.exists(dataset_csv_dir_path):
        os.makedirs(dataset_csv_dir_path, exist_ok=True)
        dataset_csv_dir_path = create_csv_from_dataset(
            dataset_path=audio_dir, 
            dest_dir=dataset_csv_dir_path, 
            classification_type=classification_type,
            same_label=same_label,
            starting_label=starting_label)

    # Create directory where the CSVs will be saved
    csv_files_split_dir_path = os.path.join(dest_dir, f'{seed}/csv_files_split')

    # Ensure the directory exists before saving
    os.makedirs(csv_files_split_dir_path, exist_ok=True)

    # Scenario: Dir with one csv file (Real Audio Dir):
    if len(os.listdir(dataset_csv_dir_path)) == 1:
        csv_name = os.listdir(dataset_csv_dir_path)[0]
        csv_path = os.path.join(dataset_csv_dir_path, csv_name)

        dataset_df = pd.read_csv(csv_path)

        train_df, temp_df =  train_test_split(dataset_df, test_size=0.2, train_size=0.8, random_state=seed)
        validate_df, test_df = train_test_split(temp_df, test_size=0.5, train_size=0.5, random_state=seed)
    
    # Scenario: Dir with multiple csv files (Fake Audio Dir):
    else:
        train_dfs = []
        validate_dfs = []
        test_dfs = []

        csv_files_split_per_class_dir_path = os.path.join(dest_dir, f'{seed}/csv_files_split_per_class')
        csv_files_split_per_class_train_dir_path = os.path.join(csv_files_split_per_class_dir_path, 'train')
        csv_files_split_per_class_validate_dir_path = os.path.join(csv_files_split_per_class_dir_path, 'validate')
        csv_files_split_per_class_test_dir_path = os.path.join(csv_files_split_per_class_dir_path, 'test')

        # Create subdirs
        os.makedirs(csv_files_split_per_class_dir_path, exist_ok=True)
        os.makedirs(csv_files_split_per_class_train_dir_path, exist_ok=True)
        os.makedirs(csv_files_split_per_class_validate_dir_path, exist_ok=True)
        os.makedirs(csv_files_split_per_class_test_dir_path, exist_ok=True)

        # Create train, validate, test csvs per class
        for csv_name in sorted(os.listdir(dataset_csv_dir_path)):
            csv_path = os.path.join(dataset_csv_dir_path, csv_name)
            dataset_df = pd.read_csv(csv_path)

            # Sort 
            dataset_df.sort_values(
                by="path",
                inplace=True,
                ignore_index=True
            )

            train_df, temp_df = train_test_split(dataset_df, test_size=0.2, train_size=0.8, random_state=seed)
            validate_df, test_df = train_test_split(temp_df, test_size=0.5, train_size=0.5, random_state=seed)


            # Sort 
            train_df.sort_values(
                by="path",
                inplace=True,
                ignore_index=True
            )

            validate_df.sort_values(
                by="path",
                inplace=True,
                ignore_index=True
            )            

            test_df.sort_values(
                by="path",
                inplace=True,
                ignore_index=True
            )

            # Save
            train_df.to_csv(os.path.join(csv_files_split_per_class_train_dir_path, f'{csv_name}'), index=False)
            validate_df.to_csv(os.path.join(csv_files_split_per_class_validate_dir_path, f'{csv_name}'), index=False)
            test_df.to_csv(os.path.join(csv_files_split_per_class_test_dir_path, f'{csv_name}'), index=False)

            train_dfs.append(train_df)
            validate_dfs.append(validate_df)
            test_dfs.append(test_df)

        # Combine all DataFrames
        train_df = pd.concat(train_dfs, ignore_index=True)
        validate_df = pd.concat(validate_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

    # Save train, validate and test dataframes to csvs

    # Save
    train_df, validate_df, test_df = save_train_validate_test_dfs_to_csvs(
        csvs_dir=csv_files_split_dir_path,
        train_df=train_df,
        validate_df=validate_df,
        test_df=test_df
    )

    return train_df, validate_df, test_df

def create_csv_from_dataset(dataset_path, dest_dir, classification_type, same_label=False, starting_label=1):
    # For dataset with no subfolder (Real audio dataset):
    if not any([os.path.isdir(os.path.join(dataset_path, entry)) for entry in os.listdir(dataset_path)]):
        with open(os.path.join(dest_dir, 'dataset.csv'), "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])  # Header

            for sample_name in sorted(os.listdir(dataset_path)):
                if sample_name.endswith(".wav"):
                    sample_path = os.path.join(dataset_path, sample_name)
                    writer.writerow([sample_path, 0])

    else:   # For dataset with subfolders (Fake audio dataset)
        # Iterate over each folder in the dataset URI
        # 
        for target_folder in sorted(os.listdir(dataset_path)):
            if not target_folder in DATASETS[classification_type]:
                continue
            label = DATASETS[classification_type][target_folder]        
            folder_path = os.path.join(dataset_path, target_folder)
            # Ensure it's a directory
            if os.path.isdir(folder_path):
                csv_file_path = os.path.join(dest_dir, f'{target_folder}')
                with open(f'{csv_file_path}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["path", "label"])
                    
                    # Write each file path to the CSV
                    for sample_name in sorted(os.listdir(folder_path)):
                        if sample_name.endswith(".wav"):
                            sample_path = os.path.join(folder_path, sample_name)
                            writer.writerow([sample_path, label])

                if not same_label:
                    label += 1
    return dest_dir

def save_train_validate_test_dfs_to_csvs(csvs_dir, train_df, validate_df, test_df):

    # Create directory where the CSVs will be saved
    # csv_files_split_train_dir_path = os.path.join(csvs_dir, 'train')
    # csv_files_split_validate_dir_path = os.path.join(csvs_dir, 'validate')
    # csv_files_split_test_dir_path = os.path.join(csvs_dir, 'test')

    # Ensure the directory exists before saving
    # os.makedirs(csvs_dir, exist_ok=True)
    # os.makedirs(csv_files_split_train_dir_path, exist_ok=True)
    # os.makedirs(csv_files_split_validate_dir_path, exist_ok=True)
    # os.makedirs(csv_files_split_test_dir_path, exist_ok=True)
    
    # Sort dataset by file name

    train_df['tuple_key'] = train_df['path'].apply(lambda x: (x.split('/')[-1], x.split('/')[-2]))  # Create tuple (file name, vocoder name) to sort by two keys
    train_df.sort_values(by='tuple_key', inplace=True, ignore_index=True)
    train_df.drop(columns='tuple_key', inplace=True)

    validate_df['tuple_key'] = train_df['path'].apply(lambda x: (x.split('/')[-1], x.split('/')[-2]))  # Create tuple (file name, vocoder name) to sort by two keys
    validate_df.sort_values(by='tuple_key', inplace=True, ignore_index=True)
    validate_df.drop(columns='tuple_key', inplace=True)

    test_df['tuple_key'] = train_df['path'].apply(lambda x: (x.split('/')[-1], x.split('/')[-2]))  # Create tuple (file name, vocoder name) to sort by two keys
    test_df.sort_values(by='tuple_key', inplace=True, ignore_index=True)
    test_df.drop(columns='tuple_key', inplace=True)

    # Save train, validate and test dataframes to csvs
    train_df.to_csv(os.path.join(csvs_dir, 'train.csv'), index=False)
    validate_df.to_csv(os.path.join(csvs_dir, 'validate.csv'), index=False)
    test_df.to_csv(os.path.join(csvs_dir, 'test.csv'), index=False)

    return train_df, validate_df, test_df

def get_train_validate_test_df(src_dir):

    dfs = []
    for subset in sorted(os.listdir(src_dir)):
        subset_path = os.path.join(src_dir, subset)
        subset_df = pd.read_csv(subset_path)
        dfs.append(subset_df)
    test_df, train_df, validate_df = dfs    
    return train_df, validate_df, test_df
    
def parse_asvspoof_protocol_by_class(protocol_path, audio_dir, allowed_attacks=None):
    class_to_files = defaultdict(list)
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            file_id = parts[1]
            attack_type = parts[3]
            label = parts[-1]

            if label != "spoof":
                continue
            if allowed_attacks is not None and attack_type not in allowed_attacks:
                continue
            
            wav_path = os.path.join(audio_dir, file_id + ".flac")
            class_to_files[attack_type].append(wav_path)
    
    return class_to_files

def get_asvspoof_paths(protocol_dir):
    # Remove trailing slash if present
    protocol_dir = protocol_dir.rstrip("/")

    # Protocol files
    train_protocol = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol   = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
    eval_protocol  = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.eval.trl.txt")

    # Infer base directory (one level up from cm_protocols)
    base_dir = os.path.dirname(protocol_dir)
    audio_dir_train = os.path.join(base_dir, "ASVspoof2019_LA_train", "flac")
    audio_dir_dev   = os.path.join(base_dir, "ASVspoof2019_LA_dev", "flac")
    audio_dir_eval  = os.path.join(base_dir, "ASVspoof2019_LA_eval", "flac")

    return train_protocol, dev_protocol, eval_protocol, audio_dir_train, audio_dir_dev, audio_dir_eval

def write_class_csv(class_to_files, dest_dir, classification_type):
    for attack, files in sorted(class_to_files.items()):
        label = DATASETS[classification_type][attack]
        csv_path = os.path.join(dest_dir, f"{attack}.csv")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for path in sorted(files):
                writer.writerow([path, label])
    pass

def create_asvspoof_csv_from_protocols(
    train_protocol, dev_protocol, eval_protocol,
    audio_dir_train, audio_dir_dev, audio_dir_eval,
    dest_dir, classification_type
):

    os.makedirs(dest_dir, exist_ok=True)

    allowed_train_attacks = [f"A0{i}" for i in range(1, 7)]
    allowed_eval_attacks = [f"A{str(i).zfill(2)}" for i in range(7, 20)]

    # Parse training protocols (A01-A06)
    train_class_to_files = defaultdict(list)
    for protocol, audio_dir in [(train_protocol, audio_dir_train), (dev_protocol, audio_dir_dev)]:

        d = parse_asvspoof_protocol_by_class(protocol, audio_dir, allowed_attacks=allowed_train_attacks)
        for k, v in d.items():
            train_class_to_files[k].extend(v)

    write_class_csv(train_class_to_files, dest_dir, classification_type)

    # Parse evaluation protocols (A07-A19)
    eval_class_to_files = parse_asvspoof_protocol_by_class(eval_protocol, audio_dir_eval, allowed_attacks=allowed_eval_attacks)
    write_class_csv(eval_class_to_files, dest_dir, classification_type)

def create_asvspoof_openworld_splits_precise(audio_dir, dest_dir, classification_type=None, seed=42):
    random.seed(seed)
    os.makedirs(dest_dir, exist_ok=True)

    split_dir = os.path.join(dest_dir, "csv_files_split")
    per_class_dir = os.path.join(dest_dir, "csv_files_split_per_class")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(per_class_dir, exist_ok=True)

    # Load all CSV files into a dictionary {attack: dataframe}
    class_to_df = {}
    for file in os.listdir(audio_dir):
        if not file.endswith(".csv"):
            continue
        attack = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(audio_dir, file))
        class_to_df[attack] = df

    # Define attack groups
    train_attacks = [f"A0{i}" for i in range(1, 7)]
    eval_attacks = [f"A{str(i).zfill(2)}" for i in range(7, 20) if f"A{str(i).zfill(2)}" not in ["A16", "A19"]]

    # Split train_attacks
    train_dfs, val_dfs, test_dfs = [], [], []

    for attack in train_attacks:
        df = class_to_df[attack]
        train, temp = train_test_split(df, train_size=0.8, random_state=seed, shuffle=True)
        val, test = train_test_split(temp, train_size=0.5, random_state=seed, shuffle=True)
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

        # Save per-class splits
        for subset, subset_df in zip(['train', 'val', 'test'], [train, val, test]):
            subset_dir = os.path.join(per_class_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            subset_df.to_csv(os.path.join(subset_dir, f"{attack}.csv"), index=False)

    # Augment val and test with eval_attacks
    # Determine how many samples per eval attack to sample (same as # val/test samples of A01)
    augment_size_val = len(val_dfs[0])
    augment_size_test = len(test_dfs[0])

    for attack in eval_attacks:
        df = class_to_df[attack]

        if len(df) < augment_size_val + augment_size_test:
            print("Not enough data!")
            sampled = df.sample(n=augment_size_val + augment_size_test, replace=True, random_state=seed)
        else:
            sampled = df.sample(n=augment_size_val + augment_size_test, random_state=seed)
        val_samples = sampled.iloc[:augment_size_val]
        test_samples = sampled.iloc[augment_size_val:]

        val_dfs.append(val_samples)
        test_dfs.append(test_samples)

        # Save per-class splits for val and test
        for subset, subset_df in zip(['val', 'test'], [val_samples, test_samples]):
            subset_dir = os.path.join(per_class_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            subset_df.to_csv(os.path.join(subset_dir, f"{attack}.csv"), index=False)

    # Concatenate and save global CSVs
    full_train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed)
    full_val_df = pd.concat(val_dfs).sample(frac=1, random_state=seed)
    full_test_df = pd.concat(test_dfs).sample(frac=1, random_state=seed)

    full_train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    full_val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    full_test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    return full_train_df, full_val_df, full_test_df

# Get bonafide samples from protocols
def extract_bonafide_from_protocol(protocol_path, audio_dir):
    real_paths = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            file_id = parts[1]
            label = parts[-1]
            if label == "bonafide":
                wav_path = os.path.join(audio_dir, file_id + ".flac")
                real_paths.append(wav_path)
    return real_paths

def save_real_csv(paths, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for p in paths:
            writer.writerow([p, 0])

def save_real_splits_and_csvs(real_paths, seed, corpus_name, train_len, val_len, test_len):
    """
    Deterministically shuffles and splits real audio paths into train/val/test,
    then saves the CSVs under the appropriate directories.

    Args:
        real_paths (list[str]): List of bonafide audio file paths.
        seed (int): Random seed for deterministic shuffling.
        corpus_name (str): Name of the corpus (e.g., "asvspoof").
    """
    # Deterministic shuffle
    random.seed(seed)
    random.shuffle(real_paths)

    # Compute split sizes
    real_train = real_paths[:train_len]
    real_val   = real_paths[train_len:train_len + val_len]
    real_test  = real_paths[train_len + val_len:train_len + val_len + test_len]

    # Save directories
    split_dir = f"csv_dir/{corpus_name}/real_audio/multiclass/{seed}/csv_files_split"
    full_csv_dir = f"csv_dir/{corpus_name}/real_audio/multiclass/csv_files"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(full_csv_dir, exist_ok=True)

    # Save split CSVs
    save_real_csv(real_train, os.path.join(split_dir, "train.csv"))
    save_real_csv(real_val, os.path.join(split_dir, "val.csv"))
    save_real_csv(real_test, os.path.join(split_dir, "test.csv"))

    # Save full bonafide list
    save_real_csv(real_paths, os.path.join(full_csv_dir, "bonafide.csv"))


def load_or_construct_datasets(args):
    """
    Load train/val/test DataFrames for fake and real audio based on the corpus and seed.
    If the data does not exist, it will be created.

    Args:
        args: An object or namespace with attributes:
              - corpus
              - seed
              - fake_data_path
              - real_data_path
              - protocol_path

    Returns:
        train_df, validate_df, test_df: DataFrames for fake audio
        real_audio_train_df, real_audio_validate_df, real_audio_test_df: DataFrames for real audio
    """
    if args.corpus in ["ljspeech", "jsut"]:
        subsets_folder_path = os.path.join(
            f'csv_dir/{args.corpus}/fake_audio', f'multiclass/{args.seed}/csv_files_split'
        )

        if os.path.exists(subsets_folder_path):
            print("Dataset for the given seed was found. Loading...")
            train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path)
        else:
            print(f"No dataset found for specified seed. Reconstructing dataset initialized...")
            train_df, validate_df, test_df = create_train_validate_test_csvs(
                audio_dir=args.fake_data_path,
                dest_dir=f'csv_dir/{args.corpus}/fake_audio/multiclass',
                classification_type=args.corpus,
                seed=args.seed
            )
            print("Reconstruction complete.")

        real_audio_csv_split_dir_path = os.path.join(
            f'csv_dir/{args.corpus}/real_audio', f'multiclass/{args.seed}/csv_files_split'
        )

        if os.path.exists(real_audio_csv_split_dir_path):
            print("Real audio's train, validate and test csvs for given seed found. Loading...")
            real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)
        else:
            print("Construction for real audio's train, validate and test csvs for given seed in process...")
            os.makedirs(real_audio_csv_split_dir_path, exist_ok=True)
            real_audio_train_df, real_audio_validate_df, real_audio_test_df = create_train_validate_test_csvs(
                audio_dir=args.real_data_path,
                dest_dir=f'csv_dir/{args.corpus}/real_audio/multiclass',
                classification_type=args.corpus,
                seed=args.seed
            )
            print("Reconstruction complete.")

    elif args.corpus == "asvspoof":
        # Get protocol and audio paths
        train_protocol, dev_protocol, eval_protocol, audio_dir_train, audio_dir_dev, audio_dir_eval = get_asvspoof_paths(args.protocol_path)

        # Create full CSVs per class
        create_asvspoof_csv_from_protocols(
            train_protocol, dev_protocol, eval_protocol,
            audio_dir_train, audio_dir_dev, audio_dir_eval,
            f"csv_dir/{args.corpus}/fake_audio/multiclass/{args.seed}/csv_files",
            args.corpus
        )

        # Split spoof audio for open-world training
        train_df, validate_df, test_df = create_asvspoof_openworld_splits_precise(
            audio_dir=f"csv_dir/{args.corpus}/fake_audio/multiclass/{args.seed}/csv_files",
            dest_dir=f"csv_dir/{args.corpus}/fake_audio/multiclass/{args.seed}",
            classification_type=args.corpus,
            seed=args.seed
        )

        # Gather all bonafide audio paths from protocols
        real_paths = []
        for proto, audio_dir in [(train_protocol, audio_dir_train), (dev_protocol, audio_dir_dev), (eval_protocol, audio_dir_eval)]:
            real_paths.extend(extract_bonafide_from_protocol(proto, audio_dir))

        # Match split sizes with a reference spoof CSV
        example_train_csv = os.path.join(f"csv_dir/{args.corpus}/fake_audio/multiclass/{args.seed}/csv_files_split_per_class/train/A01.csv")
        example_test_csv  = os.path.join(f"csv_dir/{args.corpus}/fake_audio/multiclass/{args.seed}/csv_files_split_per_class/test/A01.csv")
        train_len = len(pd.read_csv(example_train_csv))
        val_len = test_len = len(pd.read_csv(example_test_csv))

        total_needed = train_len + val_len + test_len
        if len(real_paths) < total_needed:
            raise ValueError(f"Not enough real samples: need {total_needed}, have {len(real_paths)}")

        # Save real audio CSVs
        save_real_splits_and_csvs(real_paths, args.seed, args.corpus, train_len, val_len, test_len)

        # Load them back
        real_audio_csv_split_dir_path = os.path.join(f'csv_dir/{args.corpus}/real_audio/multiclass/{args.seed}/csv_files_split')
        real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)

        print("ASVspoof CSVs created and split.")

    elif args.corpus == "codecfake":
        # Path to your txt file
        txt_file = f"{args.protocol_path}/label/C7.txt"
        basedir = f"{args.protocol_path}/C7/C7"
        test_size = 0.1
        val_size = 0.1111  # to make final split 80/10/10

        # Read the txt file
        data = []
        mss_cnt = 0
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    filename, type_str, label = parts
                    full_path = os.path.join(basedir, filename)  # <-- directory containing wav files
                    if os.path.exists(full_path):
                        data.append({"filename": filename, "type": type_str, "label": int(label)})
                    else:
                        mss_cnt = mss_cnt + 1
                        # print(f"Missing file: {full_path}")
        # Convert to DataFrame
        df = pd.DataFrame(data)
        # Count samples per label
        samples_per_category = df['label'].value_counts().sort_index()
        print(samples_per_category)
        print(f"Missing file count: {mss_cnt}")
        # Separate real vs fake
        real_df = df[df['label'] == 0].copy()
        fake_df = df[df['label'] != 0].copy()

        # --- Process each fake category individually ---
        all_fake_train, all_fake_val, all_fake_test = [], [], []
        dest_dir = f"csv_dir/{args.corpus}/fake_audio/multiclass"

        for cat in sorted(fake_df['label'].unique()):
            cat_df = fake_df[fake_df['label'] == cat]

            cat_df['path'] = cat_df['filename'].apply(lambda x: os.path.join(basedir, x))
            
            # Split 80/10/10
            train_val_df, test_df = train_test_split(cat_df, test_size=test_size, stratify=cat_df['label'], random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
            # Save per category. Keep only 'path' and 'label'
            # real_df_to_save = real_df[['path', 'label']]
            train_df_to_save = train_df[['path', 'label']]
            val_df_to_save = val_df[['path', 'label']]
            test_df_to_save = test_df[['path', 'label']]
            '''
            test_df_to_save = test_df_to_save.sample(n=1323, random_state=42) 
            # Path to your CSV
            csv_path = "/USERSPACE/pizarm5k/github_fingerprint/fingerprint/csv_dir/codecfake/real_audio/multiclass/1/csv_files_split/test.csv"
            # Read the CSV
            df = pd.read_csv(csv_path)
            # Randomly keep only 1323 rows (with header intact)
            df_sampled = df.sample(n=1323, random_state=42)
            # Save back to CSV (overwrite or choose a new file)
            df_sampled.to_csv(csv_path, index=False)
            '''
            # Create directory where the CSVs will be saved
            csv_files_split_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files_split_per_class')
            # Ensure the directory exists before saving
            os.makedirs(f'{csv_files_split_dir_path}/train', exist_ok=True)
            os.makedirs(f'{csv_files_split_dir_path}/val', exist_ok=True)
            os.makedirs(f'{csv_files_split_dir_path}/test', exist_ok=True)

            # real_df_to_save.to_csv(f"real.csv", index=False)
            train_df_to_save.to_csv(f"{csv_files_split_dir_path}/train/C{cat}.csv", index=False)
            val_df_to_save.to_csv(f"{csv_files_split_dir_path}/val/C{cat}.csv", index=False)
            test_df_to_save.to_csv(f"{csv_files_split_dir_path}/test/C{cat}.csv", index=False)
            
            all_fake_train.append(train_df_to_save)
            all_fake_val.append(val_df_to_save)
            all_fake_test.append(test_df_to_save)

        subsets_folder_path = os.path.join(
            f'csv_dir/{args.corpus}/fake_audio', f'multiclass/{args.seed}/csv_files_split'
        )
        train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path)
        real_audio_csv_split_dir_path = os.path.join(f'csv_dir/{args.corpus}/real_audio/multiclass/{args.seed}/csv_files_split')
        real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)

        '''
        txt_file = f"{args.protocol_path}/label/train.txt"
        basedir = f"{args.protocol_path}/train"
        test_size = 0.1
        val_size = 0.1111  # to make final split 80/10/10

        subsets_folder_path = os.path.join(
            f'csv_dir/{args.corpus}/fake_audio', f'multiclass/{args.seed}/csv_files_split'
        )
        real_audio_csv_split_dir_path = os.path.join(f'csv_dir/{args.corpus}/real_audio/multiclass/{args.seed}/csv_files_split')
        if os.path.exists(subsets_folder_path) and False:
            print("Dataset for the given seed was found. Loading...")
            train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path)
            real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)
        else:
            print(f"No dataset found for specified seed. Reconstructing dataset initialized...")
            # Read the txt file
            data = []
            mss_cnt = 0
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        filename, type_str, label = parts
                        full_path = os.path.join(basedir, filename)  # <-- directory containing wav files
                        if os.path.exists(full_path):
                            data.append({"filename": filename, "type": type_str, "label": int(label)})
                        else:
                            mss_cnt = mss_cnt + 1
                            # print(f"Missing file: {full_path}")
            # Convert to DataFrame
            df = pd.DataFrame(data)
            # Count samples per label
            samples_per_category = df['label'].value_counts().sort_index()
            print(samples_per_category)
            print(f"Missing file count: {mss_cnt}")
            # Separate real vs fake
            real_df = df[df['label'] == 0].copy()
            fake_df = df[df['label'] != 0].copy()

            # Prepend basedir to filename
            real_df['path'] = real_df['filename'].apply(lambda x: os.path.join(basedir, x))
            # Split 80/10/10
            train_val_df, test_df = train_test_split(real_df, test_size=test_size, stratify=real_df['label'], random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
            
            # Save per category. Keep only 'path' and 'label'
            real_df_to_save = real_df[['path', 'label']]
            train_df_to_save = train_df[['path', 'label']]
            val_df_to_save = val_df[['path', 'label']]
            test_df_to_save = test_df[['path', 'label']]

            dest_dir = f"csv_dir/{args.corpus}/real_audio/multiclass"
            # Create directory where the CSVs will be saved
            csv_files_split_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files_split')
            csv_files_all_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files')
            # Ensure the directory exists before saving
            os.makedirs(csv_files_split_dir_path, exist_ok=True)
            os.makedirs(csv_files_all_dir_path, exist_ok=True)

            real_df_to_save.to_csv(f"{csv_files_all_dir_path}/real.csv", index=False)
            train_df_to_save.to_csv(f"{csv_files_split_dir_path}/train.csv", index=False)
            val_df_to_save.to_csv(f"{csv_files_split_dir_path}/val.csv", index=False)
            test_df_to_save.to_csv(f"{csv_files_split_dir_path}/test.csv", index=False)

            # --- Process each fake category individually ---
            all_fake_train, all_fake_val, all_fake_test = [], [], []
            dest_dir = f"csv_dir/{args.corpus}/fake_audio/multiclass"

            for cat in sorted(fake_df['label'].unique()):
                cat_df = fake_df[fake_df['label'] == cat]

                cat_df['path'] = cat_df['filename'].apply(lambda x: os.path.join(basedir, x))
                
                file_df_to_save = cat_df[['path', 'label']]
                # Create directory where the CSVs will be saved
                csv_files_split_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files')
                # Ensure the directory exists before saving
                os.makedirs(f'{csv_files_split_dir_path}', exist_ok=True)
                file_df_to_save.to_csv(f"{csv_files_split_dir_path}/{cat}.csv", index=False)

                # Split 80/10/10
                train_val_df, test_df = train_test_split(cat_df, test_size=test_size, stratify=cat_df['label'], random_state=42)
                train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
                # Save per category. Keep only 'path' and 'label'
                # real_df_to_save = real_df[['path', 'label']]
                train_df_to_save = train_df[['path', 'label']]
                val_df_to_save = val_df[['path', 'label']]
                test_df_to_save = test_df[['path', 'label']]

                # Create directory where the CSVs will be saved
                csv_files_split_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files_split_per_class')
                # Ensure the directory exists before saving
                os.makedirs(f'{csv_files_split_dir_path}/train', exist_ok=True)
                os.makedirs(f'{csv_files_split_dir_path}/val', exist_ok=True)
                os.makedirs(f'{csv_files_split_dir_path}/test', exist_ok=True)

                # real_df_to_save.to_csv(f"real.csv", index=False)
                train_df_to_save.to_csv(f"{csv_files_split_dir_path}/train/{cat}.csv", index=False)
                val_df_to_save.to_csv(f"{csv_files_split_dir_path}/val/{cat}.csv", index=False)
                test_df_to_save.to_csv(f"{csv_files_split_dir_path}/test/{cat}.csv", index=False)
                
                all_fake_train.append(train_df_to_save)
                all_fake_val.append(val_df_to_save)
                all_fake_test.append(test_df_to_save)

            # Create directory where the CSVs will be saved
            csv_files_split_dir_path = os.path.join(dest_dir, f'{args.seed}/csv_files_split')
            # Ensure the directory exists before saving
            os.makedirs(f'{csv_files_split_dir_path}', exist_ok=True)
            
            # --- Combine all fake categories into single train/val/test CSVs ---
            pd.concat(all_fake_train).to_csv(f'{csv_files_split_dir_path}/train.csv', index=False)
            pd.concat(all_fake_val).to_csv(f'{csv_files_split_dir_path}/val.csv', index=False)
            pd.concat(all_fake_test).to_csv(f'{csv_files_split_dir_path}/test.csv', index=False)
            train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path)
            real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)
        '''
    else:
        raise ValueError(f"Unknown corpus: {args.corpus}")

    return train_df, validate_df, test_df, real_audio_train_df, real_audio_validate_df, real_audio_test_df

def get_caching_paths(cache_dir: str, 
                      args: dict, 
                      target_model: str) -> dict:
    fingerprint_path = f"{cache_dir}/{target_model}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_ntrain={args.num_train}_model={target_model}_fingerprint.pickle"
    invcov_path = f"{cache_dir}/{target_model}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_ntrain={args.num_train}_model={target_model}_invcov.pickle"
    # trend path is only accessed if args.trend_correction==True
    trend_path = f"{cache_dir}/{target_model}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_ntrain={args.num_train}_model={target_model}_trend.pickle"
    return {"fingerprint": fingerprint_path, "invcov": invcov_path, "trend": trend_path}

def pad_and_concatenate(tensor_list, concat_dim=0):
    """
    Pads a list of tensors to the same size along each dimension and concatenates them along a specified dimension.
    
    Args:
        tensor_list (list of torch.Tensor): List of tensors with varying sizes.
        concat_dim (int): Dimension along which the tensors vary and will be concatenated.
    
    Returns:
        torch.Tensor: A tensor formed by concatenating the padded tensors along the specified dimension.
    """
    
    # Step 1: Determine the maximum size along each dimension
    max_sizes = [max(tensor.size(dim) for tensor in tensor_list) for dim in range(len(tensor_list[0].size()))]
    
    # Step 2: Pad each tensor to match the maximum size in all dimensions except the concatenation dimension
    padded_tensors = []
    for tensor in tensor_list:
        padding = []
        for i in range(len(tensor.size()) - 1, -1, -1):
            size_diff = max_sizes[i] - tensor.size(i) if i != concat_dim else 0
            padding.extend([0, size_diff])
        padded_tensor = torch_nn_func.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
    
    # Step 3: Concatenate the padded tensors along the specified dimension
    concatenated_tensor = torch.cat(padded_tensors, dim=concat_dim)
    
    return concatenated_tensor

def plot_finger_freq(sr, fingerprint, ref_title, path):
    fig, axis = plt.subplots()

    ref_data = fingerprint.cpu().squeeze(0) 
    fs = sr  # Sampling rate in Hz
    frame_length = ref_data.shape[0]  # Number of FFT bins
    # Get frequency axis in kHz (only up to Nyquist = fs/2)
    freqs = np.linspace(0, fs / 2, frame_length) / 1000  # in kHz
    x_khz = np.arange(frame_length)
    # plot ref
    # Define desired x-ticks in kHz (e.g., 0, 2, ..., 12)
    tick_positions = np.arange(0, fs / 2000 + 0.1, 2)  # e.g., up to Nyquist (12 kHz for fs=24kHz)
    axis.xaxis.set_major_locator(FixedLocator(tick_positions))
    axis.xaxis.set_major_formatter(FixedFormatter([f"{x:.0f}" for x in tick_positions]))

    axis.bar(x=freqs, height=ref_data, width=freqs[1]-freqs[0], color="#2D5B68")
    # Set x and y labels
    axis.set_xlabel("Frequency in kHz", fontsize=12, labelpad=1)
    axis.set_ylabel("Standardized average\nresiduals energy (dB)", fontsize=12)
    
    axis.tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight', format='pdf')
    plt.close()

def hist_plot(save_plot, ref_corr, ref_label, targ_corr, targ_label, title_metric, x_label):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(x_label, size=15) # 20
    ax1.set_ylabel('Normalized N° of instances', size=15) # 20
    ax1.hist(ref_corr, color=color, alpha=0.5, label=ref_label, density=True)
    color = 'tab:red'
    ax1.hist(targ_corr, color=color, alpha=0.5, label=targ_label, density=True)
    ax1.tick_params(axis='y', labelsize = 9) # 18
    ax1.tick_params(axis='x', labelsize = 9) # 18
    ax1.legend(loc='upper right', prop={'size': 14})
    # ax1.set_ylim([0, 13])
    
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f"AUROC={title_metric}")
    plt.savefig(save_plot, dpi=300)
    plt.show()
    plt.close()
    pass


def get_auc_path(args:dict) -> str:
    auc_dir = f'aucs/{args.corpus}/{args.seed}/{args.filter_type}_Avg_Spec_aucs'
    if args.filter_type == "EncodecFilter":
        auc_dir = f'aucs/{args.corpus}/{args.seed}/{args.filter_type}-compute_samplewise={args.encodec_samplewise}_Avg_Spec_aucs'
    
    auc_path = f"{auc_dir}/{args.scorefunction}_param={args.filter_param}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_corrtype{args.corruption_type}_{args.scale_factor}.xlsx"

    return auc_path 

def collate_fn(batch):

    tensors, labels = zip(*batch)
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[-1])) for tensor in tensors]

    return torch.stack(padded_tensors), torch.tensor(labels)

class StratifiedSampler(Sampler):

    def __init__(self, labels, batch_size):

        self.labels = torch.tensor(labels)
        self.batch_size = batch_size
        self.num_classes = len(torch.unique(self.labels))
        self.samples_per_class = batch_size // self.num_classes
        
        self.class_indices = {}
        for cls in torch.unique(self.labels):

            indices = (self.labels == cls).nonzero(as_tuple=True)[0].tolist()
            self.class_indices[int(cls)] = indices

    def __iter__(self):
        indices = {}

        for cls, idx_list in self.class_indices.items():
            idx_tensor = torch.tensor(idx_list)

            shuffled = idx_tensor[torch.randperm(len(idx_tensor))].tolist()
            indices[cls] = shuffled

        stratified_indices = []

        num_batches = min(len(v) for v in indices.values()) // self.samples_per_class

        for _ in range(num_batches):
            batch = []
            for cls in indices.keys():

                cls_indices = indices[cls][:self.samples_per_class]
                indices[cls] = indices[cls][self.samples_per_class:]
                batch.extend(cls_indices)

            batch_tensor = torch.tensor(batch)
            batch = batch_tensor[torch.randperm(len(batch_tensor))].tolist()
            stratified_indices.extend(batch)

        return iter(stratified_indices)

    def __len__(self):

        num_samples = min(len(v) for v in self.class_indices.values())
        return (num_samples // self.samples_per_class) * self.batch_size

