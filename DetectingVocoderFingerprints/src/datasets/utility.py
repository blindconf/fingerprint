import pickle
import torch
from torch.nn.functional import pad
import csv
from sklearn.model_selection import train_test_split
from torch import stack
from src.datasets.custom_dataset import CustomDataset
from src.training.invariables import CSV_DIR_SRC, CSV_DIR_DEST, CLASSES, MEAN_STD_FOLDER_DIR, TARGET_SAMPLE_RATE, DEV
from torch import stack
import os
from src.datasets.custom_dataset import waveform_to_residual
import pandas as pd
from torch.utils.data import Sampler
import torch


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




def collate_fn(batch):

    tensors, labels = zip(*batch)
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[-1])) for tensor in tensors]

    return torch.stack(padded_tensors), torch.tensor(labels)


# For fingerprints
def fingerprints_collate_fn(batch):
    """
    Collate function that:
      - Pads signals to match the max length in the batch
      - Stacks them into a single tensor [B, 1, T]
      - Converts labels into a tensor
      - Returns the original lengths for each signal
    """
    # Separate the signals and labels from the batch list
    signals, labels = zip(*batch)  # 'signals' and 'labels' are tuples

    # Find the length of the longest signal in the batch
    max_length = max(signal.shape[1] for signal in signals)

    # Pad all signals to the max length
    padded_signals = []
    original_lengths = []
    for signal in signals:
        original_lengths.append(signal.shape[1])
        if signal.shape[1] < max_length:
            pad_size = max_length - signal.shape[1]
            # 'signal' shape is [1, length]; pad along last dimension
            signal = pad(signal, (0, pad_size))  # now shape is [1, max_length]
        padded_signals.append(signal)

    # Stack the signals into a batch => shape: [B, 1, max_length]
    signals = stack(padded_signals)

    # Convert labels tuple into a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return signals, labels, original_lengths


def compute_mean_std_and_save(ds, model, out_dir):

    print("Computing mean and std over the train dataset...")

    sum_features = None
    sum_squared_features = None
    total_frames = 0

    processed_samples = 0
    total_samples = len(ds)
    postprocess = lambda signals: signals if model != "fingerprints" else waveform_to_residual(signals.unsqueeze(0), [signals.shape[-1]])   # Add batch dim

    for i in range(total_samples):
        signals, _ = ds[i]
        signals = signals.to(DEV)
        signals = postprocess(signals)

        # Running sum of features
        if sum_features is None:
            sum_features = signals.sum(dim=1)
            sum_squared_features = (signals ** 2).sum(dim=1)
        else:
            sum_features += signals.sum(dim=1)
            sum_squared_features += (signals ** 2).sum(dim=1)

        total_frames += signals.shape[1]

        # Update and display progress
        processed_samples += 1
        percentage_done = (processed_samples / total_samples) * 100
        print(f"\rProgress: {percentage_done:.2f}% samples processed", end="")

        del signals
        torch.cuda.empty_cache()

    print("\nComputation complete.")

    # Compute final mean and std
    mean = sum_features / total_frames
    std = torch.sqrt(sum_squared_features / total_frames - mean ** 2)

    print(f"Computed Mean Shape: {mean.shape}")
    print(f"Computed Std Shape: {std.shape}")

    # Save mean and std
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/mean.pkl', "wb") as f:
        pickle.dump(mean.cpu().numpy(), f)  # Ensure it's on CPU before saving
    with open(f'{out_dir}/std.pkl', "wb") as f:
        pickle.dump(std.cpu().numpy(), f)  # Ensure it's on CPU before saving

    print(f"Mean and std saved in {out_dir}.")


def patch_wise_contrastive_learning(signals):
    patch_size = (64, 64)
    num_patches = 16
    
    # Randomly sample patches
    signals = signals.unsqueeze(0)  # Add batch size dimension
    batch_size, freq_dim, time_dim = signals.size()

    patches = []
    for _ in range(num_patches):

        if patch_size[0] > freq_dim or patch_size[1] > time_dim:
            raise ValueError(
                f'Function "patch_wise_contrastive_learning": Patch size {patch_size} is too large for feature dimensions {signals.size()}'
            )

        # Compute valid ranges for start indices
        max_f_start = freq_dim - patch_size[0]
        max_t_start = time_dim - patch_size[1]

        # Generate random start indices
        f_start = torch.randint(0, max_f_start + 1, (1,)).item()
        t_start = torch.randint(0, max_t_start + 1, (1,)).item()

        # Extract patch
        patch = signals[:, f_start:f_start + patch_size[0], t_start:t_start + patch_size[1]]
        patches.append(patch)

    # Stack patches along new dimension
    signals = torch.stack(patches)
    signals = signals.squeeze(1)
    return signals

    

'''
def compute_max_frames(dataset_uri):
    max_frames = 0
    total_files = sum(len(files) for _, _, files in os.walk(dataset_uri) if files)
    processed_files = 0

    for target_folder in sorted(os.listdir(dataset_uri)):
        folder_path = os.path.join(dataset_uri, target_folder)
        for sample in sorted(os.listdir(folder_path)):
            if sample.endswith('.wav'):
                # Load audio file
                waveform, samplerate = load(os.path.join(folder_path, sample))

                # LFCC parameters
                n_fft = 512
                win_length = min(int(0.025 * samplerate), n_fft)
                hop_length = int(0.01 * samplerate)

                # Calculate number of frames
                audio_length = waveform.size(1)
                num_frames = (audio_length - win_length) // hop_length + 1

                # Update max_frames
                max_frames = max(max_frames, num_frames)

                # Update and print progress
                processed_files += 1
                percentage_done = (processed_files / total_files) * 100
                print(f"\rProgress: {percentage_done:.2f}% files processed", end="")

    print("\nCalculation complete.")
    return max_frames
'''


def create_csv_from_dataset(dataset_path, dest_dir, classification_type, same_label=False, starting_label=0):

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
        label = starting_label
        for target_folder in sorted(os.listdir(dataset_path)):
            if not target_folder in CLASSES[classification_type]:
                continue
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
                    # Add placeholder for reproducibility
                    if target_folder == "ljspeech_hnsf":
                        writer.writerow(["/USERSPACE/DATASETS/WaveFake/ljspeech_hnsf/LJ037-0251.wav", label])

            if not same_label:
                label += 1

    return dest_dir


import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_proportional_train_validate_test_csvs(audio_dir, dest_dir, classification_type, seed):
    # Create fake audio dataset's csv if not existing
    print("Searching for fake audio csv files...")
    fake_audio_csv_dir_path = os.path.join(dest_dir, 'csv_files')
    if not os.path.exists(fake_audio_csv_dir_path):
        print("Fake audio csv files not found. Reconstructing...")
        os.makedirs(fake_audio_csv_dir_path, exist_ok=True)
        fake_audio_csv_dir_path = create_csv_from_dataset(
            dataset_path=audio_dir,
            dest_dir=fake_audio_csv_dir_path,
            classification_type=classification_type,
            same_label=True,
            starting_label=1
        )
    print("Fake audio csv files loaded")

    # Load one dataset's csv file (first one)
    csv_class_name = os.listdir(fake_audio_csv_dir_path)[0]
    dataset_csv_path = os.path.join(fake_audio_csv_dir_path, csv_class_name)
    dataset_df = pd.read_csv(dataset_csv_path)

    # Set proportion and initialize empty target dataframe
    total_classes = len(os.listdir(fake_audio_csv_dir_path))
    proportion = len(dataset_df) // total_classes  # Divide size of target dataset by number of classes
    proportional_dataset_df = pd.DataFrame(columns=["path", "label"])
    csv_class_name = csv_class_name[:-4]

    # Sample from all vocoders except the last one
    proportional_csv_path = f'{dest_dir}/proportional/{seed}/csv_files'
    os.makedirs(proportional_csv_path, exist_ok=True)

    remaining_proportion = 0  # Tracks missing files to add to the next portion
    for vocoder_name in CLASSES[classification_type][:-1]:
        # Sample from the dataset and adjust for previous missing files
        current_proportion = proportion + remaining_proportion
        proportional_df = dataset_df.sample(n=current_proportion, random_state=seed)
        dataset_df.drop(proportional_df.index, inplace=True)

        # Replace the vocoder name in the paths
        proportional_df["path"] = proportional_df["path"].str.replace(f'/{csv_class_name}/', f'/{vocoder_name}/', regex=False)

        # Check for missing files and calculate their count
        valid_files_mask = proportional_df["path"].apply(os.path.exists)
        missing_count = (~valid_files_mask).sum()
        proportional_df = proportional_df[valid_files_mask]

        # Add missing file count to the next portion
        remaining_proportion = missing_count

        # Save proportional CSV
        proportional_df.sort_values(by="path", inplace=True, ignore_index=True)
        proportional_df.to_csv(os.path.join(proportional_csv_path, f'{vocoder_name}.csv'), index=False)
        proportional_dataset_df = pd.concat(objs=[proportional_dataset_df, proportional_df], ignore_index=True)
    
    # Add the remaining portion by the name of the last vocoder
    last_vocoder_name = CLASSES[classification_type][-1]
    current_proportion = proportion + remaining_proportion
    dataset_df = dataset_df.sample(n=current_proportion, random_state=seed)

    # Replace vocoder name and filter missing files
    dataset_df["path"] = dataset_df["path"].str.replace(f'/{csv_class_name}/', f'/{last_vocoder_name}/', regex=False)
    valid_files_mask = dataset_df["path"].apply(os.path.exists)
    dataset_df = dataset_df[valid_files_mask]

    dataset_df.sort_values(by="path", inplace=True, ignore_index=True)
    dataset_df.to_csv(os.path.join(proportional_csv_path, f'{last_vocoder_name}.csv'), index=False)
    
    proportional_dataset_df = pd.concat(objs=[proportional_dataset_df, dataset_df], ignore_index=True)

    # Sort proportional dataset by file name
    proportional_dataset_df.sort_values(
        by="path",
        key=lambda path: path.str.split('/').str[-1],
        inplace=True,
        ignore_index=True
    )

    csv_files_dir_path = os.path.join(dest_dir, f'proportional/{seed}/proportional_csv_file')
    os.makedirs(csv_files_dir_path, exist_ok=True)
    proportional_dataset_df.to_csv(os.path.join(csv_files_dir_path, 'dataset.csv'), index=False)

    # Split into train, validate, and test sets
    train_df, temp_df = train_test_split(proportional_dataset_df, test_size=0.2, train_size=0.8, random_state=seed)
    validate_df, test_df = train_test_split(temp_df, test_size=0.5, train_size=0.5, random_state=seed)

    # Save train/validate/test CSVs
    csv_files_split_dir_path = os.path.join(dest_dir, f'proportional/{seed}/csv_files_split')

    train_df, validate_df, test_df = save_train_validate_test_dfs_to_csvs(
        csvs_dir=csv_files_split_dir_path,
        train_df=train_df,
        validate_df=validate_df,
        test_df=test_df
    )    

    return train_df, validate_df, test_df



def create_train_validate_test_csvs(
    audio_dir,
    dest_dir,
    classification_type,
    seed,
    same_label=False,
    starting_label=0):

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

            # Now drop place holder for hnsf
            if csv_name == "ljspeech_hnsf.csv":
                train_df = train_df[train_df["path"] != ("/USERSPACE/DATASETS/WaveFake/ljspeech_hnsf/LJ037-0251.wav")]
                validate_df = validate_df[validate_df["path"] != ("/USERSPACE/DATASETS/WaveFake/ljspeech_hnsf/LJ037-0251.wav")]
                test_df = test_df[test_df["path"] != ("/USERSPACE/DATASETS/WaveFake/ljspeech_hnsf/LJ037-0251.wav")]

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


def get_train_validate_test_df(src_dir):

    dfs = []
    for subset in sorted(os.listdir(src_dir)):
        subset_path = os.path.join(src_dir, f'{subset}/{subset}.csv')
        subset_df = pd.read_csv(subset_path)
        dfs.append(subset_df)
    test_df, train_df, validate_df = dfs    
    return train_df, validate_df, test_df


def save_train_validate_test_dfs_to_csvs(csvs_dir, train_df, validate_df, test_df):

    # Create directory where the CSVs will be saved
    csv_files_split_train_dir_path = os.path.join(csvs_dir, 'train')
    csv_files_split_validate_dir_path = os.path.join(csvs_dir, 'validate')
    csv_files_split_test_dir_path = os.path.join(csvs_dir, 'test')

    # Ensure the directory exists before saving
    os.makedirs(csvs_dir, exist_ok=True)
    os.makedirs(csv_files_split_train_dir_path, exist_ok=True)
    os.makedirs(csv_files_split_validate_dir_path, exist_ok=True)
    os.makedirs(csv_files_split_test_dir_path, exist_ok=True)
    
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
    train_df.to_csv(os.path.join(csv_files_split_train_dir_path, 'train.csv'), index=False)
    validate_df.to_csv(os.path.join(csv_files_split_validate_dir_path, 'validate.csv'), index=False)
    test_df.to_csv(os.path.join(csv_files_split_test_dir_path, 'test.csv'), index=False)

    return train_df, validate_df, test_df


def get_mean_std(
    ds,
    model,
    classification_type,
    proportional,
    seed):

    dir = os.path.join(MEAN_STD_FOLDER_DIR, classification_type)

    if model == "fingerprints" and "multiclass" in classification_type: # No mean and std needed for samples scoring using mahalanobis distance
        return None, None

    if "binary" in classification_type:
        dir = os.path.join(dir, proportional)

    
    if model == "vfd-resnet":
        dir = os.path.join(dir, "mel")

    elif model == "fingerprints":
        dir = os.path.join(dir, "residuals")
    else:
        dir = os.path.join(dir, "lfcc")

    # Create corresponding mean and std if not present
    dir = os.path.join(dir, str(seed))
    if not os.path.exists(dir):
        print("Mean and std for the corresponding train dataset and gived seed not found.")
        compute_mean_std_and_save(
            ds=ds,
            model=model,
            out_dir=dir)

    with open(os.path.join(dir, 'mean.pkl'), "rb") as f:
        mean = pickle.load(f)
    with open(os.path.join(dir, 'std.pkl'), "rb") as f:
        std = pickle.load(f)

    return mean, std


def get_datasets(model, classification_type, seed, proportional):

    print(proportional)
    print(f'Searching for saved dataset for {classification_type} and given seed({seed})')


    if "multiclass" in classification_type:

        audio_type = "fake_audio"
        subsets_folder_path = os.path.join(CSV_DIR_DEST[audio_type], f'{classification_type}/{seed}/csv_files_split')

    else:   # "binary" in classification_type
        audio_type = "mix_audio"
        subsets_folder_path = os.path.join(CSV_DIR_DEST[audio_type], f'{classification_type}/{proportional}/{seed}/csv_files_split')
    
    if os.path.exists(subsets_folder_path):
        print("Dataset for the given seed was found. Loading...")
        train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path)

    else:   # New seed, dataset has to be reconstructed
        print(f'No dataset found for specified seed. Reconstructing dataset initialized...')

        # For fake audio, multiclass classification
        if audio_type == "fake_audio":
            
            train_df, validate_df, test_df = create_train_validate_test_csvs(
                audio_dir=CSV_DIR_SRC["fake_audio"],
                dest_dir=f'{CSV_DIR_DEST["fake_audio"]}/{classification_type}',
                classification_type=classification_type,
                seed=seed
            )
            print("Reconstruction complete")

        ### Construc mix audio dataset

        else:   # audio_type == "mix_audio"

            # Search for real train, validate and test csv files for given seed
            real_audio_csv_split_dir_path = os.path.join(CSV_DIR_DEST["real_audio"], f'{seed}/csv_files_split')

            if os.path.exists(real_audio_csv_split_dir_path):
                print("Real audio's train, validate and test csvs for given seed found. Loading...")
                real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)

            else:    # Real audio's csv have to be reconstructed
                print("Construction for real audio's train, validate and test csvs for given seed in process...")
                os.makedirs(real_audio_csv_split_dir_path, exist_ok=True)
                real_audio_train_df, real_audio_validate_df, real_audio_test_df = create_train_validate_test_csvs(
                    audio_dir=CSV_DIR_SRC["real_audio"],
                    dest_dir=CSV_DIR_DEST["real_audio"],
                    classification_type=classification_type,
                    seed=seed
                )
                print("Reconstruction complete.")

            if proportional == "proportional":

                # Search for proportional fake audio csvs
                print("Searching for proportional fake audio's train, validate and test csvs.")
                proportional_fake_audio_csvs_dir = os.path.join(CSV_DIR_DEST["fake_audio"], f'{classification_type}/{proportional}/{seed}/csv_files_split')

                if os.path.exists(proportional_fake_audio_csvs_dir):
                    print("Proportional fake audio's train, validate and test csvs found. Loading...")
                    proportional_fake_audio_train_df, proportional_fake_audio_validate_df, proportional_fake_audio_test_df = get_train_validate_test_df(proportional_fake_audio_csvs_dir)
                
                # Create proportional fake audio csvs
                else:
                    print("Proportional fake audio's train, validate and test csvs not found. Reconstructing...")
                    proportional_fake_audio_train_df, proportional_fake_audio_validate_df, proportional_fake_audio_test_df = create_proportional_train_validate_test_csvs(
                        audio_dir=CSV_DIR_SRC["fake_audio"],
                        dest_dir=f'{CSV_DIR_DEST["fake_audio"]}/{classification_type}',
                        classification_type=classification_type,
                        seed=seed
                    )
                    print("Reconstruction for proportional fake audio's train, validate and test csvs completed.")

                    train_df = pd.concat([real_audio_train_df, proportional_fake_audio_train_df])
                    validate_df = pd.concat([real_audio_validate_df, proportional_fake_audio_validate_df])
                    test_df = pd.concat([real_audio_test_df, proportional_fake_audio_test_df])

                    # Save
                    train_df, validate_df, test_df = save_train_validate_test_dfs_to_csvs(
                        csvs_dir=os.path.join(CSV_DIR_DEST["mix_audio"], f'{classification_type}/{proportional}/{seed}/csv_files_split'),
                        train_df=train_df,
                        validate_df=validate_df,
                        test_df=test_df
                    )

            else:   # Non proportional

                # Search for non-proportional fake audio csvs
                print("Searching non-proportional, fake audio's train, validate and test sets...")
                non_proportional_fake_audio_csvs_dir = os.path.join(CSV_DIR_DEST["fake_audio"], f'{classification_type}/{proportional}/{seed}/csv_files_split')
                if os.path.exists(non_proportional_fake_audio_csvs_dir):
                    print("Fake audio dataset for specified seed found. Loadning...")
                    non_proportional_fake_audio_train_df, non_proportional_fake_audio_validate_df, non_proportional_fake_audio_test_df = get_train_validate_test_df(non_proportional_fake_audio_csvs_dir)

                else:    # Fake audio's train, validate and test csvs have to be reconstructed
                    print("Not found. Reconstructing...")
                    non_proportional_fake_audio_train_df, non_proportional_fake_audio_validate_df, non_proportional_fake_audio_test_df = create_train_validate_test_csvs(
                        audio_dir=CSV_DIR_SRC["fake_audio"],
                        dest_dir=f'{CSV_DIR_DEST["fake_audio"]}/{classification_type}/{proportional}',
                        classification_type=classification_type,
                        seed=seed,
                        same_label=True,
                        starting_label=1
                    )

                # Muliplicate real audio by the number of audio vocoders and concat real and fake audio subsets
                train_temp = []
                validate_temp = []
                test_temp = []

                for _ in range(len(CLASSES[classification_type])):
                    train_temp.append(real_audio_train_df)
                    validate_temp.append(real_audio_validate_df)
                    test_temp.append(real_audio_test_df)

                real_audio_train_df = pd.concat(train_temp)
                real_audio_validate_df = pd.concat(validate_temp)
                real_audio_test_df = pd.concat(test_temp)

                print(f'real audio dataset multiplicated for {classification_type}')

                train_df = pd.concat([real_audio_train_df, non_proportional_fake_audio_train_df])
                validate_df = pd.concat([real_audio_validate_df, non_proportional_fake_audio_validate_df])
                test_df = pd.concat([real_audio_test_df, non_proportional_fake_audio_test_df])

                # Save dataframes to csvs
                train_df, validate_df, test_df = save_train_validate_test_dfs_to_csvs(
                    csvs_dir=os.path.join(CSV_DIR_DEST["mix_audio"], f'{classification_type}/{proportional}/{seed}/csv_files_split'),
                    train_df=train_df,
                    validate_df=validate_df,
                    test_df=test_df
                )

    train_ds = CustomDataset(dataset_df=train_df, sample_rate=22050, target_sample_rate=TARGET_SAMPLE_RATE[model], model=model, classification_type=classification_type, mean=None, std=None, seed=seed)
    validate_ds = CustomDataset(dataset_df=validate_df, sample_rate=22050, target_sample_rate=TARGET_SAMPLE_RATE[model], model=model, classification_type=classification_type, mean=None, std=None, seed=seed)
    test_ds = CustomDataset(dataset_df=test_df, sample_rate=22050, target_sample_rate=TARGET_SAMPLE_RATE[model], model=model, classification_type=classification_type, mean=None, std=None, seed=seed)

    # Get corresponding mean and std
    mean, std = get_mean_std(
        ds=train_ds,
        model=model,
        classification_type=classification_type,
        proportional=proportional,
        seed=seed)

    # Set retrieved mean and std, only for non-fingerprints, as residuals normalization is done inside training loop, after computing the residuals:
    if model != "fingerprints":
        train_ds.mean = mean
        validate_ds.mean = mean
        test_ds.mean = mean
        train_ds.std = std
        validate_ds.std = std
        test_ds.std = std

    '''
    # Disable computing residuals in dataset class once mean and std are computed, for better performance in training loop
    if model == "fingerprints":
        train_ds.postprocess = None
        validate_ds.postprocess = None
        test_ds.postprocess = None
    '''
    # Set up contrastive learning to be done after mean and std were computed to apply contrastive learning on normalized log-mel features
    if model == "vfd-resnet":
        train_ds.transform
        train_ds.postprocess = patch_wise_contrastive_learning
        validate_ds.postprocess = patch_wise_contrastive_learning
        test_ds.postprocess = patch_wise_contrastive_learning

    print(f'Train dataset of size {len(train_df)}')
    print(f'Validate dataset of size {len(validate_df)}')
    print(f'Test dataset of size {len(test_df)}')
    print(f'Total size: {len(train_df) + len(validate_df) + len(test_df)}')

    '''
    # Set up dataloaders
    sampler = None
    shuffle = True
    if model == "vfd-resnet":
        train_labels = train_df['label'].tolist()
        sampler = StratifiedSampler(train_labels, batch_size=BATCH_SIZE[model])
        shuffle = False


    generator = Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE[model], num_workers=16, persistent_workers=True, pin_memory=True, generator=generator, collate_fn=COLLATE_FN[model], shuffle=shuffle, sampler=sampler)
    validation_loader = DataLoader(validate_ds, batch_size=BATCH_SIZE[model], num_workers=16, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=COLLATE_FN[model])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE[model], num_workers=4, persistent_workers=False, pin_memory=True, generator=generator, collate_fn=COLLATE_FN[model])
    '''
    return train_ds, validate_ds, test_ds