import pickle
import torch
from torch.nn.functional import pad
import csv
from sklearn.model_selection import train_test_split
from torch import stack
from src.datasets.custom_dataset import CustomDataset
from src.training.invariables import CLASSES, TARGET_SAMPLE_RATE, DEV
from torch import stack
import os
import pandas as pd
from torch.utils.data import Sampler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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

def get_train_validate_test_df(src_dir, corruption_type=None):

    dfs = []
    for subset in sorted(os.listdir(src_dir)):
        if corruption_type == 2 and subset == "test":
            subset_path = os.path.join(src_dir, f'{subset}/{subset}_mp3.csv')
        else:
            subset_path = os.path.join(src_dir, f'{subset}')
        subset_df = pd.read_csv(subset_path)

        # Fix paths for specific vocoders → replace with *_gen.wav
        subset_df['path'] = subset_df['path'].apply(rewrite_path)

        dfs.append(subset_df)
    test_df, train_df, validate_df = dfs    
    return train_df, validate_df, test_df

def rewrite_path(path: str) -> str:
    if "ljspeech_hifiGAN" in path:
        if path.endswith(".wav") and not path.endswith("_generated.wav"):
            base, ext = os.path.splitext(path)
            path = f"{base}_generated{ext}"
    elif any(v in path for v in [
        "ljspeech_melgan_large",
        "ljspeech_multi_band_melgan",
        "ljspeech_parallel_wavegan"
    ]):
        if path.endswith(".wav") and not path.endswith("_gen.wav"):
            base, ext = os.path.splitext(path)
            path = f"{base}_gen{ext}"

    return path

def save_train_validate_test_dfs_to_csvs(csvs_dir, train_df, validate_df, test_df, seed):

    # Ensure the directory exists before saving
    os.makedirs(csvs_dir, exist_ok=True)
    
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    validate_df = validate_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Save train, validate and test dataframes to csvs
    train_df.to_csv(os.path.join(csvs_dir, 'train.csv'), index=False)
    validate_df.to_csv(os.path.join(csvs_dir, 'validate.csv'), index=False)
    test_df.to_csv(os.path.join(csvs_dir, 'test.csv'), index=False)

    return train_df, validate_df, test_df


def get_mean_std(
    ds,
    model,
    classification_type,
    seed, 
    mean_std_path,
    filter_fn,
    trans_fn):

    dir = os.path.join(mean_std_path)

    if model == "fingerprint" and "multiclass" in classification_type: # No mean and std needed for samples scoring using mahalanobis distance
        return None, None
   
    if model == "vfd-resnet":
        dir = os.path.join(dir, "mel")

    elif model == "fingerprint":
        dir = os.path.join(dir, "residuals")
    else:
        dir = os.path.join(dir, "lfcc")

    # Create corresponding mean and std if not present
    dir = os.path.join(dir, str(seed))
    if not os.path.exists(dir) : # not os.path.exists(dir):
        print(f"Mean and std for the corresponding train dataset and gived seed not found. {dir}")
        compute_mean_std_and_save(
            ds=ds,
            model=model,
            out_dir=dir,
            filter_fn=filter_fn,
            trans_fn=trans_fn)

    with open(os.path.join(dir, 'mean.pkl'), "rb") as f:
        mean = pickle.load(f)
    with open(os.path.join(dir, 'std.pkl'), "rb") as f:
        std = pickle.load(f)

    return mean, std

# filter_fn, AVG_SPEC
def get_datasets(model, classification_type, seed, corruption_type, scale_factor, corpus, mean_std_dir, 
                sample_rate, coef, n_fft, hop_length):

    print(f'Searching for saved dataset for {classification_type} and given seed({seed})')

    BASE_DIR = os.getcwd()  # gets the directory where you run the code

    URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES = os.path.join(BASE_DIR, "csv_dir", corpus, "fake_audio", "multiclass")
    URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES = os.path.join(BASE_DIR, "csv_dir", corpus, "real_audio", "multiclass")
    URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES  = os.path.join(BASE_DIR, "csv_dir", corpus, "mix_audio")

    CSV_DIR_DEST = {
        "real_audio": URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES,
        "fake_audio": URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES,
        "mix_audio": URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES,
    }

    if "multiclass" in classification_type:

        audio_type = "fake_audio"
        subsets_folder_path = os.path.join(CSV_DIR_DEST[audio_type], f'{seed}/csv_files_split')

    else:   # "binary" in classification_type
        audio_type = "mix_audio"
        subsets_folder_path = os.path.join(CSV_DIR_DEST[audio_type], f'{classification_type}/{seed}/csv_files_split')

    if os.path.exists(subsets_folder_path):
        print("Dataset for the given seed was found. Loading...", subsets_folder_path)
        train_df, validate_df, test_df = get_train_validate_test_df(subsets_folder_path, corruption_type)

    else:   # New seed, dataset has to be reconstructed
        print(f'No dataset found for specified seed!')
        # audio_type == "mix_audio"
        # Search for real train, validate and test csv files for given seed
        real_audio_csv_split_dir_path = os.path.join(CSV_DIR_DEST["real_audio"], f'{seed}/csv_files_split')
        if not os.path.exists(real_audio_csv_split_dir_path):
            print(f"Directory does not exist {real_audio_csv_split_dir_path}. Exiting program.")
            sys.exit()
        print("Real audio's train, validate and test csvs for given seed found. Loading...")
        real_audio_train_df, real_audio_validate_df, real_audio_test_df = get_train_validate_test_df(real_audio_csv_split_dir_path)
        # Non proportional
        # Search for non-proportional fake audio csvs
        print("Searching non-proportional, fake audio's train, validate and test sets...")
        
        non_proportional_fake_audio_csvs_dir = os.path.join(CSV_DIR_DEST["fake_audio"], str(seed), "csv_files_split")
        if not os.path.exists(non_proportional_fake_audio_csvs_dir):
            print(f"Directory does not exist {non_proportional_fake_audio_csvs_dir}. Exiting program.")
            sys.exit()
        print("Fake audio dataset for specified seed found. Loadning...")
        fake_audio_train_df, fake_audio_validate_df, fake_audio_test_df = get_train_validate_test_df(non_proportional_fake_audio_csvs_dir)
        
        # Muliplicate real audio by the number of audio vocoders and concat real and fake audio subsets
        train_temp = []
        validate_temp = []
        test_temp = []
        print(len(CLASSES[classification_type][corpus]))
        for _ in range(len(CLASSES[classification_type][corpus])):
            print(len(real_audio_test_df))
            train_temp.append(real_audio_train_df)
            validate_temp.append(real_audio_validate_df)
            test_temp.append(real_audio_test_df)
        if corpus == "codecfake":
            test_temp.append(real_audio_test_df)
        elif corpus == "asvspoof":
            for _ in range(11):
                test_temp.append(real_audio_test_df)
        real_audio_train_df = pd.concat(train_temp)
        real_audio_validate_df = pd.concat(validate_temp)
        real_audio_test_df = pd.concat(test_temp)
        # Check size of testing, training and validation !!!
        print(f'real audio dataset multiplicated for {classification_type}')
        fake_audio_train_df["label"] = 1
        fake_audio_validate_df["label"] = 1
        fake_audio_test_df["label"] = 1
        train_df = pd.concat([real_audio_train_df, fake_audio_train_df])
        validate_df = pd.concat([real_audio_validate_df, fake_audio_validate_df])
        test_df = pd.concat([real_audio_test_df, fake_audio_test_df])

        # Save dataframes to csvs
        train_df, validate_df, test_df = save_train_validate_test_dfs_to_csvs(
            csvs_dir=os.path.join(CSV_DIR_DEST["mix_audio"], f'{classification_type}/{seed}/csv_files_split'),
            train_df=train_df,
            validate_df=validate_df,
            test_df=test_df,
            seed=seed
        )
    n_classes = len(CLASSES[classification_type][corpus])
    print("Number of training classes: ", n_classes)
    # Filter by num_classes
    if classification_type == "multiclass":
        train_df = train_df[train_df["label"] <= n_classes]
        validate_df = validate_df[validate_df["label"] <= n_classes]
        test_df = test_df[test_df["label"] <= n_classes]
        for _ in range(n_classes):
            print(len(train_df[train_df["label"] == _ + 1]), len(validate_df[validate_df["label"] == _ + 1]), len(test_df[test_df["label"] == _ + 1]), _ + 1)
    else:
        for _ in range(n_classes):
            print(len(train_df[train_df["label"] == _]), len(validate_df[validate_df["label"] == _]), len(test_df[test_df["label"] == _]), _)

    if model == "fingerprint" or model == "fingerprint_2":
        target_sr = sample_rate
    else:
        target_sr = TARGET_SAMPLE_RATE[model]
    if corruption_type == 1:
        # Evasion Attack
        # Get 100 samples per category
        test_df = (
                        test_df.groupby("label", group_keys=False)
                        .apply(lambda x: x.sample(n=100, random_state=seed))
                    )

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    validate_df = validate_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_ds = CustomDataset(dataset_df=train_df, sample_rate=sample_rate, target_sample_rate=target_sr, model=model, classification_type=classification_type, mean=None, std=None, seed=seed, coef=coef, n_fft=n_fft, hop_length=hop_length)
    validate_ds = CustomDataset(dataset_df=validate_df, sample_rate=sample_rate, target_sample_rate=target_sr, model=model, classification_type=classification_type, mean=None, std=None, seed=seed, coef=coef, n_fft=n_fft, hop_length=hop_length)
    test_ds = CustomDataset(dataset_df=test_df, sample_rate=sample_rate, target_sample_rate=target_sr, model=model, classification_type=classification_type, mean=None, std=None, seed=seed, corruption_type=corruption_type, scale_factor=scale_factor, coef=coef, n_fft=n_fft, hop_length=hop_length)

    print(f'Train dataset of size {len(train_df)}')
    print(f'Validate dataset of size {len(validate_df)}')
    print(f'Test dataset of size {len(test_df)}')
    print(f'Total size: {len(train_df) + len(validate_df) + len(test_df)}')

    return train_ds, validate_ds, test_ds, None