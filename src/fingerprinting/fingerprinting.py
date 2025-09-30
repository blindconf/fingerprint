import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.datasets.utility_2 import pad_and_concatenate
from speechbrain.processing.speech_augmentation import AddReverb
import os
import glob
import pickle


class WaveformToAvgSpec:
    def __init__(self, 
                 n_fft,
                 hop_length,
                 to_db=True,
                 device="cuda"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.transf = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                        hop_length=self.hop_length).to(device)
        self.device = device  
        self.to_db = to_db   
        
    def forward(self, batch: torch.Tensor, original_audio_lengths: list) -> torch.Tensor: 
        # batch = batch.squeeze(0).to(self.device)
        max_audio_len = batch.shape[2]
        num_samps = batch.shape[0]
        mask = torch.arange(max_audio_len).expand(num_samps, max_audio_len) >= torch.tensor(original_audio_lengths).unsqueeze(1)
        batch[mask.unsqueeze(1)] = float('nan')
        spec = self.transf(batch)
        # print(spec)
        if self.to_db:
            spec = 10. * torch.log10(spec + 1e-10)
        # energy = torch.mean(spec, dim=3)  
        # return energy.unsqueeze(0)
        return torch.nanmean(spec, dim=3)

class FingerprintingWrapper:
    name = "fingerprint"

    def __init__(self, 
                 filter=None, 
                 num_samples: int = 100, 
                 tolerable_fnr: list = [0.0, 0.001, 0.005, 0.01], 
                 transformation=None, 
                 name=None,
                 scoring="mahalanobis",
                 keep_percentage=[0., 1.],
                 reweight=False,
                 filter_trend_correction=False) -> None:
        self.num_samples = num_samples
        self.fingerprint = None
        self.tolerable_fnr = tolerable_fnr
        self.thresholds = {}
        self.filter = filter
        self.transformation = transformation
        self.name = name
        self.spectrograms_avg = None
        self.spect_filter_avg = None
        self.keep_percentage = keep_percentage
        self.reweight = reweight 
        self.filter_trend_correction = filter_trend_correction
        assert scoring in ["correlation", "mahalanobis"], "scoring should be correlation or mahalanobis"
        self.scoring = scoring 
 
    def train(self, dl: DataLoader, ds_real: DataLoader = None):
        residuals = []

        if self.filter.name == 'Oracle':
            total_batches = min(len(dl), len(ds_real))
            for batch1, batch2 in tqdm(zip(dl, ds_real), total=total_batches, desc="Processing Batches"):
                output_1_path = batch1[3]
                output_2_path = batch2[3]
                # Create sets of extracted names for easy comparison
                names_list1 = {extract_name(f) for f in output_1_path}
                names_list2 = {extract_name(p) for p in output_2_path}
                
                # Find matches by taking the intersection of both sets
                matching_names = names_list1.intersection(names_list2)
                if len(names_list1) != len(matching_names):
                    raise NotImplementedError("To obtain ORACLE results, the fake audio used for training must correspond to the same original speech from which it was generated.")
                avg_ = self.transformation.forward(batch1[0], batch1[1])
                filtered_avg_ = self.transformation.forward(batch2[0], batch2[1])
                residual_avg_ = avg_ - filtered_avg_
                residuals.append(residual_avg_) 
            residuals = pad_and_concatenate(residuals)
            fingerprint = torch.mean(residuals, dim=0)

            if self.filter_trend_correction:
                fingerprint = fingerprint - self.trend  
            
            if self.scoring == "correlation":
                self.fingerprint = self.zero_mean_unit_norm(fingerprint)
            elif self.scoring == "mahalanobis":
                self.fingerprint = fingerprint
                covariance = torch.cov(residuals.squeeze().T)
                self.invcov = torch.inverse(covariance)
        else:            
            if self.filter_trend_correction:
                raise NotImplementedError("Trend correction is not implemented for the current filter")
                assert ds_real is not None, "ds_real was not specified"

            for i in tqdm(dl, desc="Computing fingerprint"):
                batch = i[0]
                original_audio_lengths = i[1]
                batch_sample_rate = i[2][0]

                if self.filter.name in ["low_pass_filter", 
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter"]:
                    avg_ = self.transformation.forward(batch, original_audio_lengths)
                    filtered_batch = self.filter.forward(batch)
                    filtered_avg_ = self.transformation.forward(filtered_batch, original_audio_lengths)
                    residual_avg_ = avg_ - filtered_avg_
                    residuals.append(residual_avg_) 
                elif self.filter.name == "EncodecFilter":
                    features = self.transformation.forward(batch, original_audio_lengths)
                    filtered_batch = self.filter.forward(batch, batch_sample_rate)
                    filtered_features = self.transformation.forward(filtered_batch, original_audio_lengths)
                    residual_features = features - filtered_features
                    residuals.append(residual_features)
                
            residuals = pad_and_concatenate(residuals)
            fingerprint = torch.mean(residuals, dim=0)
            if self.filter_trend_correction:
                fingerprint = fingerprint - self.trend  
            if self.scoring == "correlation":
                self.fingerprint = self.zero_mean_unit_norm(fingerprint)
            elif self.scoring == "mahalanobis":
                self.fingerprint = fingerprint
                covariance = torch.cov(residuals.squeeze().T)
                self.invcov = torch.inverse(covariance)

        pass
                    
    def forward(self, dl: torch.Tensor, ds_real: DataLoader = None, cutoff=None, corruption_type=0, scale_factor=1.0) -> float:
        scores = []
        if self.filter.name == 'Oracle':
            total_batches = min(len(dl), len(ds_real))
            for batch1, batch2 in tqdm(zip(dl, ds_real), total=total_batches, desc="Evaluating fingerprint"):
                output_1_path = batch1[3]
                output_2_path = batch2[3]
                # Create sets of extracted names for easy comparison
                names_list1 = {extract_name(f) for f in output_1_path}
                names_list2 = {extract_name(p) for p in output_2_path}

                # Find matches by taking the intersection of both sets
                matching_names = names_list1.intersection(names_list2)
                if len(names_list1) != len(matching_names):
                    raise NotImplementedError("To obtain ORACLE results, the fake audio used for training must correspond to the same original speech from which it was generated.")
                avg_ = self.transformation.forward(batch1[0], batch1[1])
                filtered_avg_ = self.transformation.forward(batch2[0], batch2[1])
                residual = avg_ - filtered_avg_
                if self.scoring=="correlation":
                    residual = self.zero_mean_unit_norm(residual)
                    fingerprint = self.zero_mean_unit_norm(self.fingerprint)
                    score = correlation_score(fingerprint, residual)
                elif self.scoring=="mahalanobis":
                    fingerprint = self.fingerprint
                    score = mahalanobis_score(fingerprint, residual, self.invcov)
                scores.append(score)
        else:
            if corruption_type == 1:
                reverb_path = "/USERSPACE/DATASETS/LibriSpeech/reverb.csv"
                reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=scale_factor)    
            for i in tqdm(dl, desc="Evaluating fingerprint"):
                original_audio_lengths = i[1]
                batch_sample_rate = i[2][0]
                path = i[3]
                if corruption_type == 1:
                    audio_rev = i[0].squeeze(1)
                    # Convert to tensor
                    lengths_tensor = torch.tensor(original_audio_lengths, dtype=torch.float32)
                    # Normalize by the maximum length
                    normalized = lengths_tensor / lengths_tensor.max()
                    # Sort lengths ascending, and get sorted indices
                    sorted_lengths, indices = torch.sort(normalized, descending=False)
                    # Reorder x using those indices
                    audio_rev = audio_rev[indices]
                    '''
                    torchaudio.save("example.wav", reverb(audio_rev, normalized).cpu(), batch_sample_rate, encoding="PCM_S", bits_per_sample=16)
                    
                    reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=0.5)
                    torchaudio.save("example_1.wav", reverb(audio_rev, normalized).cpu(), batch_sample_rate, encoding="PCM_S", bits_per_sample=16)

                    reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=0.2)
                    torchaudio.save("example_2.wav", reverb(audio_rev, normalized).cpu(), batch_sample_rate, encoding="PCM_S", bits_per_sample=16)

                    reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=1.2)
                    torchaudio.save("example_3.wav", reverb(audio_rev, normalized).cpu(), batch_sample_rate, encoding="PCM_S", bits_per_sample=16)

                    reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=1.5)
                    torchaudio.save("example_4.wav", reverb(audio_rev, normalized).cpu(), batch_sample_rate, encoding="PCM_S", bits_per_sample=16)
                    '''
                    audio = reverb(audio_rev, normalized).reshape(i[0].shape)
                    # print(path)
                else:
                    audio = i[0]
                if self.filter.name == "EncodecFilter":
                    filtered_audio = self.filter.forward(audio, batch_sample_rate)
                elif self.filter.name in ["low_pass_filter", 
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter"]:
                    filtered_audio = self.filter.forward(audio)            
                else:
                    filtered_audio = audio
                
                if self.filter.name in ["low_pass_filter", 
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter",
                                        "Oracle"]:
                    avg_ = self.transformation.forward(audio, original_audio_lengths)
                    filtered_avg_ = self.transformation.forward(filtered_audio, original_audio_lengths)
                    residual = avg_ - filtered_avg_ 
                    if self.filter_trend_correction:
                        residual = residual - self.trend
                    
                    if self.scoring=="correlation":
                        residual = self.zero_mean_unit_norm(residual)
                        fingerprint = self.zero_mean_unit_norm(self.fingerprint)
                        score = correlation_score(fingerprint, residual)
                    elif self.scoring=="mahalanobis":
                        fingerprint = self.fingerprint
                        score = mahalanobis_score(fingerprint, residual, self.invcov)
                        
                elif self.filter.name == "EncodecFilter":
                    features = self.transformation.forward(audio, original_audio_lengths)
                    filtered_features = self.transformation.forward(filtered_audio, original_audio_lengths)
                    residual = features - filtered_features
                    if self.filter_trend_correction:
                        residual = residual - self.trend
                    
                    if cutoff is not None: 
                        residual = residual[:, :cutoff]
                        print("Use of a cutoff!")
                    if self.fingerprint.shape[1] != cutoff: 
                        fingerprint = self.fingerprint[:, :cutoff]
                        if self.scoring=="correlation":
                            fingerprint = self.zero_mean_unit_norm(self.fingerprint)                             
                    else:
                        fingerprint = self.fingerprint
                    
                    if self.scoring=="correlation":
                        residual = self.zero_mean_unit_norm(residual)
                        score = correlation_score(fingerprint, residual)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual, self.invcov)
                
                scores.append(score)
        return pad_and_concatenate(scores)
    
    @staticmethod
    def spec_in_db(spec):
        return 10. * torch.log(spec + 10e-13)

    @staticmethod
    def db_in_spec(db):
        return  torch.exp(db / 10.0) - 10e-13 

    @staticmethod
    def zero_mean_unit_norm(array: torch.tensor) -> torch.tensor:
        # Calculate the mean and standard deviation along the first dimension
        array = array - array.mean(dim=-1, keepdim=True)
        return array / array.norm(dim=-1, keepdim=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pickle.dump(self.fingerprint, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.fingerprint = pickle.load(f)

    def __repr__(self):
        return f"FingerprintingWrapper(num_samples={self.num_samples})"

def correlation_score(fingerprint, input_residual):
    # Calculate the correlation scores using inner product
    # We need to remove the singleton dimensions from the fingerprint and batch elements before using torch.inner
    correlation_scores = torch.inner(input_residual.squeeze(1), fingerprint.squeeze(0))
    # return torch.inner(fingerprint.flatten(), input_residual.flatten())
    return correlation_scores

def mahalanobis_score(fingerprint, batch_residual, invcov):
    scores = []
    for i in range(batch_residual.shape[0]):
        input_residual = batch_residual[i, :, :]
        delta = input_residual.flatten() - fingerprint.flatten()   
        score = torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))
        scores.append(-1 * score.item())
    return torch.tensor(scores)

# Function to extract the name up to the first underscore
def extract_name(filepath):
    # Get just the filename from the full path
    filename = filepath.split('/')[-1]
    # Split by _ or . and take the first part
    return filename.split('_')[0].split('.')[0]

def load_fingerprints(fing_path, filter_param, scorefunction, nfft, hop_len, classes, DEV):
    fingerprints = {}

    for vocoder_dir in sorted(os.listdir(fing_path)):
        if not vocoder_dir in classes:
            continue
        vocoder_path = os.path.join(fing_path, vocoder_dir)
        
        if not os.path.isdir(vocoder_path):
            continue
        # Find the unique fingerprint file        

        # All fingerprint files in that folder
        all_fingerprints = glob.glob(os.path.join(vocoder_path, "*_fingerprint.pickle"))

        # Filter the one that matches your known partial prefix
        partial_prefix = f"param={filter_param}_score={scorefunction}_nfft={nfft}_hoplen={hop_len}"
        matches = [f for f in all_fingerprints if os.path.basename(f).startswith(partial_prefix)]

        if len(matches) == 0:
            raise FileNotFoundError(f"No fingerprint file found in {vocoder_path}")
        elif len(matches) > 1:
            raise RuntimeError(f"More than one fingerprint file found in {vocoder_path}: {matches}")
        else:
            fingerprint_files = matches[0]

        all_invcov = glob.glob(os.path.join(vocoder_path, "*_invcov.pickle"))
        # Filter the one that matches your known partial prefix
        matches = [f for f in all_invcov if os.path.basename(f).startswith(partial_prefix)]

        if len(matches) == 0:
            raise FileNotFoundError(f"No covariance file found in {vocoder_path}")
        elif len(matches) > 1:
            raise RuntimeError(f"More than one covariance file found in {vocoder_path}: {matches}")
        else:
            invcov_files = matches[0]

        if len(fingerprint_files.strip().split("\n")) != len(invcov_files.strip().split("\n")):
            raise ValueError(f"Mismatch in number of fingerprint and invcov files in {vocoder_dir}")

        # Extract parameters from filenames
        params = {}
        transformation_type = "Avg_Spec"
        
        fp_file= os.path.basename(fingerprint_files)
        filename_parts = fp_file.replace(".pickle", "").split("_")
        for item in filename_parts:
            key_value = item.split("=")
            if len(key_value) == 2:
                params[key_value[0]] = key_value[1]

        params["transformation_type"] = transformation_type

        # Load fingerprint and invcov
        with open(fingerprint_files, "rb") as f:
            fingerprint = pickle.load(f)
        with open(invcov_files, "rb") as f:
            invcov = pickle.load(f)

        # Initialize vocoder dictionary if not already present
        if vocoder_dir not in fingerprints:
            fingerprints[vocoder_dir] = []

        # Append the fingerprint and related data
        fingerprints[vocoder_dir].append({
            "fingerprint": fingerprint.to(DEV),
            "invcov": invcov.to(DEV),
            "params": params
        })
    # print("fingerprints: ", fingerprints)
    return fingerprints


def mahalanobis_score(fingerprint, batch_residual, invcov, DEV):
    
    batch_size = batch_residual.shape[0]
    scores = torch.empty(batch_size, device=DEV)
    
    for i in range(batch_size):
        input_residual = batch_residual[i, :, :]
        delta = input_residual.flatten() - fingerprint.flatten()
        scores[i] = -torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))


    return scores

def evasion_attack_scores(residuals, fingerprints, orig_labels, target_labels, label_map_inv, DEV):
    batch_size = residuals.shape[0]
    D = residuals.shape[-1]
    num_fingerprints = len(fingerprints)

    scores = torch.zeros((batch_size, num_fingerprints), device=DEV)
    fingerprints_list = sorted(fingerprints.keys())  # system names

    # Pre-build mapping from system name → (fp, invcov)
    sys_to_fp = {}
    for sys_name, entries in fingerprints.items():
        fp = entries[0]["fingerprint"].to(DEV)
        invcov = entries[0]["invcov"].to(DEV)
        sys_to_fp[sys_name] = (fp, invcov)

    # Build per-sample source/target
    source_fps, target_fps = [], []
    # print(orig_labels.tolist())
    # print(target_labels.tolist())
    for ol, tl in zip(orig_labels.tolist(), target_labels.tolist()):
        src_sys = label_map_inv[ol]
        tgt_sys = label_map_inv[tl]
        source_fps.append(sys_to_fp[src_sys][0])
        target_fps.append(sys_to_fp[tgt_sys][0])

    source_fps = torch.stack(source_fps, dim=0).to(DEV)
    target_fps = torch.stack(target_fps, dim=0).to(DEV)
    # print(residuals.shape, source_fps.shape, target_fps.shape)
    residuals_2 = residuals - source_fps + target_fps

        # Compute scores
    for fingerprint_index, sys_name in enumerate(fingerprints_list):
        fp, invcov = sys_to_fp[sys_name]
        fingerprint_score = mahalanobis_score(fp, residuals_2, invcov)
        scores[:, fingerprint_index] = fingerprint_score

    # print(scores.shape)
    
    return scores

def compute_mahalanobis_scores(residuals, fingerprints, DEV):

    num_fingerprints = len(fingerprints)
    batch_size = residuals.shape[0]

    scores = torch.zeros((batch_size, num_fingerprints), device=DEV)

    fingerprints_list = sorted(fingerprints.keys())


    for fingerprint_index, fingerprint_name in enumerate(fingerprints_list):

        for data in fingerprints[fingerprint_name]:
            fingerprint = data["fingerprint"]
            invcov = data["invcov"]
            fingerprint_score = mahalanobis_score(fingerprint, residuals, invcov, DEV)
            
            scores[:, fingerprint_index] = fingerprint_score
    
    return scores


def assign_vocoders(scores):
    
    # Get the best vocoder index for each sample based on highest Mahalanobis score
    best_vocoder_indices = torch.argmax(scores, dim=1)
    
    # Convert predictions and labels to tensors
    preds_tensor = best_vocoder_indices.float()
    return preds_tensor