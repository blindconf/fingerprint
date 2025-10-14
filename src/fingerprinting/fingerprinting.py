import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.datasets.utility_2 import pad_and_concatenate
import os
import glob
import pickle


class WaveformToAvgSpec:
    """
    Converts raw waveforms into averaged spectrogram representations.
    
    Parameters:
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length between frames.
        to_db (bool): Convert spectrogram magnitude to decibels.
        device (str): Device to run computations ('cuda' or 'cpu').
    """
    def __init__(self, 
                 window_size,
                 hop_size,
                 sample_rate,
                 to_db=True,
                 device="cuda"):
        
        self.n_fft = int(window_size / 1000 * sample_rate) 
        self.hop_length = int(hop_size / 1000 * sample_rate) 
        
        self.transf = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                        hop_length=self.hop_length).to(device)
        self.device = device  
        self.to_db = to_db   
        
    def forward(self, batch: torch.Tensor, original_audio_lengths: list) -> torch.Tensor:
        """
        Compute the spectrogram for a batch of audio waveforms and average over time.
        
        Parameters:
            batch (Tensor): Batch of audio waveforms (B x C x T).
            original_audio_lengths (list): Original lengths of each audio sample in the batch.
        
        Returns:
            Tensor: Averaged spectrogram for each sample in the batch (B x C x F).
        """
        max_audio_len = batch.shape[2]
        num_samps = batch.shape[0]
        mask = torch.arange(max_audio_len).expand(num_samps, max_audio_len) >= torch.tensor(original_audio_lengths).unsqueeze(1)
        # Mask padded audio with NaNs
        batch[mask.unsqueeze(1)] = float('nan')
        # Compute spectrogram
        spec = self.transf(batch)
        if self.to_db:
            spec = 10. * torch.log10(spec + 1e-10)
        # Average over time dimension
        return torch.nanmean(spec, dim=3)

class FingerprintingWrapper:
    """
    Wrapper class to compute audio fingerprints for detecting synthetic speech.
    
    Attributes:
        fingerprint (Tensor): The computed fingerprint after training.
        filter: Audio filter applied to signals before fingerprinting.
        transformation: Transformation applied to waveforms (e.g., WaveformToAvgSpec).
        scoring (str): Scoring method ('correlation' or 'mahalanobis').
        filter_trend_correction (bool): Whether to remove global trend from fingerprints.
    """
    name = "fingerprint"

    def __init__(self, 
                 filter=None, 
                 num_samples: int = 100, 
                 transformation=None, 
                 name=None,
                 scoring="mahalanobis",
                 filter_trend_correction=False) -> None:
        self.num_samples = num_samples
        self.fingerprint = None
        self.thresholds = {}
        self.filter = filter
        self.transformation = transformation
        self.name = name
        self.filter_trend_correction = filter_trend_correction
        assert scoring in ["correlation", "mahalanobis"], "scoring should be correlation or mahalanobis"
        self.scoring = scoring 
 
    def train(self, dl: DataLoader, ds_real: DataLoader = None):
        """
        Train the fingerprint from a DataLoader of fake/generated audio.
        Optionally uses real audio to remove global trends.
        
        Parameters:
            dl (DataLoader): DataLoader for the fake/generated audio.
            ds_real (DataLoader, optional): DataLoader for real audio (needed for trend correction or Oracle filter).
        """
        residuals = []
         # Compute global trend if trend correction is enabled
        if self.filter_trend_correction:
            assert ds_real is not None, "ds_real was not specified"
            residuals_real = []
            for i in tqdm(ds_real, desc="Computing trend"):
                batch = i[0]
                original_audio_lengths = i[1]
                avg_feature = self.transformation.forward(batch, original_audio_lengths)
                filtered_batch = self.filter.forward(batch)
                filtered_avg_feature = self.transformation.forward(filtered_batch, original_audio_lengths)
                residual_avg_feature = avg_feature - filtered_avg_feature
                residuals_real.append(residual_avg_feature)
            residuals_real = pad_and_concatenate(residuals_real)
            self.trend = torch.mean(residuals_real, dim=0)
        # Handle Oracle filter separately
        if self.filter.name == 'Oracle':
            assert ds_real is not None, "ds_real was not specified"
            total_batches = min(len(dl), len(ds_real))
            for batch1, batch2 in tqdm(zip(dl, ds_real), total=total_batches, desc="Processing Batches"):
                output_1_path = batch1[3]
                output_2_path = batch2[3]
                # Verify correspondence between real and fake audio
                names_list1 = {extract_name(f) for f in output_1_path}
                names_list2 = {extract_name(p) for p in output_2_path}
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
            # Compute fingerprint for standard filters        
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
                    
    def forward(self, dl: torch.Tensor, ds_real: DataLoader = None) -> float:
        """
        Evaluate audio batches against the trained fingerprint to compute similarity scores.
        
        Parameters:
            dl (DataLoader or Tensor): Batch of audio to evaluate.
            ds_real (DataLoader, optional): Real audio DataLoader (needed for Oracle filter).
        
        Returns:
            float: Similarity scores (higher indicates stronger match with fingerprint).
        """
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
                
                if self.filter_trend_correction:
                    residual = residual - self.trend
                
                if self.scoring=="correlation":
                    residual = self.zero_mean_unit_norm(residual)
                    fingerprint = self.zero_mean_unit_norm(self.fingerprint)
                    score = correlation_score(fingerprint, residual)
                elif self.scoring=="mahalanobis":
                    fingerprint = self.fingerprint
                    score = mahalanobis_score(fingerprint, residual, self.invcov)
                scores.append(score)
        else:
            for i in tqdm(dl, desc="Evaluating fingerprint"):
                audio = i[0]
                original_audio_lengths = i[1]
                batch_sample_rate = i[2][0]
                path = i[3]                
                if self.filter.name == "EncodecFilter":
                    filtered_audio = self.filter.forward(audio, batch_sample_rate)
                else:
                    filtered_audio = self.filter.forward(audio)            
                
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
                scores.append(score)
        return pad_and_concatenate(scores)
    
    @staticmethod
    def spec_in_db(spec):
        """
        Convert a spectrogram magnitude to decibel scale.
        
        Parameters:
            spec (Tensor): Spectrogram magnitude.
        
        Returns:
            Tensor: Spectrogram in decibels.
        """
        return 10. * torch.log10(spec + 1e-10)

    @staticmethod
    def db_in_spec(db):
        """
        Convert decibel-scaled spectrogram back to linear magnitude.
        
        Parameters:
            db (Tensor): Spectrogram in decibels.
        
        Returns:
            Tensor: Linear magnitude spectrogram.
        """
        return  10 **(db / 10.0) - 1e-10

    @staticmethod
    def zero_mean_unit_norm(array: torch.tensor) -> torch.tensor:
        """
        Normalize a tensor to have zero mean and unit norm along the last dimension.
        
        Parameters:
            array (Tensor): Input tensor.
        
        Returns:
            Tensor: Normalized tensor.
        """
        array = array - array.mean(dim=-1, keepdim=True)
        return array / array.norm(dim=-1, keepdim=True)

    def save(self, path: Path) -> None:
        """
        Save the trained fingerprint to a file.
        
        Parameters:
            path (Path): File path to save fingerprint.
        """
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pickle.dump(self.fingerprint, f)

    def load(self, path: Path) -> None:
        """
        Load a fingerprint from a file.
        
        Parameters:
            path (Path): File path containing a saved fingerprint.
        """
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
'''
def mahalanobis_score(fingerprint, batch_residual, invcov, DEV):
    batch_size = batch_residual.shape[0]
    scores = torch.empty(batch_size, device=DEV)
    for i in range(batch_size):
        input_residual = batch_residual[i, :, :]
        delta = input_residual.flatten() - fingerprint.flatten()
        scores[i] = -torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))
    return scores
'''
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
    return fingerprints

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
    for ol, tl in zip(orig_labels.tolist(), target_labels.tolist()):
        src_sys = label_map_inv[ol]
        tgt_sys = label_map_inv[tl]
        source_fps.append(sys_to_fp[src_sys][0])
        target_fps.append(sys_to_fp[tgt_sys][0])
    source_fps = torch.stack(source_fps, dim=0).to(DEV)
    target_fps = torch.stack(target_fps, dim=0).to(DEV)
    residuals_2 = residuals - source_fps + target_fps
        # Compute scores
    for fingerprint_index, sys_name in enumerate(fingerprints_list):
        fp, invcov = sys_to_fp[sys_name]
        fingerprint_score = mahalanobis_score(fp, residuals_2, invcov)
        scores[:, fingerprint_index] = fingerprint_score
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