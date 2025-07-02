import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.datasets.utility import pad_and_concatenate


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
            spec = 10. * torch.log(spec + 10e-13)
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
                    
    def forward(self, dl: torch.Tensor, ds_real: DataLoader = None, cutoff=None) -> float:
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
            for i in tqdm(dl, desc="Evaluating fingerprint"):
                audio = i[0]
                original_audio_lengths = i[1]
                batch_sample_rate = i[2][0]
                path = i[2]
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