import gc
import torch
from torchaudio import load
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from torchaudio.transforms import LFCC, MelSpectrogram
from src.datasets.filters import filter_fn
from torchaudio.transforms import LFCC, MelSpectrogram, MFCC, Spectrogram
from src.training.invariables import DEV


class CustomDataset(Dataset):

    def __init__(self, dataset_df, sample_rate, target_sample_rate, model, classification_type, mean, std, seed, postprocess=None) -> None:

        self.df = dataset_df
        self.model = model
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.classification_type = classification_type
        self.mean = mean
        self.std = std
        self.resampler = Resample(self.sample_rate, self.target_sample_rate)
        self.postprocess = postprocess
        self.seed = seed
        
        self.lfcc = LFCC(
            n_filter=20,
            n_lfcc=60,
            speckwargs={
                "n_fft": 512,
                "win_length": int(0.025 * self.target_sample_rate),
                "hop_length": int(0.01 * self.target_sample_rate)
            }
        )


        self.mel = MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=300,
            win_length=1200,
            n_mels=80,
            f_min=0,
            f_max=12000,
            window_fn=torch.hamming_window
        )

        if model in ["resnet", "se-resnet", "lcnn", "x-vector"]:
            self.transform = self.lfcc
        elif model == "vfd-resnet":
            self.transform = self.log_mel_transform
        else:
            self.transform = None


    def __getitem__(self, index):

        row = self.df.iloc[index]
        sample_uri = row["path"]
        label = row["label"]

        waveform, samplerate = load(sample_uri)
        waveform = waveform.float()
        waveform = self.resample(waveform, samplerate)

        if self.transform is not None:
            waveform = self.transform(waveform)
            waveform = waveform.squeeze(0)  # Remove batch dimension

        if self.mean is not None and self.std is not None:
            waveform = (waveform - self.mean[:, None]) / self.std[:, None]

        if self.postprocess is not None:
            waveform = self.postprocess(waveform)

        label = torch.tensor(label, dtype=torch.long)
        return waveform, label


    def log_mel_transform(self, waveform):

        return torch.log(self.mel(waveform) + 1e-6)

    
    def resample(self, signal, sr):
        if sr != self.target_sample_rate:            
            signal = self.resampler(signal)
        return signal


    def __len__(self):
        return len(self.df)


# Transforms for fingerprints
class WaveformToAvgMFCC: 
    def __init__(self, sample_rate, n_mfcc, melkwargs, device):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.melkwargs = melkwargs
        
        self.transf = MFCC(sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, melkwargs=self.melkwargs).to(device)
        self.device = device
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0).to(self.device)
        mfcc = self.transf(batch)
        energy = torch.mean(mfcc.squeeze(0), dim=1)  
        return energy.unsqueeze(0)


class WaveformToAvgMel:
    def __init__(self, 
                 sample_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 device,
                 to_db=True):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.transf = MelSpectrogram(sample_rate=self.sample_rate,
                                                           n_fft=self.n_fft,
                                                           hop_length=self.hop_length,
                                                           n_mels=self.n_mels).to(device)
        self.device = device  
        self.to_db = to_db   
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor: 
        #batch = batch.squeeze(0).to(self.device)
        mfcc = self.transf(batch)
        if self.to_db:
            mfcc = 10. * torch.log(mfcc + 10e-13)
        energy = torch.mean(mfcc.squeeze(0), dim=1)  
        return energy.unsqueeze(0)


class WaveformToAvgSpec:
    def __init__(self, 
                 n_fft,
                 hop_length,
                 device,
                 to_db=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.transf = Spectrogram(n_fft=self.n_fft,
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
        if self.to_db:
            spec = 10. * torch.log(spec + 10e-13)
        # energy = torch.mean(spec, dim=3)  
        # return energy.unsqueeze(0)
        return torch.nanmean(spec, dim=3)

x = []
file_in = open(f"/home/pizarm5k/audio-fingerprint/filter_coefs/low_pass_filter/1.0khz.txt", 'r')
for y in file_in.read().split('\n'):
    x.append(float(y))
coef = torch.tensor(x)
FILTER = filter_fn(1, coef, dev=DEV)


AVG_SPEC = WaveformToAvgSpec(n_fft=128, hop_length=2, device=DEV).forward


def waveform_to_residual(signals, original_lens=None):
    
    if original_lens is None:
        original_lens = [signals.shape[-1]]

    # Apply filter and transformation, and calculate residual
    # print(f'Siganls shape: {signals.shape}')
    transformed_features = AVG_SPEC(signals, original_lens)
    filtered_signals = FILTER.forward(signals)
    transformed_filtered_features = AVG_SPEC(filtered_signals, original_lens)
    
    residuals = transformed_features - transformed_filtered_features

    return residuals