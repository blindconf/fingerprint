import gc
import torch
from torchaudio import load
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from src.datasets.filters import filter_fn
from torchaudio.transforms import LFCC, MelSpectrogram, MFCC, Spectrogram
from src.training.invariables import DEV
import os
import warnings
from speechbrain.processing.speech_augmentation import AddReverb


class CustomDataset(Dataset):

    def __init__(self, dataset_df, sample_rate, target_sample_rate, model, classification_type, mean, std, seed, postprocess=None, corruption_type=0, scale_factor=1.0) -> None:

        self.df = dataset_df
        self.model = model
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.classification_type = classification_type
        self.mean = mean
        self.std = std
        # Resampler (will be identity if sr == target_sr)
        self.resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        self.postprocess = postprocess
        self.seed = seed
        
        self.corruption_type = corruption_type
        reverb_path = "/USERSPACE/DATASETS/LibriSpeech/reverb.csv"
        self.reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=scale_factor)
        '''
        if self.corruption_type == 2:
            # 320 kbps → "near-CD quality", very little audible difference from WAV for most people.
            # 192 kbps → good balance, most streaming platforms (like Spotify free tier) use this.
            # 128 kbps → standard web/voice quality, but some artifacts in music.
            # 64 kbps or lower → intelligible speech, but noticeable degradation.
            if scale_factor == 192:
                self.bitrate="192k"
            elif scale_factor == 128:
                self.bitrate="128k"
            elif scale_factor == 64:
                self.bitrate="64k"
            else:
                self.bitrate=scale_factor
        '''
        if model in ["resnet", "se-resnet", "lcnn", "x-vector"]:
            self.transform = LFCC(
                                n_filter=20,
                                n_lfcc=60,
                                speckwargs={
                                    "n_fft": 512,
                                    "win_length": int(0.025 * self.target_sample_rate),
                                    "hop_length": int(0.01 * self.target_sample_rate)
                                }
                            )
        elif model == "vfd-resnet":
            mel = MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=2048,
                hop_length=300,
                win_length=1200,
                n_mels=80,
                f_min=0,
                f_max=12000,
                window_fn=torch.hamming_window
            )
            self.transform = lambda x: torch.log(mel(x) + 1e-6)
        else:
            self.transform = None


    def __getitem__(self, index):

        row = self.df.iloc[index]
        sample_uri, label = row["path"], row["label"]

        waveform, sr = load(sample_uri)
        waveform = waveform.float()
        # print(self.sample_rate, self.target_sample_rate)
        waveform = self.resampler(waveform)

        if self.transform is not None:
            features  = self.transform(waveform).squeeze(0)
        else:
            features = waveform

        '''
        # Normalize if stats provided
        if self.mean is not None and self.std is not None:
            features = (features - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        '''
        if self.postprocess is not None:
            # '''
            if features.shape[1] < 64:
                repeat_factor = (64 // features.shape[1]) + 1
                features = features.repeat(1, repeat_factor)
                features = features[:, :64]
            # '''
            features = self.postprocess(features)
                
        return features, torch.tensor(label, dtype=torch.long)

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

