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


class CustomDataset(Dataset):

    def __init__(self, dataset_df, sample_rate, target_sample_rate, model, classification_type, mean, std, seed, 
                postprocess=None, corruption_type=0, scale_factor=1.0, coef=None, n_fft=None, hop_length=None) -> None:

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
        self.coef = coef
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.corruption_type = corruption_type

        if model in ["resnet", "se-resnet", "lcnn", "x-vector"]:            
            self.transform = LFCC(
                                n_filter=20,
                                n_lfcc=60,
                                speckwargs={
                                    "n_fft": 512,
                                    "win_length": int(0.025 * self.target_sample_rate),
                                    "hop_length": int(0.01 * self.target_sample_rate)
                                }
                            )# .to(DEV)


        elif model in ["fingerprint", "fingerprint_2"]:
            self.transform = Spectrogram(
                                    n_fft = n_fft,
                                    hop_length = hop_length
                                    )# .to(DEV)
            self.filter_fn = filter_fn(1, coef)

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
        waveform = waveform.float()# .to(DEV)
        # print(self.sample_rate, self.target_sample_rate)
        waveform = self.resampler(waveform)

        if self.model in ["fingerprint", "fingerprint_2"]:
            filt_feat = self.filter_fn.forward(waveform)
            waveform, filt_feat = self.match_length(waveform, filt_feat) 
            trans_feat = self.transform(waveform).squeeze(0)
            trans_filt_feat = self.transform(filt_feat).squeeze(0)
            features = trans_feat - trans_filt_feat

        elif self.transform is not None:
            features  = self.transform(waveform).squeeze(0)
            # print(features.shape)
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

    def match_length(self, a: torch.Tensor, b: torch.Tensor):
        """Trim both tensors along the last dimension to the same minimum length."""
        min_len = min(a.shape[-1], b.shape[-1])
        return a[..., :min_len], b[..., :min_len]
