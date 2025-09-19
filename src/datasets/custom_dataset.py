import gc
import torch
from torchaudio import load
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from torchaudio.transforms import LFCC, MelSpectrogram
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
        self.resampler = Resample(self.sample_rate, self.target_sample_rate)
        self.postprocess = postprocess
        self.seed = seed
        
        self.corruption_type = corruption_type
        reverb_path = "/USERSPACE/DATASETS/LibriSpeech/reverb.csv"
        self.reverb = AddReverb(reverb_prob=1, csv_file=reverb_path, rir_scale_factor=scale_factor)
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
        # print(sample_uri)
        label = row["label"]
        try:
            waveform, samplerate = load(sample_uri, normalize=False)
            if waveform.dtype == torch.int16:  # WAV PCM16
                waveform = waveform.float() / 32768.0
            elif waveform.dtype == torch.float32:  # MP3 (already float [-1,1])
                pass  # do nothing
            else:
                raise ValueError(f"Unexpected dtype {waveform.dtype}")
            
            # waveform, samplerate = load(sample_uri, normalize=True)
            # print(waveform.shape, "entro")
            # waveform = waveform.float() / 32768.0  # manual normalization from PCM16
        except Exception as e:
            print(f"Error reading file: {sample_uri}")
            raise e
        waveform = waveform.float()
        
        if self.corruption_type == 1:
            waveform = self.reverb(waveform, torch.tensor([1.0]))
            '''
            import os
            torchaudio.save(os.path.basename(sample_uri), audio.cpu(), samplerate)  # , encoding="PCM_S", bits_per_sample=16)
            print(waveform.shape, audio.shape, sample_uri, os.path.basename(sample_uri), self.target_sample_rate, samplerate)
            print(afsasf)
            '''

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


def waveform_to_residual(signals, filter_fn, trans_fn, original_lens=None):
    
    if original_lens is None:
        original_lens = [signals.shape[-1]]

    # Apply filter and transformation, and calculate residual
    # print(f'Siganls shape: {signals.shape}')
    transformed_features = trans_fn(signals, original_lens)
    filtered_signals = filter_fn.forward(signals)
    transformed_filtered_features = trans_fn(filtered_signals, original_lens)
    
    residuals = transformed_features - transformed_filtered_features

    return residuals