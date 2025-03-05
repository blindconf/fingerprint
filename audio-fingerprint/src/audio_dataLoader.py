import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import random

class AudioDataSet(Dataset):
    """
    A custom dataset class for loading and processing audio data.
    
    :param annotation_file: Path to the CSV file containing the audio file paths and annotations.
    :param target_sample_rate: The desired sample rate to which audio will be resampled.
    :param train_nrows: Number of rows to load from the annotation file (used for training).
    :param snr: Signal-to-noise ratio (optional), used for selecting specific noisy audio files.
    :param device: Device to load the data onto (e.g., 'cpu' or 'cuda').
    """
    def __init__(self, 
                annotation_file, 
                target_sample_rate,
                train_nrows,
                deterministic = None,
                snr = None,
                device='cpu',
                ):
        self.full_data = pd.read_csv(annotation_file, header=None)
        self.annotations = self.full_data.sample(n=train_nrows, random_state=deterministic)  # None = Random sampling
        self.snr = snr
        self.device = device
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        :return: Length of the dataset (number of audio samples).
        """
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
        Loads and processes an audio sample.
        
        :param index: Index of the sample to retrieve.
        :return: Processed audio signal, target sample rate, and audio sample path.
        """
        audio_sample_path = self._get_audio_sample_path(index)
        # if self.snr != None:
        #    audio_sample_path = audio_sample_path.replace("wavefake_noise", f"wavefake_noise/high_snr_{self.snr}_{self.snr}")
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)
        return signal, self.target_sample_rate, audio_sample_path

    def _cut_if_necessary(self, signal):
        """
        Cuts the signal to the specified number of samples if it exceeds the limit.
        
        :param signal: Audio signal to be cut.
        :return: Cut audio signal.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Pads the signal to the required number of samples if it is too short.
        
        :param signal: Audio signal to be padded.
        :return: Padded audio signal.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        """
        Resamples the signal to the target sample rate if necessary.
        
        :param signal: Audio signal to be resampled.
        :param sr: Original sample rate of the audio signal.
        :return: Resampled audio signal.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
        Mixes down multi-channel audio signals to a single channel (mono).
        
        :param signal: Audio signal to be mixed down.
        :return: Mono audio signal.
        """
        if signal.shape[0] > 1 :
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        """
        Retrieves the file path of the audio sample at the given index.
        
        :param index: Index of the audio sample in the annotations.
        :return: Path to the audio sample.
        """
        return self.annotations.iloc[index, 0]

def collate_fn(batch):
    """
    Custom collate function to pad audio signals in a batch to the same length.
    
    :param batch: A batch of data consisting of tuples (signal, target_sample_rate, audio_sample_path).
                  - signal: The audio signal (tensor).
                  - target_sample_rate: The sample rate of the audio signal.
                  - audio_sample_path: The file path of the audio sample.
    
    :return: 
        - signals: Tensor of padded signals, all having the same length (batch_size, max_length).
        - original_lengths: List of original lengths of each signal before padding.
        - target_sample_rate: The target sample rate, same for all signals in the batch.
        - audio_sample_paths: List of file paths corresponding to the audio samples.
    """
    signals, target_sample_rate, audio_sample_paths = zip(*batch)    
    # Find the length of the longest signal in the batch
    max_length = max(signal.shape[1] for signal in signals)
    # Pad all signals to the max length
    padded_signals = []
    original_lengths = []
    for signal in signals:
        original_lengths.append(signal.shape[1])
        if signal.shape[1] < max_length:
            pad_size = max_length - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, pad_size))
        padded_signals.append(signal)      
    # Stack the signals into a batch
    signals = torch.stack(padded_signals)    
    return signals, original_lengths, target_sample_rate, audio_sample_paths

def create_data_loader(train_data, batch_size):
    """
    Create a DataLoader for batching and shuffling the training data.
    
    :param train_data: The dataset to load, typically an instance of AudioDataSet.
    :param batch_size: The number of samples per batch to load.
    
    :return: DataLoader object that yields batches of data from train_data.
    """
    train_dL = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dL