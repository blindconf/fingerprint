import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import mel_spec_to_spec, correlation_score, mahalanobis_score, pad_and_concatenate
from scipy.ndimage import binary_erosion, binary_dilation

from src.prnu import extract_single
import noisereduce as nr
import matplotlib.pyplot as plt
from noisereduce.torchgate import TorchGate as TG
import torchaudio
import scipy.signal as signal
import librosa
from math import ceil
import warnings
from torchaudio.functional import allpass_biquad,band_biquad, bandpass_biquad, bandreject_biquad, equalizer_biquad, filtfilt
from torchaudio.functional import highpass_biquad, lfilter, lowpass_biquad, treble_biquad
from scipy.signal import butter 
from .utils import TimeInvFIRFilter, preemphasis, extract_name
from scipy.spatial.distance import mahalanobis

import encodec 
import time 

# from src.filters import OracleFilter


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AllPassBiQuadFilter:
    name = "AllPassBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q 
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = allpass_biquad(batch, self.sample_rate, self.central_freq, self.Q)
        return batch.unsqueeze(0)

class BandBiQuadFilter:
    name = "BandBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707, const_skirt_gain: bool = False):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.const_skirt_gain = const_skirt_gain
        
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = band_biquad(batch, self.sample_rate, self.central_freq, self.Q, self.const_skirt_gain)
        return batch.unsqueeze(0)

class BandPassBiQuadFilter:
    name = "BandPassBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707, const_skirt_gain: bool = False):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.const_skirt_gain = const_skirt_gain
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = bandpass_biquad(batch, self.sample_rate, self.central_freq, self.Q, self.const_skirt_gain)
        return batch.unsqueeze(0)

class BandRejectBiQuadFilter:
    name = "BandRejectBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = bandreject_biquad(batch, self.sample_rate, self.central_freq, self.Q)
        return batch.unsqueeze(0)

class EqualizerBiQuadFilter:
    name = "EqualizerBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707, gain: float = 0.0):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.gain = gain
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = equalizer_biquad(batch, self.sample_rate, self.central_freq, self.Q, self.gain)
        return batch.unsqueeze(0)

class FiltFiltFilter:
    name = "FiltFiltFilter"
    def __init__(self, sample_rate, butter_order, butter_freq, filter_type, clamp=True, device="cuda"):
        self.sample_rate = sample_rate
        self.device = device
        assert filter_type in ['lowpass', 'highpass', 'bandpass', 'bandstop'], "Filter type should be lowpass, highpass, bandpass or bandstop" 
        butter_b, butter_a = butter(butter_order, butter_freq, btype=filter_type, fs=self.sample_rate)
        self.butter_b = torch.tensor(butter_b, dtype=torch.float32).to(device)
        self.butter_a = torch.tensor(butter_a, dtype=torch.float32).to(device)
        self.clamp = clamp 
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = filtfilt(batch, self.butter_a, self.butter_b, clamp=self.clamp)
        return batch.unsqueeze(0)

class HighPassBiQuadFilter:
    name = "HighPassBiQuadFilter"
    def __init__(self, sample_rate, cutoff_freq, Q: float = 0.707):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.Q = Q 
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = highpass_biquad(batch, self.sample_rate, self.cutoff_freq, self.Q)
        return batch.unsqueeze(0)

class LFilterFilter:
    name = "LFilterFilter"
    def __init__(self, sample_rate, butter_order, butter_freq, filter_type, clamp=True, device="cuda"):
        self.sample_rate = sample_rate
        self.device = device
        assert filter_type in ['lowpass', 'highpass', 'bandpass', 'bandstop'], "Filter type should be lowpass, highpass, bandpass or bandstop" 
        butter_b, butter_a = butter(butter_order, butter_freq, btype=filter_type, fs=self.sample_rate)
        self.butter_b = torch.tensor(butter_b, dtype=torch.float32).to(device)
        self.butter_a = torch.tensor(butter_a, dtype=torch.float32).to(device)
        self.clamp = clamp 
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = lfilter(batch, self.butter_a, self.butter_b, clamp=self.clamp)
        return batch.unsqueeze(0)
    
class LowPassBiQuadFilter:
    name = "LowPassBiQuadFilter"
    def __init__(self, sample_rate, cutoff_freq, Q: float = 0.707):
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.Q = Q 
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = lowpass_biquad(batch, self.sample_rate, self.cutoff_freq, self.Q)
        return batch.unsqueeze(0)

class TrebleBiQuadFilter:
    name = "TrebleBiQuadFilter"
    def __init__(self, sample_rate, central_freq, Q: float = 0.707, gain: float = 0.0):
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.Q = Q
        self.gain = gain
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.squeeze(0)
        batch = treble_biquad(batch, self.sample_rate, central_freq=self.central_freq, Q=self.Q, gain=self.gain)
        return batch.unsqueeze(0)



class MorphologicalFilter:
    """ Credits: https://github.com/joeljose/audio_denoising """
    name = "MorphologicalFilter"
    def __init__(self, sample_rate, transf, n_fft, threshold, amp, invers): # , , mel_spec_to_spec=True):
        self.threshold = threshold
        # self.mel_spec_to_spec = mel_spec_to_spec
        self.sample_rate = sample_rate
        self.transf = transf
        self.invers = invers
        self.n_fft = n_fft
        self.amp = amp

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(torch.abs(batch))
        '''
        spectrogram = self.transf(batch)
        Zxx = spectrogram[0].cpu()
        zmax_t = torch.max(Zxx)
        gray_spect = Zxx * (255/zmax_t)
        thresh_spect = np.where(gray_spect > self.threshold, 1, 0)
        mask_spect = binary_erosion(thresh_spect, iterations=1)
        mask_spect = binary_dilation(mask_spect, iterations=2)
        Rxx = np.where(mask_spect==1,Zxx*self.amp ,Zxx/self.amp)
        Rxx = torch.from_numpy(Rxx)[None, :].to('cuda')
        xrec = self.invers(Rxx)
        xrec = (xrec / torch.max(torch.abs(xrec)))*max_value
        Rxx = self.transf(xrec)
        # torchaudio.save("example.wav", xrec.float().cpu(), self.sample_rate)
        '''
        #'''
        freq, ti, Zxx = signal.stft(batch.cpu(), fs=self.sample_rate, nperseg=self.n_fft) 
        Zxx = Zxx[0]
        zmax=(np.max(np.abs(Zxx)))        
        gray_spect=np.abs(Zxx)*(255/zmax)
        thresh_spect = np.where(gray_spect > self.threshold, 1, 0)
        mask_spect = binary_erosion(thresh_spect, iterations=1)
        mask_spect = binary_dilation(mask_spect, iterations=2)
        Rxx = np.where(mask_spect==1,Zxx*self.amp ,Zxx/self.amp)
        # _, xrec = signal.istft(Rxx, self.sample_rate)
        Rxx = np.abs(Rxx)
        Rxx = torch.from_numpy(Rxx)[None, :].to('cuda')
        # xrec = torch.from_numpy(xrec)[None, :].to('cuda')
        # xrec = (xrec / torch.max(torch.abs(xrec)))*max_value
        # Rxx = self.transf(xrec)
        # torchaudio.save("example.wav", xrec.float().cpu(), self.sample_rate)
        # print(safsa)
        #'''
        '''
        fig, axs = plt.subplots(3, 1)
        axs[0].set_ylabel("freq_bin")
        axs[0].imshow(librosa.power_to_db(gray_spect), origin="lower", aspect="auto", interpolation="nearest")
        axs[1].set_ylabel("freq_bin")
        axs[1].imshow(librosa.power_to_db(thresh_spect), origin="lower", aspect="auto", interpolation="nearest")
        axs[2].set_ylabel("freq_bin")
        axs[2].imshow(librosa.power_to_db(mask_spect), origin="lower", aspect="auto", interpolation="nearest")
        fig.tight_layout()
        plt.savefig('example.png')
        '''
        return Rxx

    def spect_to_audio(self, batch: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(torch.abs(batch))
        freq, ti, Zxx = signal.stft(batch.cpu(), fs=self.sample_rate, nperseg=self.n_fft) 
        Zxx = Zxx[0]
        zmax=(np.max(np.abs(Zxx)))        
        gray_spect=np.abs(Zxx)*(255/zmax)
        thresh_spect = np.where(gray_spect > self.threshold, 1, 0)
        mask_spect = binary_erosion(thresh_spect, iterations=1)
        mask_spect = binary_dilation(mask_spect, iterations=2)
        Rxx = np.where(mask_spect==1,Zxx*self.amp ,Zxx/self.amp)
        _, xrec = signal.istft(Rxx, self.sample_rate)
        xrec = torch.from_numpy(xrec)[None, :].to('cuda')
        xrec = (xrec / torch.max(torch.abs(xrec)))*max_value        
        return xrec.float().cpu()

class MarrasFilter:
    """ Filter used in https://arxiv.org/abs/1812.11842"""
    name = "MarrasFilter"
    def __init__(self, transf, levels: int = 4, sigma: float = 5, wdft_sigma: float = 0):
        """
        :param levels: number of wavelet decomposition levels
        :param sigma: estimated noise power
        :param wdft_sigma: estimated DFT noise power
        """
        self.transf = transf
        self.levels = levels
        self.sigma = sigma
        self.wdft_sigma = wdft_sigma

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Input data should have the shape (B, T, F) or (T, F)
        :param spectrogram:
        :return:
        """
        # spectrogram = self.transf(batch)#
        spectrogram = batch 
        # print(spectrogram, spectrogram.shape)
        '''
        if len(spectrogram.shape) == 3:
            residuals = []
            for sample in spectrogram:
                residuals.append(extract_single(sample, self.levels, self.sigma, self.wdft_sigma))
            return residuals
        elif len(spectrogram.shape) == 2:
            return extract_single(spectrogram, self.levels, self.sigma, self.wdft_sigma)
        else:
            raise ValueError("Input data should have the shape (B, T, F) or (T, F)")
        '''
        return extract_single(spectrogram, self.levels, self.sigma, self.wdft_sigma)

class BaselineFilter:
    """ Filter that simply returns zeros. 
    Therefore, the fingerprinting approach simply computes the mean on the training set and 
    compares the correlation to the test samples. 
    """
    name = "BaselineFilter"
    def __init__(self, transf, cuda=True):
        self.transf = transf 
        if cuda:
            self.device = torch.device('cuda')

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transf(batch) # TODO: I just need the space of spectogram, no need to compute it
        return torch.zeros_like(spectrogram).to(self.device)


class LowHighFreq:
    name = "LowHighFreq"
    def __init__(self, sample_rate, freq_bins, threshold, low_freq_flg, transf, invers):
        self.sample_rate = sample_rate
        self.freq_bins = freq_bins
        self.cutoff = max(1, int(self.freq_bins * threshold / 100))
        self.transf = transf
        self.low_freq_flg = low_freq_flg
        self.invers = invers

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Input data should have the shape (B, 1, F, T) or (1, F, T)
        :param spectrogram:
        :return:
        """
        spectrogram = self.transf(batch)
        if self.low_freq_flg:
            spectrogram = spectrogram[:, :self.cutoff, :]
        else:
            spectrogram = spectrogram[:, self.cutoff:, :]
        return spectrogram
        '''
        # print(spectrogram.shape)
        if len(spectrogram.shape) == 4: # i.e. batchwise
            assert spectrogram.shape[2] == self.num_frequency_bins
            return spectrogram[:, :, self.cutoff:, :]
            # return spectrogram[:, :, self.cutoff:, :self.time_bound] # remove low frequencies
        elif len(spectrogram.shape) == 3:
            assert spectrogram.shape[1] == self.num_frequency_bins
            return spectrogram[:, self.cutoff:, :]
            # return spectrogram[:, self.cutoff:, :self.time_bound]
        else:
            raise ValueError("Input data should have the shape (B, 1, F, T) or (1, F, T)")
        '''

class SpectralGating:
    name = "SpectralGating"
    def __init__(self, sample_rate, nonstationary, transf=None):
        self.sample_rate = sample_rate
        self.nonstationary = nonstationary
        self.transf = transf

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Input data should have the shape (B, 1, F, T) or (1, F, T)
        :param spectrogram:
        :return:
        """
        # wavs = torch.from_numpy(nr.reduce_noise(y=batch.cpu(), sr=self.sample_rate, stationary=True)).to('cuda')
        # Create TorchGating instance
        tg = TG(sr=self.sample_rate, nonstationary=self.nonstationary).to('cuda')
        wavs = tg(batch)
        # print(wavs_2)
        # print(wavs)
        # residual = batch - wavs
        # print(residual)
        # torchaudio.save("example.wav", wavs.float().cpu(), self.sample_rate)
        # torchaudio.save("example_1.wav", wavs_2.float().cpu(), self.sample_rate)
        
        '''
        fig, axs = plt.subplots(3, 1)
        axs[0].set_ylabel("freq_bin")
        axs[0].imshow(librosa.power_to_db(self.transf(wavs_2)[0].cpu()), origin="lower", aspect="auto", interpolation="nearest")
        axs[1].set_ylabel("freq_bin")
        axs[1].imshow(librosa.power_to_db(self.transf(residual)[0].cpu()), origin="lower", aspect="auto", interpolation="nearest")
        axs[2].set_ylabel("freq_bin")
        axs[2].imshow(librosa.power_to_db(self.transf(wavs)[0].cpu()), origin="lower", aspect="auto", interpolation="nearest")
        fig.tight_layout()
        plt.savefig('example.png')
        '''
        spectrogram = self.transf(wavs)
        # spectrogram = spectrogram.squeeze(0)
        return spectrogram # spectrogram

    def spect_to_audio(self, batch: torch.Tensor) -> torch.Tensor:
        tg = TG(sr=self.sample_rate, nonstationary=True).to('cuda')
        wavs = tg(batch)
        # wavs = torch.from_numpy(nr.reduce_noise(y=batch.cpu(), sr=self.sample_rate, stationary=True)).to('cuda')
        return wavs.float().cpu()

class MovingAverage:
    name = "MovingAverage"
    def __init__(self, sample_rate, transf, window_size=1):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.transf = transf

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        window = np.ones(self.window_size)/float(self.window_size)
        # print(batch.shape)
        convolved_data = np.convolve(torch.squeeze(batch).cpu(), window, 'same')
        convolved_data = torch.unsqueeze(torch.from_numpy(convolved_data).to('cuda'), 0)
        # print("Original data: {}".format(batch))
        # print("Moving average filtered data with window size {}: {}".format(self.window_size, convolved_data))
        return self.transf(convolved_data)

    def spect_to_audio(self, batch: torch.Tensor) -> torch.Tensor:
        window = np.ones(self.window_size)/float(self.window_size)
        # print(batch.shape)
        convolved_data = np.convolve(torch.squeeze(batch).cpu(), window, 'same')
        convolved_data = torch.unsqueeze(torch.from_numpy(convolved_data).to('cuda'), 0)   
        return convolved_data.float().cpu()

class SFA_filter:
    "Credits: https://sklearn-sfa.readthedocs.io/en/latest/user_guide.html"
    name = "SFA"
    # def __init__(self, sample_rate, transf, sfa_transformer, pf, component):
    def __init__(self, sample_rate, transf, sfa_transformer, component):
        self.sample_rate = sample_rate
        self.transf = transf
        self.sfa_transformer = sfa_transformer
        # self.pf = pf
        self.component = component

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        
        max_value = torch.max(torch.abs(batch)).cpu()
        # spect_input = self.transf(batch).squeeze(0)
        
        batch = batch.squeeze(0).cpu()
        batch = batch.reshape(-1, 1)
        # expanded_data = self.pf.fit_transform(batch)
        # self.sfa_transformer.fit(expanded_data)
        # extracted_features = self.sfa_transformer.transform(expanded_data)
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted_features = self.sfa_transformer.fit_transform(expanded_data)
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted_features = self.sfa_transformer.fit_transform(batch)
        # print(extracted_features, extracted_features.shape)
        slow_signal = torch.from_numpy(extracted_features[:, self.component - 1])[None, :]
        slow_signal = (slow_signal / torch.max(torch.abs(slow_signal)))*max_value
        # print(slow_signal, slow_signal.shape)
        # torchaudio.save("example.wav", slow_signal.float().cpu(), self.sample_rate)
        # print(spect_input.shape)
        '''
        fig, axs = plt.subplots(2, 1)
        axs[0].set_ylabel("freq_bin")
        axs[0].imshow(librosa.power_to_db(spect_input.cpu()), origin="lower", aspect="auto", interpolation="nearest")
        axs[1].set_ylabel("freq_bin")
        axs[1].imshow(librosa.power_to_db(self.transf(slow_signal.to('cuda')).squeeze(0).cpu()), origin="lower", aspect="auto", interpolation="nearest")
        fig.tight_layout()
        plt.savefig('example.png')
        '''
        return self.transf(slow_signal.to('cuda'))

    def spect_to_audio(self, batch: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(torch.abs(batch)).cpu()     
        batch = batch.squeeze(0).cpu()
        batch = batch.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted_features = self.sfa_transformer.fit_transform(batch)
        # print(extracted_features, extracted_features.shape)
        slow_signal = torch.from_numpy(extracted_features[:, self.component - 1])[None, :]
        slow_signal = (slow_signal / torch.max(torch.abs(slow_signal)))*max_value 
        return slow_signal.float().cpu()


class OracleLFCCFilter:
    name = "Oracle_MFCC"
    def __init__(self, sample_rate, n_lfcc, speckwargs, path_real_dir, device="cuda"):
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc  
        self.speckwargs = speckwargs 
        self.path_real_dir = path_real_dir
        self.device = device
        self.transf = torchaudio.transforms.LFCC(sample_rate=self.sample_rate, n_lfcc=self.n_lfcc, speckwargs=self.speckwargs).to(device)

    def forward(self, batch: torch.Tensor, path: str) -> torch.Tensor:
        base_name = os.path.basename(path)
        if base_name.endswith("_gen.wav"):
            base_name = base_name.replace("_gen.wav", ".wav")
        elif base_name.endswith("_generated.wav"):
            base_name = base_name.replace("_generated.wav", ".wav")
            
        path_real = f"{self.path_real_dir}{base_name}"
        signal, sr = torchaudio.load(path_real)
        signal = signal.to(self.device)
        # TODO below shouldnt be necessary 
        #signal = self._resample_if_necessary(signal, sr)
        #signal = self._mix_down_if_necessary(signal)
        spectrogram_real = self.transf(signal)
        return spectrogram_real 
    
class OracleMFCCFilter:
    name = "Oracle_MFCC"
    def __init__(self, sample_rate, n_mfcc, melkwargs, path_real_dir, device="cuda"):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc  
        self.melkwargs = melkwargs 
        self.path_real_dir = path_real_dir
        self.device = device
        self.transf = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, melkwargs=self.melkwargs).to(device)

    def forward(self, batch: torch.Tensor, path: str) -> torch.Tensor:
        base_name = os.path.basename(path)
        if base_name.endswith("_gen.wav"):
            base_name = base_name.replace("_gen.wav", ".wav")
        elif base_name.endswith("_generated.wav"):
            base_name = base_name.replace("_generated.wav", ".wav")
            
        path_real = f"{self.path_real_dir}{base_name}"
        signal, sr = torchaudio.load(path_real)
        signal = signal.to(self.device)
        # TODO below shouldnt be necessary 
        #signal = self._resample_if_necessary(signal, sr)
        #signal = self._mix_down_if_necessary(signal)
        spectrogram_real = self.transf(signal)
        return spectrogram_real 
                                                    
class EncodecFilter:
    name = "EncodecFilter"
    def __init__(self, model, bandwidth, computations_samplewise=False, device="cuda"):
        self.model = model.to(device) # TODO Check whether it works... 
        self.bandwidth = bandwidth
        self.device = device
        self.model.set_target_bandwidth(bandwidth)   
        self.computations_samplewise = computations_samplewise     
    
    def forward(self, batch: torch.Tensor, batch_sample_rate) -> torch.Tensor:
        with torch.no_grad():    
            sample_transform = torchaudio.transforms.Resample(batch_sample_rate, self.model.sample_rate).to(self.device)
            batch_converted = sample_transform(batch)
            # For some reason I observed the following behavior:
            # self.model(batch_converted)[0] != self.model(batch_converted[0][None])[0] 
            # ToDo: I dont know why this is the case... A hot fix could be to just iterate through all items in batch_converted... 
            if self.computations_samplewise:
                output = []
                for i in range(batch_converted.size(0)):
                    output.append(self.model(batch_converted[i][None])[0])
                output = torch.stack(output)
            else:
                output = self.model(batch_converted)#[0] 
            resample_transform = torchaudio.transforms.Resample(self.model.sample_rate, batch_sample_rate).to(self.device)
            output = resample_transform(output) 
        return output

class GriffinLimFilter:
    name = "GriffinLim"
    def __init__(self, to_spec, n_fft, hop_length):
        self.to_spec = to_spec
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        spec = self.to_spec(batch)
        reconstructed_waveform = torchaudio.transforms.GriffinLim(n_fft=self.n_fft, 
                                                           hop_length=self.hop_length,
                                                           length=batch.size(1)).to("cuda")(spec)
        reconstructed_spec = self.to_spec(reconstructed_waveform) 
        residual = spec - reconstructed_spec
        residual_db = 10. * torch.log(spec + 10e-13) - 10. * torch.log(reconstructed_spec + 10e-13) 
        return residual.squeeze(0), residual_db.squeeze(0) 

class WaveformToAvgMFCC: 
    def __init__(self, sample_rate, n_mfcc, melkwargs, device="cuda"):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.melkwargs = melkwargs
        
        self.transf = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, melkwargs=self.melkwargs).to(device)
        self.device = device
        
    def forward(self, batch: torch.Tensor, original_audio_lengths: list) -> torch.Tensor: 
        num_samps, num_channels, max_audio_len = batch.shape
        original_audio_lengths = torch.tensor(original_audio_lengths, device=self.device)        
        mfcc_list = []
        for i in range(num_samps):
            # Extract audio up to its original length
            audio_len = original_audio_lengths[i]
            audio = batch[i, :, :audio_len]  # Shape: (num_channels, audio_len)
            
            # Apply MFCC transformation on the extracted segment
            mfcc = self.transf(audio).unsqueeze(0) # Shape: (n_mfcc, transformed_len)
            # Store the result in a list
            mfcc_list.append(mfcc)

        # Determine maximum transformed length across all samples
        max_mfcc_len = max(mfcc.shape[3] for mfcc in mfcc_list)
        
        # Initialize padded MFCC tensor with NaNs
        mfcc_padded = torch.full((num_samps, 1, self.n_mfcc, max_mfcc_len), float('nan'), device=self.device)
        
        # Copy each MFCC to the padded tensor
        for i, mfcc in enumerate(mfcc_list):
            mfcc_padded[i, :, :, :mfcc.shape[3]] = mfcc
        
        # Compute mean across the time dimension, ignoring padding if needed
        result = torch.nanmean(mfcc_padded, dim=3)  # Shape: (num_samps, n_mfcc)
        return result

class WaveformToAvgMel:
    def __init__(self, 
                 sample_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 to_db=True,
                 device="cuda"):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.transf = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                           n_fft=self.n_fft,
                                                           hop_length=self.hop_length,
                                                           n_mels=self.n_mels).to(device)
        self.device = device  
        self.to_db = to_db   
        
    def forward(self, batch: torch.Tensor, original_audio_lengths: list) -> torch.Tensor: 
        max_audio_len = batch.shape[2]
        num_samps = batch.shape[0]
        mask = torch.arange(max_audio_len).expand(num_samps, max_audio_len) >= torch.tensor(original_audio_lengths).unsqueeze(1)
        batch[mask.unsqueeze(1)] = float('nan')

        # batch = batch.squeeze(0).to(self.device)
        mfcc = self.transf(batch)
        if self.to_db:
            mfcc = 10. * torch.log(mfcc + 10e-13)
        # energy = torch.mean(mfcc.squeeze(0), dim=1)  
        # return energy.unsqueeze(0)
        return torch.nanmean(mfcc, dim=3)

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
                 scoring="correlation",
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
        spectrograms = []
        filter_spectrograms = []
        residuals = []
        residuals_db = []
        real_energies_db = []
        # count = 0
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
                    raise NotImplementedError("To obtain ORACLE results, the fake audio training data must correspond to the original audio data.")
                avg_mfcc = self.transformation.forward(batch1[0], batch1[1])
                filtered_avg_mfcc = self.transformation.forward(batch2[0], batch2[1])
                residual_avg_mfcc = avg_mfcc - filtered_avg_mfcc
                residuals.append(residual_avg_mfcc) 
            residuals = pad_and_concatenate(residuals)
            # print(residuals.shape)  # Check the shape of the resulting tensor
            fingerprint = torch.mean(residuals, dim=0)
            # fingerprint = torch.mean(torch.stack(residuals), dim=0)
            # print(fingerprint.shape)
            if self.filter_trend_correction:
                fingerprint = fingerprint - self.trend  
            
            if self.scoring == "correlation":
                self.fingerprint = self.zero_mean_unit_norm(fingerprint)
            elif self.scoring == "mahalanobis":
                self.fingerprint = fingerprint
                # covariance = torch.cov(torch.stack(residuals).squeeze().T)
                covariance = torch.cov(residuals.squeeze().T)
                self.invcov = torch.inverse(covariance)
        else:            
            if self.filter_trend_correction:
                raise NotImplementedError("Trend correction is not implemented for the current filter")
                assert ds_real is not None, "ds_real was not specified"
                residuals_real = []
                for i in tqdm(ds_real, desc="Computing trend"):
                    batch = i[0]
                    # batch = preemphasis(i[0], 0.97)
                    # path = i[2]
                    
                    avg_feature = self.transformation.forward(batch)
                    filtered_batch = self.filter.forward(batch)
                    filtered_avg_feature = self.transformation.forward(filtered_batch)
                    residual_avg_feature = avg_feature - filtered_avg_feature
                    residuals_real.append(residual_avg_feature)
                residuals_real = pad_and_concatenate(residuals_real)
                # self.trend = torch.mean(torch.stack(residuals_real), dim=0)
                self.trend = torch.mean(residuals_real, dim=0)
                # print(residuals.shape)  # Check the shape of the resulting tensor
            for i in tqdm(dl, desc="Computing fingerprint"):
                batch = i[0]
                original_audio_lengths = i[1]
                batch_sample_rate = i[2][0]
                # batch = preemphasis(i[0], 0.97)
                # path = i[2]

                if self.filter.name == 'LowHighFreq':
                    filtered_spectrogram = self.filter.forward(batch)
                    filtered_spectrogram = filtered_spectrogram.squeeze(0)
                    filtered_spectrogram = torch.mean(filtered_spectrogram, dim=1)
                    residual_spectrogram = 10. * torch.log(filtered_spectrogram + 10e-13)
                    residuals.append(residual_spectrogram)
                elif self.filter.name == 'HighFreq-Spec':
                    filtered_spectrogram = self.filter.forward(batch)
                    filtered_spectrogram = filtered_spectrogram.squeeze(0)
                    residual_spectrogram = 10. * torch.log(filtered_spectrogram + 10e-13)
                    residuals.append(residual_spectrogram)
                elif self.filter.name == 'Oracle':
                    features = self.transformation.forward(batch)
                    filtered_batch = self.filter.forward(batch, path=path)
                    filtered_features = self.transformation.forward(filtered_batch)
                    residual_features = features - filtered_features
                    residuals.append(residual_features) 
                    
                    """
                    spectrogram_real = self.filter.forward(batch, path)
                    spectrogram_real = spectrogram_real.squeeze(0)
                    spectrogram = self.filter.transf(batch) 
                    spectrogram = spectrogram.squeeze(0)
                    
                    energy = torch.mean(spectrogram.squeeze(0), dim=1)
                    energy_real = torch.mean(spectrogram_real.squeeze(0), dim=1)
                    energy_db = 10. * torch.log(energy + 10e-13)
                    energy_real_db = 10. * torch.log(energy_real + 10e-13)
                    # TODO: Perhaps its better to not rescale to db? 
                    # Edit: Nope, its worse! Keep DB
                    residual_energy = energy_db - energy_real_db

                    residuals.append(residual_energy)
                    if self.reweight:
                        real_energies_db.append(energy_real_db)
                    """
                elif self.filter.name == 'Oracle_MFCC_Avg':
                    spectrogram_real = self.filter.forward(batch, path)
                    spectrogram_real = spectrogram_real.squeeze(0)
                    spectrogram = self.filter.transf(batch) 
                    spectrogram = spectrogram.squeeze(0)
                    energy = torch.mean(spectrogram.squeeze(0), dim=1)
                    energy_real = torch.mean(spectrogram_real.squeeze(0), dim=1)
                    residual_energy = energy - energy_real
                    residuals.append(residual_energy)
                    
                elif self.filter.name in ['Oracle_Spec', 'Oracle_MFCC', 'Oracle_LFCC']:
                    spectrogram_real = self.filter.forward(batch, path)
                    spectrogram_real = spectrogram_real.squeeze(0)
                    spectrogram = self.filter.transf(batch) 
                    spectrogram = spectrogram.squeeze(0)
                    
                    # TODO test whether I want to do db! 
                    # EDIT: Yes, I want to do db! It improves the performance 
                    if self.filter.name == 'Oracle_Spec':
                        spectrogram = 10. * torch.log(spectrogram + 10e-13)
                        spectrogram_real = 10. * torch.log(spectrogram_real + 10e-13)
                    
                    # bring both in same size 
                    if spectrogram.shape[1] != spectrogram_real.shape[1]:
                        if spectrogram.shape[1] > spectrogram_real.shape[1]:
                            spectrogram = spectrogram[:, :spectrogram_real.shape[1]]
                        else:
                            spectrogram_real = spectrogram_real[:, :spectrogram.shape[1]]
                    residual_spectrogram = spectrogram - spectrogram_real
                    residuals.append(residual_spectrogram)
                
                elif self.filter.name == "BaselineFilter":
                    spectrogram = self.transformation(batch) 
                    spectrogram = spectrogram.squeeze(0)
                    #spectrogram = torch.mean(spectrogram, dim=1) # TODO Test how I think it should be done 
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)
                    residuals.append(residual_spectrogram)
                elif self.filter.name == "MarrasFilter":
                    spectrogram = self.transformation(batch)
                    # spectrogram to decibel 
                    spectrogram = 10. * torch.log(spectrogram + 10e-13) 
                    # linearily scaling spectrogram to 0-255 
                    spectrogram = (spectrogram - torch.min(spectrogram)) / (torch.max(spectrogram) - torch.min(spectrogram)) * 255
                    spectrogram = spectrogram.cpu().numpy().astype(np.uint8).squeeze()
                    residual_spectrogram = torch.tensor(self.filter.forward(spectrogram))
                    residuals.append(residual_spectrogram)
                elif self.filter.name == "GriffinLim":
                    residual_spectrogram, residual_spectrogram_db = self.filter.forward(batch)
                    residuals.append(residual_spectrogram)
                    residuals_db.append(residual_spectrogram_db)
                elif self.filter.name in ["MovingAverage-Spec", "SFA-Spec"]:
                    spectrogram = self.transformation(batch)
                    filtered_spectrogram = self.filter.forward(batch)
                    spectrogram = spectrogram.squeeze(0)        
                    
                    filtered_spectrogram = filtered_spectrogram.squeeze(0)
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)  - 10. * torch.log(filtered_spectrogram + 10e-13) 

                    spectrograms.append(spectrogram)
                    filter_spectrograms.append(filtered_spectrogram)
                    residuals.append(residual_spectrogram)
                elif self.filter.name in ["AllPassBiQuadFilter", 
                                        "HighPassBiQuadFilter", 
                                        "LowPassBiQuadFilter", 
                                        "BandPassBiQuadFilter", 
                                        "BandRejectBiQuadFilter",
                                        "BandBiQuadFilter",
                                        "TrebleBiQuadFilter",
                                        "EqualizerBiQuadFilter",
                                        "LFilterFilter",
                                        "FiltFiltFilter", 
                                        "low_pass_filter", 
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter"]:
                    avg_mfcc = self.transformation.forward(batch, original_audio_lengths)
                    filtered_batch = self.filter.forward(batch)
                    filtered_avg_mfcc = self.transformation.forward(filtered_batch, original_audio_lengths)
                    residual_avg_mfcc = avg_mfcc - filtered_avg_mfcc
                    residuals.append(residual_avg_mfcc) 
                elif self.filter.name == "EncodecFilter":
                    features = self.transformation.forward(batch, original_audio_lengths)
                    filtered_batch = self.filter.forward(batch, batch_sample_rate)
                    filtered_features = self.transformation.forward(filtered_batch, original_audio_lengths)
                    residual_features = features - filtered_features
                    residuals.append(residual_features)
                    
                else:
                    
                    features = self.transformation(batch)
                    filtered_features = self.transformation(self.filter.forward(batch))
                    if filtered_features.shape[2] != features.shape[2]:
                        if filtered_features.shape[2] > features.shape[2]:
                            filtered_features = filtered_features[:, :,:features.shape[2]]
                        else:
                            features = features[:, :,:filtered_features.shape[2]]
                    
                    features = features.squeeze(0)
                    avg_features = torch.mean(features, dim=1)      
                    # TODO ab hier!!!      
                    # filtered_spectrogram = self.transformation(self.filter.forward(batch))
                    
                    filtered_spectrogram = filtered_spectrogram.squeeze(0)
                    filtered_spectrogram = torch.mean(filtered_spectrogram, dim=1)
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)  - 10. * torch.log(filtered_spectrogram + 10e-13) 

                    spectrograms.append(spectrogram)
                    filter_spectrograms.append(filtered_spectrogram)
                    residuals.append(residual_spectrogram)
            if self.filter.name == 'LowHighFreq':
                average_residuals = torch.stack(residuals)
                # average_residuals = 10. * torch.log(average_residuals + 10e-13)
                fingerprint = torch.mean(average_residuals, dim=0)
                if fingerprint.shape[0] != 1:
                    self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                else:
                    self.fingerprint = fingerprint
                    # print("test", self.fingerprint)
                    # print("freq_bins: ", self.filter.freq_bins)
                self.spectrograms_avg = None
                self.spect_filter_avg = None
            elif self.filter.name in ['Oracle', 'Oracle_MFCC_Avg']:
                average_residuals = torch.stack(residuals)
                fingerprint = torch.mean(average_residuals, dim=0)
                # Keep only the percentage of the fingerprint
                fingerprint = fingerprint[int(len(fingerprint) * self.keep_percentage[0]):int(len(fingerprint) * self.keep_percentage[1])]
                if self.reweight:
                    real_energies_db = torch.stack(real_energies_db)
                    real_energies_db = torch.mean(real_energies_db, dim=0)
                    self.real_energies_db = real_energies_db[int(len(real_energies_db) * self.keep_percentage[0]):int(len(real_energies_db) * self.keep_percentage[1])]
                    
                    fingerprint = fingerprint * self.real_energies_db
                
                
                if self.filter_trend_correction:
                    fingerprint = fingerprint - self.trend
                    
                # TODO is this implemented for all relevant filters?
                if self.scoring == "correlation":
                    self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                elif self.scoring == "mahalanobis":
                    self.fingerprint = fingerprint
                    covariance = torch.cov(torch.stack(residuals).squeeze().T)
                    self.invcov = torch.inverse(covariance)
                    
            elif self.filter.name in ['Oracle_Spec', 'Oracle_MFCC', 'Oracle_LFCC']: 
                self.min_size = min([r.shape[1] for r in residuals])
                residuals = [r[:, :self.min_size] for r in residuals]
                average_residuals = torch.stack(residuals)
                fingerprint = torch.mean(average_residuals, dim=0)
                self.fingerprint = self.zero_mean_unit_norm(fingerprint)
            
            elif self.filter.name in ["BaselineFilter", "MarrasFilter", 'HighFreq-Spec', 'MovingAverage-Spec', 'SFA-Spec']:
                # Crop all to min size: 
                self.min_size = min([r.shape[1] for r in residuals])
                residuals = [r[:, :self.min_size] for r in residuals]
                
                average_residuals = torch.stack(residuals)
                # average_residuals = 10. * torch.log(average_residuals + 10e-13)
                fingerprint = torch.mean(average_residuals, dim=0)
                if fingerprint.shape[0] != 1:
                    self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                else:
                    self.fingerprint = fingerprint
                self.spectrograms_avg = None
                self.spect_filter_avg = None
            elif self.filter.name == "GriffinLim":
                # Crop all to min size: 
                self.min_size = min([r.shape[1] for r in residuals])
                residuals = [r[:, :self.min_size] for r in residuals]
                residuals_db = [r[:, :self.min_size] for r in residuals_db]
                
                average_residuals = torch.stack(residuals)
                average_residuals_db = torch.stack(residuals_db)
                # average_residuals = 10. * torch.log(average_residuals + 10e-13)
                fingerprint = torch.mean(average_residuals, dim=0)
                fingerprint_db = torch.mean(average_residuals_db, dim=0)

                if fingerprint.shape[0] != 1:
                    self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                    self.fingerprint_db = self.zero_mean_unit_norm(fingerprint_db)
                else:
                    self.fingerprint = fingerprint
                self.spectrograms_avg = None
                self.spect_filter_avg = None
            elif self.filter.name in ["AllPassBiQuadFilter", 
                                        "HighPassBiQuadFilter", 
                                        "LowPassBiQuadFilter", 
                                        "BandPassBiQuadFilter", 
                                        "BandRejectBiQuadFilter",
                                        "BandBiQuadFilter",
                                        "TrebleBiQuadFilter",
                                        "EqualizerBiQuadFilter",
                                        "LFilterFilter",
                                        "FiltFiltFilter",
                                        "low_pass_filter",
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter",
                                        "EncodecFilter"]:
                residuals = pad_and_concatenate(residuals)
                # print(residuals.shape)  # Check the shape of the resulting tensor
                fingerprint = torch.mean(residuals, dim=0)
                # fingerprint = torch.mean(torch.stack(residuals), dim=0)
                # print(fingerprint.shape)
                if self.filter_trend_correction:
                    fingerprint = fingerprint - self.trend  
                
                if self.scoring == "correlation":
                    self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                elif self.scoring == "mahalanobis":
                    self.fingerprint = fingerprint
                    # covariance = torch.cov(torch.stack(residuals).squeeze().T)
                    covariance = torch.cov(residuals.squeeze().T)
                    self.invcov = torch.inverse(covariance)
            else:
                average_spectrograms = torch.stack(spectrograms)
                average_spectrograms = 10. * torch.log(average_spectrograms + 10e-13)
                fingerprint_spectrograms = torch.mean(average_spectrograms, dim=0)
                average_filters = torch.stack(filter_spectrograms)
                average_filters = 10. * torch.log(average_filters + 10e-13)
                fingerprint_filters = torch.mean(average_filters, dim=0)

                average_residuals = torch.stack(residuals)
                # average_residuals = 10. * torch.log(average_residuals + 10e-13)
                fingerprint = torch.mean(average_residuals, dim=0)
                self.fingerprint = self.zero_mean_unit_norm(fingerprint)
                self.spectrograms_avg = fingerprint_spectrograms
                self.spect_filter_avg = fingerprint_filters
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
                    raise NotImplementedError("To obtain ORACLE results, the fake audio testing data must correspond to the original audio data.")
                avg_mfcc = self.transformation.forward(batch1[0], batch1[1])
                filtered_avg_mfcc = self.transformation.forward(batch2[0], batch2[1])
                residual = avg_mfcc - filtered_avg_mfcc
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
                # audio = preemphasis(i[0], 0.97)
                path = i[2]
                if self.filter.name == "EncodecFilter":
                    filtered_audio = self.filter.forward(audio, batch_sample_rate)
                elif self.filter.name not in ['MarrasFilter', 'GriffinLim', 'Oracle', 'Oracle_Spec', 'Oracle_MFCC', 'Oracle_LFCC', 'Oracle_MFCC_Avg']:
                    filtered_audio = self.filter.forward(audio)            
                else:
                    filtered_audio = audio
                if self.filter.name == 'LowHighFreq':
                    residual_spectrogram = filtered_audio.squeeze(0)
                    # residual_spectrogram = 10. * torch.log(residual_spectrogram + 10e-13)
                    residual_spectrogram = torch.mean(residual_spectrogram, dim=1)
                    residual_spectrogram = 10. * torch.log(residual_spectrogram + 10e-13)
                    if self.scoring=="correlation":
                        if self.fingerprint.shape[0] != 1:
                            residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram)
                        score = correlation_score(self.fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(self.fingerprint, residual_spectrogram, self.invcov)

                elif self.filter.name == 'Oracle2':
                    filtered_audio = self.filter.forward(audio, path, test_mode=True)
                    filtered_features = self.transformation.forward(filtered_audio).squeeze(0)
                    features = self.transformation.forward(audio).squeeze(0)
                    residual = features - filtered_features 
                    
                    if self.filter_trend_correction:
                        residual = residual - self.trend
                    
                    if self.scoring=="correlation":
                        if cutoff is not None: 
                            residual = residual[:, :cutoff]
                        if self.fingerprint.shape[1] != cutoff: 
                            self.fingerprint = self.fingerprint[:, :cutoff]
                            self.fingerprint = self.zero_mean_unit_norm(self.fingerprint)
                        residual = self.zero_mean_unit_norm(residual)
                    
                        score = correlation_score(self.fingerprint, residual)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(self.fingerprint, residual, self.invcov)
                elif self.filter.name == 'Oracle_MFCC_Avg':
                    spectrogram_real = self.filter.forward(audio, path)
                    spectrogram_real = spectrogram_real.squeeze(0)
                    spectrogram = self.filter.transf(audio) 
                    spectrogram = spectrogram.squeeze(0)
                    
                    energy = torch.mean(spectrogram.squeeze(0), dim=1)
                    energy_real = torch.mean(spectrogram_real.squeeze(0), dim=1)
                    residual_energy = energy - energy_real
                    residual_energy = residual_energy[int(len(residual_energy) * self.keep_percentage[0]):int(len(residual_energy) * self.keep_percentage[1])]
                    if self.reweight:
                        # TODO another strategy would be to multiply by real_energies_db 
                        residual_energy = residual_energy * self.real_energies_db
                    
                    if self.scoring=="correlation":
                        residual_energy = self.zero_mean_unit_norm(residual_energy)
                        score = correlation_score(self.fingerprint, residual_energy)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(self.fingerprint, residual_energy, self.invcov)
                elif self.filter.name in ['Oracle_Spec', 'Oracle_MFCC', 'Oracle_LFCC']:
                    spectrogram_real = self.filter.forward(audio, path)
                    spectrogram_real = spectrogram_real.squeeze(0)
                    spectrogram = self.filter.transf(audio) 
                    spectrogram = spectrogram.squeeze(0)
                    
                    # bring to db 
                    if self.filter.name == 'Oracle_Spec':
                        spectrogram = 10. * torch.log(spectrogram + 10e-13)
                        spectrogram_real = 10. * torch.log(spectrogram_real + 10e-13)
                    
                    if spectrogram.shape[1] != spectrogram_real.shape[1]:
                        if spectrogram.shape[1] > spectrogram_real.shape[1]:
                            spectrogram = spectrogram[:, :spectrogram_real.shape[1]]
                        else:
                            spectrogram_real = spectrogram_real[:, :spectrogram.shape[1]]
                            
                    residual_spectrogram = spectrogram - spectrogram_real
                    min_size = min(self.fingerprint.shape[1], residual_spectrogram.shape[1])
                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram[:, :min_size])
                    fingerprint = self.fingerprint[:, :min_size]

                    if self.scoring=="correlation":
                        fingerprint = self.zero_mean_unit_norm(fingerprint)
                        score = correlation_score(self.fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual_spectrogram, self.invcov[:min_size, :min_size]) 
            
                elif self.filter.name == "BaselineFilter":
                    spectrogram = self.transformation(audio) 
                    spectrogram = spectrogram.squeeze(0)
                    spectrogram = spectrogram[:, :self.fingerprint.shape[1]]
                    #spectrogram = torch.mean(spectrogram, dim=1) # TODO Test how I think it should be done 
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)
                    if self.fingerprint.shape[0] != 1:
                        residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram)
                    
                    if self.scoring=="correlation":
                        score = correlation_score(self.fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(self.fingerprint, residual_spectrogram, self.invcov)
                elif self.filter.name == "MarrasFilter":
                    spectrogram = self.transformation(audio).squeeze()
                    # spectrogram to decibel 
                    spectrogram = 10. * torch.log(spectrogram + 10e-13) 
                    # linearily scaling spectrogram to 0-255 
                    spectrogram = (spectrogram - torch.min(spectrogram)) / (torch.max(spectrogram) - torch.min(spectrogram)) * 255
                    spectrogram = spectrogram.cpu().numpy().astype(np.uint8)
                    residual_spectrogram = torch.tensor(self.filter.forward(spectrogram))
                    min_size = min(self.fingerprint.shape[1], residual_spectrogram.shape[1])
                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram[:, :min_size])
                    fingerprint = self.fingerprint[:, :min_size]
                    
                    if self.scoring=="correlation":
                        fingerprint = self.zero_mean_unit_norm(fingerprint)
                        score = correlation_score(fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual_spectrogram, self.invcov[:min_size, :min_size]) 
                elif self.filter.name == "GriffinLim":
                    #spectrogram = self.transformation(audio).squeeze()
                    residual_spectrogram, residual_spectrogram_db = self.filter.forward(audio)
                    min_size = min(self.fingerprint.shape[1], residual_spectrogram_db.shape[1])
                    residual_spectrogram_db = self.zero_mean_unit_norm(residual_spectrogram_db[:, :min_size])
                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram[:, :min_size])
                    fingerprint = self.fingerprint_db[:, :min_size]

                    if self.scoring=="correlation":
                        fingerprint = self.zero_mean_unit_norm(fingerprint)
                        score = correlation_score(fingerprint, residual_spectrogram_db)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual_spectrogram_db, self.invcov[:min_size])
                        
                elif self.filter.name == 'HighFreq-Spec':
                    residual_spectrogram = filtered_audio.squeeze(0)
                    residual_spectrogram = 10. * torch.log(residual_spectrogram + 10e-13)
                    min_size = min(self.fingerprint.shape[1], residual_spectrogram.shape[1])
                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram[:, :min_size])
                    fingerprint = self.zero_mean_unit_norm(self.fingerprint[:, :min_size])
                    
                    if self.scoring=="correlation":
                        score = correlation_score(fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual_spectrogram, self.invcov)
                        
                elif self.filter.name in ["MovingAverage-Spec", "SFA-Spec"]:
                    spectrogram = self.transformation(audio)
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)  - 10. * torch.log(filtered_audio + 10e-13) 
                    residual_spectrogram = residual_spectrogram.squeeze(0)
                    min_size = min(self.fingerprint.shape[1], residual_spectrogram.shape[1])
                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram[:, :min_size])
                    fingerprint = self.zero_mean_unit_norm(self.fingerprint[:, :min_size])
                    
                    if self.scoring=="correlation":
                        score = correlation_score(fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual_spectrogram, self.invcov)
                elif self.filter.name in ["AllPassBiQuadFilter", 
                                        "HighPassBiQuadFilter", 
                                        "LowPassBiQuadFilter", 
                                        "BandPassBiQuadFilter", 
                                        "BandRejectBiQuadFilter",
                                        "BandBiQuadFilter",
                                        "TrebleBiQuadFilter",
                                        "EqualizerBiQuadFilter",
                                        "LFilterFilter",
                                        "FiltFiltFilter",
                                        "low_pass_filter", 
                                        "high_pass_filter",
                                        "band_pass_filter",
                                        "band_stop_filter",
                                        "Oracle"]:
                    avg_mfcc = self.transformation.forward(audio, original_audio_lengths)
                    filtered_avg_mfcc = self.transformation.forward(filtered_audio, original_audio_lengths)
                    residual = avg_mfcc - filtered_avg_mfcc 
                    if self.filter_trend_correction:
                        residual = residual - self.trend
                    '''
                    if cutoff is not None: 
                        residual = residual[:, :cutoff]
                        print("Use of a cutoff!")
                    if self.fingerprint.shape[1] != cutoff: 
                        fingerprint = self.fingerprint[:, :cutoff]
                    '''
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
                            fingerprint = self.zero_mean_unit_norm(self.fingerprint) # Redundant, cause the fingerprint is already normlaized. 
                            # @Matias it is not redundant because we are cutting off certain features. Cutting off makes them not standardized anymore! 
                    else:
                        fingerprint = self.fingerprint
                    
                    if self.scoring=="correlation":
                        residual = self.zero_mean_unit_norm(residual)
                        score = correlation_score(fingerprint, residual)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(fingerprint, residual, self.invcov)
                
                else:
                    spectrogram = self.transformation(audio)
                    if filtered_audio.shape[2] != spectrogram.shape[2]:
                        if filtered_audio.shape[2] > spectrogram.shape[2]:
                            filtered_audio = filtered_audio[:, :,:spectrogram.shape[2]]
                        else:
                            spectrogram = spectrogram[:, :,:filtered_audio.shape[2]]
                    residual_spectrogram = 10. * torch.log(spectrogram + 10e-13)  - 10. * torch.log(filtered_audio + 10e-13) 
                    # residual_spectrogram = self.transformation(residual)
                    residual_spectrogram = residual_spectrogram.squeeze(0)
                    # residual_spectrogram = 10. * torch.log(residual_spectrogram + 10e-13)
                    residual_spectrogram = torch.mean(residual_spectrogram, dim=1)

                    residual_spectrogram = self.zero_mean_unit_norm(residual_spectrogram)
                    if self.scoring=="correlation":
                        score = correlation_score(self.fingerprint, residual_spectrogram)
                    elif self.scoring=="mahalanobis":
                        score = mahalanobis_score(self.fingerprint, residual_spectrogram, self.invcov)
                        
                # scores.append(score.item())
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

class ClosedWorldFingerprintingWrapper:
    name = "fingerprint"

    def __init__(self, 
                 filter, 
                 num_samples: int = 100, 
                 tolerable_fnr: list = [0.0, 0.001, 0.005, 0.01], 
                 transformation=None, 
                 name=None,
                 fingerprint_paths=[], 
                 trend_paths=[],
                 keep_percentage=[0., 1.],
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
        self.fingerprint_paths = fingerprint_paths
        self.trend_paths = trend_paths 
        self.filter_trend_correction = filter_trend_correction 
        
    
        
 
    def train(self):
        self.fingerprints = {}
        self.trends = {} 
        for path in self.fingerprint_paths:
            folder_name = path.split("/")[-2]
            with open(path, 'rb') as f:
                self.fingerprints[folder_name] = pickle.load(f)
        if self.trend_paths != []:
            for path in self.trend_paths:
                folder_name = path.split("/")[-2]
                with open(path, 'rb') as f:
                    self.trends[folder_name] = pickle.load(f)
        
                    
    def forward(self, wave_tensor: torch.Tensor) -> float:
        corr = []
        for i in tqdm(wave_tensor, desc="Evaluating fingerprint"):
            audio = i[0]
            path = i[2]
            
            
            avg_features = self.transformation.forward(audio)
            filtered_audio = self.filter.forward(audio)
            filtered_avg_features = self.transformation.forward(filtered_audio)
            residual = avg_features - filtered_avg_features 
          

            
            correlations = [] 
            for path in self.fingerprint_paths:
                folder_name = path.split("/")[-2]
                fingerprint = self.fingerprints[folder_name]
                if self.filter_trend_correction:
                    trend = self.trends[folder_name]
                    residual_wo_trend = residual - trend 
                else:
                    residual_wo_trend = residual 
        
                correlation = torch.inner(fingerprint.flatten(), self.zero_mean_unit_norm(residual_wo_trend).flatten())
                correlations.append(correlation.item())
            # one-hot vector with maximum correlation 
            corr.append(correlations.index(max(correlations)))
        return corr
    
    @staticmethod
    def spec_in_db(spec):
        return 10. * torch.log(spec + 10e-13)

    @staticmethod
    def db_in_spec(db):
        return  torch.exp(db / 10.0) - 10e-13 

    @staticmethod
    def zero_mean_unit_norm(array: torch.tensor) -> torch.tensor:
        array = array - torch.mean(array)
        return array / torch.norm(array)

    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pickle.dump(self.fingerprint, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.fingerprint = pickle.load(f)

    def __repr__(self):
        return f"FingerprintingWrapper(num_samples={self.num_samples})"
