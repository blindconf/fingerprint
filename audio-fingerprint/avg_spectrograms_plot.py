from src.data import read_pickle_spectrogram_data
from src.utils import spec_to_mel_spec
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
from sklearn.metrics import roc_auc_score
# New imports
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import recognizer.tools as tools
from src.audio_dataLoader import AudioDataSet
from scipy.io.wavfile import read as read_wav

from tqdm import tqdm 

def zero_mean_unit_norm(array: torch.tensor) -> torch.tensor:
        array = array - torch.mean(array)
        return array / torch.norm(array)
    
    
CSV_PATH = "csv_files/"
#"ljspeech_melgan_train.csv",
TRAIN_CSV = [
                "ljspeech_melgan_train.csv",
                "ljspeech_parallel_wavegan_train.csv",
                "ljspeech_multi_band_melgan_train.csv",
                "ljspeech_melgan_large_train.csv",
                "ljspeech_full_band_melgan_train.csv",
                "ljspeech_hifiGAN_train.csv",
                "ljspeech_waveglow_train.csv",
                #"LJSpeech_train.csv"
                ]


LENGTH_AUDIO = 'full'


SAMPLE_RATE = 22050
TRAINING_SAMPLE = 1000 #10480
N_FFT = 1024 
HOP_LENGTH = 128

os.makedirs("plots_avg_spectrograms", exist_ok=True)


if __name__=="__main__":

    spectrograms_train = []

    # process training data
    for path_csv in TRAIN_CSV:
        path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
        name = os.path.splitext(os.path.basename(path))[0]    
        
        audio_ds = AudioDataSet(path, 
                                target_sample_rate=SAMPLE_RATE,
                                train_nrows=TRAINING_SAMPLE,
                                device='cuda'
                                )

        transformation = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda')
        energies = [] 
        for signal, _, _ in tqdm(audio_ds, desc="Computing Avg Spectrogram"):
            spectrogram = transformation(signal)
            spectrogram = 10. * torch.log10(spectrogram + 1e-6)
            energy = torch.mean(spectrogram.squeeze(), dim=1) 
            energies.append(energy)
        avg_spec = torch.mean(torch.stack(energies), dim=0)
        avg_spec = zero_mean_unit_norm(avg_spec).cpu().numpy().flatten()
        plt.clf()
        plt.bar(range(len(avg_spec)), avg_spec)
        plt.savefig(f"plots_avg_spectrograms/{name}.png")
        
        
    
    