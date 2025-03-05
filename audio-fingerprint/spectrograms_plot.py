from src.data import read_pickle_spectrogram_data
from src.utils import spec_to_mel_spec
from src.fingerprinting import HighPassFilter, FingerprintingWrapper
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
import recognizer.feature_extraction as fe
import csv
from scipy.io.wavfile import read as read_wav
import librosa


n_train = 2000
my_gan = "waveglow"
# Generate audio features during training
paths = {"melgan": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_melgan.csv",
         "hifiGAN": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_hifiGAN.csv",
         "waveglow": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_waveglow.csv"}

os.makedirs("spectrogram_plots", exist_ok=True)
os.makedirs("plots", exist_ok=True)

if __name__=="__main__":
    print(torch.__version__)
    print(torchaudio.__version__)
    path_train = paths[my_gan]

    spectrograms_train = []

    counter_plots = 0
    min_num_time_bins = 100000
    max_num_time_bins = 0
    # process training data
    with open(paths[my_gan], newline='') as f:
        reader = csv.reader(f)
        while counter_plots < 5:
            row1 = next(reader) 
            # Load audio
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(row1[0])
            # Normalized audio data between -1 and 1
            max_amp = torch.max(torch.abs(SPEECH_WAVEFORM))
            fig, axs = plt.subplots(1, 1)
            tools.plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform")
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/original_wave_{counter_plots}.png")
            plt.close()
            # n_ftts defines resolution of the frequency axis for a given sampling rate
            n_ffts = [32, 128, 512, 2048]
            hop_length = 64
            specs = []
            for n_fft in n_ffts:
                spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
                spec = spectrogram(SPEECH_WAVEFORM)
                specs.append(spec)
            fig, axs = plt.subplots(len(specs), 1, sharex=True)
            for i, (spec, n_fft) in enumerate(zip(specs, n_ffts)):
                tools.plot_spectrogram(spec[0], ylabel=f"n_fft={n_fft}", ax=axs[i])
                axs[i].set_xlabel(None)
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/spectrogram_nfft_{counter_plots}.png")
            plt.close()

            n_fft = 1024
            win_length = None
            hop_length = 512
            n_mels = 128

            mel_spectrogram = T.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                n_mels=n_mels,
                mel_scale="htk",
            )

            melspec = mel_spectrogram(SPEECH_WAVEFORM)
            fig, axs = plt.subplots(1, 1)
            tools.plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/MelSpectrogram_{counter_plots}.png")
            plt.close()
            
            n_fft = 2048
            win_length = None
            hop_length = 512
            n_mels = 256
            n_mfcc = 256

            mfcc_transform = T.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )

            mfcc = mfcc_transform(SPEECH_WAVEFORM)
            fig, axs = plt.subplots(1, 1)
            tools.plot_spectrogram(mfcc[0], title="MFCC")
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/MFCC_{counter_plots}.png")
            plt.close()
            
            n_fft = 2048
            win_length = None
            hop_length = 512
            n_lfcc = 256

            lfcc_transform = T.LFCC(
                sample_rate=SAMPLE_RATE,
                n_lfcc=n_lfcc,
                speckwargs={
                    "n_fft": n_fft,
                    "win_length": win_length,
                    "hop_length": hop_length,
                },
            )

            lfcc = lfcc_transform(SPEECH_WAVEFORM)
            fig, axs = plt.subplots(1, 1)
            tools.plot_spectrogram(lfcc[0], title="LFCC")
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/LFCC_{counter_plots}.png")
            plt.close()

            n_fft = 2048
            win_length = None
            hop_length = 512
            n_mels = 256
            n_mfcc = 256

            mfcc_transform = T.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )
            # delta and deltadeltas
            deltas_function = T.ComputeDeltas()
            mfcc = mfcc_transform(SPEECH_WAVEFORM)
            # add delta and delta deltas
            deltas = deltas_function(mfcc)
            deltadeltas = deltas_function(deltas)
            mfcc = torch.cat((mfcc, deltas, deltadeltas), 1)

            fig, axs = plt.subplots(1, 1)
            tools.plot_spectrogram(mfcc[0], title="MFCC+Delta+Delta")
            # _, ax = plt.subplots(1, 1)
            # ax.set_title("MFCC+Delta+Delta")
            # ax.set_ylabel("freq_bin")
            # print(torch.min(mfcc[0]))
            # im = ax.imshow(np.log(mfcc[0]+270.2067), origin="lower", aspect="auto", interpolation="nearest")    
            # fig.colorbar(im, ax=ax)     
            fig.tight_layout()
            plt.savefig(f"spectrogram_plots/MFCC_DELTA_DELTA_{counter_plots}.png")
            plt.close()

            counter_plots += 1
