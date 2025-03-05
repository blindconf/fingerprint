import argparse
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio.functional import detect_pitch_frequency, spectral_centroid
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC, GriffinLim
import pickle
import os
from src.audio_dataLoader import AudioDataSet, create_data_loader
from src.fingerprinting import OracleFilter, FingerprintingWrapper, MovingAverage, MorphologicalFilter, MarrasFilter, LowHighFreq, BaselineFilter, GriffinLimFilter
from src.utils import plot_difference, save_audio_files, hist_plot, plot_finger_freq
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import csv
import torchaudio
from sksfa import SFA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd 

# Latex font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

CSV_PATH = "csv_files/"
#"ljspeech_melgan_train.csv",
TRAIN_CSV = [
                "ljspeech_parallel_wavegan_train.csv",
                "ljspeech_multi_band_melgan_train.csv",
                "ljspeech_melgan_large_train.csv",
                "ljspeech_full_band_melgan_train.csv",
                "ljspeech_hifiGAN_train.csv",
                "ljspeech_waveglow_train.csv",
                #"LJSpeech_train.csv"
                ]

TEST_CSV =     ["ljspeech_melgan_test.csv",
                "ljspeech_parallel_wavegan_test.csv",
                "ljspeech_multi_band_melgan_test.csv",
                "ljspeech_melgan_large_test.csv",
                "ljspeech_full_band_melgan_test.csv",
                "ljspeech_hifiGAN_test.csv",
                "ljspeech_waveglow_test.csv",
                #"LJSpeech_test.csv"
                ]

LENGTH_AUDIO = 'full'
FILTER_TYPE = 'Oracle'
FETAURES = 'STFT'
# FILTER_TYPE  =  'SG' # 'MarFil' 'MorFil' 'MA' 'SFA' 'SG' 'HighFreq' 'LowFreq'
N_FFT_lst = [32, 128, 512, 2048] # [32, 128, 512, 2048] # 256

# TODO 
N_FFT_lst = [1024]
#HOP_LENGTHS = [None, 64, 128]
HOP_LENGTHS = [None]
THRESHOLD = [5, 10, 50, 250]#, 5, 10, 15]
# MorFil:       [1, 5, 10, 15] 
# MA:           [2, 5, 10, 15]
# LowHighFreq:  [2, 5, 10, 20]
# SFA:          [1, 2, 3, 4]
THRESHOLD = [1, 2, 3, 4] # SFA
THRESHOLD = [5]
SAMPLE_RATE = 22050
# WINDOW_SIZE = 2

TRAINING_SAMPLE = 1000 #10480
TEST_SAMPLE = 100 #2620

KEEP_PERCENTAGE = [0., 1.]

    
def main():
    
    data_dir_path = 'avg_freq/' + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE
 
    
    
    # Load all fingerprints 
    all_fingerprints = {}
    for path_csv in TRAIN_CSV: 
        for N_FFT in N_FFT_lst:
            for hop_length in HOP_LENGTHS:
                path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
                name = os.path.splitext(os.path.basename(path))[0]    

                folder_name = name.replace("_train", "")
                fingerprint_path = f"{data_dir_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_fingerprint.pickle"

                with open(f"{data_dir_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_fingerprint.pickle", 'rb') as f:
                        print("fingerprint: ", f"{data_dir_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_fingerprint.pickle")
                        all_fingerprints[folder_name] = pickle.load(f).cpu().numpy()
    
    # Compute all correlations:    
    all_correlations = {} 
    for folder_path in all_fingerprints.keys():
        correlations = {}
        for folder_path2 in all_fingerprints.keys():
            correlations[folder_path2] = np.inner(all_fingerprints[folder_path], all_fingerprints[folder_path2])
        all_correlations[folder_path] = correlations
    
    # Convert into matrix form (useful for plotting)
    matrix_correlations = np.array([[all_correlations[row][col] for col in all_fingerprints.keys()] for row in all_fingerprints.keys()])
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix_correlations, cmap='coolwarm')

    # Adding color bar
    plt.colorbar(cax)

    # Setting the ticks and labels
    ax.set_xticks(np.arange(len(all_fingerprints.keys())))
    ax.set_yticks(np.arange(len(all_fingerprints.keys())))
    ax.set_xticklabels([" ".join(name.split("_", 1)[1:]) for name in all_fingerprints.keys()])
    ax.set_yticklabels([" ".join(name.split("_", 1)[1:]) for name in all_fingerprints.keys()])

    # Rotating the x-axis labels for better readability
    plt.xticks(rotation=90)

    # Adding text annotations
    for i in range(len(all_fingerprints.keys())):
        for j in range(len(all_fingerprints.keys())):
            ax.text(j, i, f'{matrix_correlations[i, j]:.2f}', ha='center', va='center', color='black')

    plt.tight_layout()
    # Display the plot
    plt.savefig("correlation_between_fingerprints.png")

if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 python mp_statistics.py /USERSPACE/DATASETS/LJSpeech-1.1/wavs,ljspeech -a 4 -s
