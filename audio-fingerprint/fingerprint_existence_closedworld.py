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
from src.fingerprinting import OracleFilter, ClosedWorldFingerprintingWrapper, MovingAverage, MorphologicalFilter, MarrasFilter, LowHighFreq, BaselineFilter, GriffinLimFilter
from src.utils import plot_difference, save_audio_files, hist_plot, plot_finger_freq
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score 
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

TEST_CSV =     ["ljspeech_parallel_wavegan_test.csv",
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

KEEP_PERCENTAGE = [0., 1.0]
def main():
    reference_data = None
    reference_name = None

    global FILTER_TYPE
    if FILTER_TYPE == 'SG':
        if THRESHOLD[0] == 1:
            FILTER_TYPE = 'SG_nonstationary'
            non_stat_flg = True
        else:
            FILTER_TYPE = 'SG_stationary'
            non_stat_flg = False
    data_dir_path = 'avg_freq/' + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE
    plot_path = "plots/" + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE 
    audio_path = "audio_examples/" + LENGTH_AUDIO + "/" + FILTER_TYPE 

    acc_results = {} 
    
    
    # Load all fingerprints 
    all_fingerprints_paths = []
    counter = 0 
    for path_csv in TRAIN_CSV: 
        for N_FFT in N_FFT_lst:
            for hop_length in HOP_LENGTHS:
                path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
                name = os.path.splitext(os.path.basename(path))[0]    

                folder_name = name.replace("_train", "")
                fingerprint_path = f"{data_dir_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_fingerprint.pickle"
                all_fingerprints_paths.append(fingerprint_path)
                        
    for path_csv in TRAIN_CSV: 
        for N_FFT in N_FFT_lst:
            for hop_length in HOP_LENGTHS:
                path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
                name = os.path.splitext(os.path.basename(path))[0]    

                
                audio_filter = OracleFilter(transf=Spectrogram(n_fft=N_FFT, hop_length=hop_length).to('cuda'), 
                                            path_real_dir="/USERSPACE/DATASETS/LJSpeech-1.1/wavs/")
                

                wrapper = ClosedWorldFingerprintingWrapper(filter=audio_filter, 
                                                transformation=Spectrogram(n_fft=N_FFT, hop_length=hop_length).to('cuda'),
                                                name=name,
                                                keep_percentage=KEEP_PERCENTAGE,
                                                fingerprint_paths=all_fingerprints_paths)
                wrapper.train() 

                    
                outputs = {}      
                for test_path in TEST_CSV: 
                    test_file = f"{CSV_PATH}/{LENGTH_AUDIO}/{test_path}"
                    name_test = os.path.splitext(os.path.basename(test_file))[0] 
                    print(name_test, hop_length, N_FFT)
                    audio_test_ds = AudioDataSet(test_file, 
                                        target_sample_rate=SAMPLE_RATE,
                                        train_nrows=TEST_SAMPLE,
                                        device='cuda'
                                        )
                    
                    
                    output = wrapper.forward(audio_test_ds)
                    outputs[name_test] = output
                
                
                accuracies = []
                for key_dict in outputs.keys():
                    if name.replace("_train", "") != key_dict.replace("_test", ""):
                        labels = [1] * len(outputs[name.replace("_train", "_test")]) + [0] * len(outputs[key_dict])
                        predictions = outputs[name.replace("_train", "_test")] + outputs[key_dict]
                        # predictions = 0 if not equal to counter else 1 
                        predictions = np.array(predictions) == counter 
                        acc = accuracy_score(labels, predictions)   
              
                        print(f"{acc} {hop_length}_{KEEP_PERCENTAGE}_{N_FFT} {name} vs {key_dict}")
                        
                        key_test_against = f"{N_FFT}_{key_dict}"
                        accuracies.append({'vs_model': key_test_against, 'ACC': acc})
                        
                acc_results[f'{name.replace("_train", "")}-{N_FFT}'] = accuracies 
        counter += 1
    print(acc_results)
    


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 python mp_statistics.py /USERSPACE/DATASETS/LJSpeech-1.1/wavs,ljspeech -a 4 -s
