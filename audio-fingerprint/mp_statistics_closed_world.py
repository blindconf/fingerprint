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
from src.fingerprinting import SFA_filter, SpectralGating, ClosedWorldFingerprintingWrapper, MovingAverage, MorphologicalFilter, MarrasFilter, LowHighFreq, BaselineFilter, GriffinLimFilter
from src.fingerprinting import AllPassBiQuadFilter, HighPassBiQuadFilter, LowPassBiQuadFilter, BandPassBiQuadFilter, BandRejectBiQuadFilter
from src.fingerprinting import LFilterFilter, BandBiQuadFilter, TrebleBiQuadFilter, EqualizerBiQuadFilter, FiltFiltFilter, WaveformToAvgMFCC, WaveformToAvgMel, WaveformToAvgSpec
from src.utils import plot_difference, save_audio_files, hist_plot, plot_finger_freq
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
                "ljspeech_melgan_train.csv",
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
FILTER_TYPE = 'SFA-Spec'
FETAURES = 'STFT'
# FILTER_TYPE  =  'SG' # 'MarFil' 'MorFil' 'MA' 'SFA' 'SG' 'HighFreq' 'LowFreq'
N_FFT_lst = [32, 128, 512, 2048] # [32, 128, 512, 2048] # 256


THRESHOLD = [5, 10, 50, 250]#, 5, 10, 15]
# MorFil:       [1, 5, 10, 15] 
# MA:           [2, 5, 10, 15]
# LowHighFreq:  [2, 5, 10, 20]
# SFA:          [1, 2, 3, 4]
THRESHOLD = [1, 2, 3, 4] # SFA
SAMPLE_RATE = 22050
# WINDOW_SIZE = 2

TRAINING_SAMPLE = 1000 #10480
TEST_SAMPLE = 100 # 2620

#### NEW! 
NEW_FILTERS = {#"AllPassBiQuadFilter": AllPassBiQuadFilter,
               #"HighPassBiQuadFilter": HighPassBiQuadFilter,
               #"LowPassBiQuadFilter": LowPassBiQuadFilter,
               "BandPassBiQuadFilter": BandPassBiQuadFilter,
               #"BandRejectBiQuadFilter": BandRejectBiQuadFilter,
               #"LFilterFilter": LFilterFilter,
               #"BandBiQuadFilter": BandBiQuadFilter,
               #"TrebleBiQuadFilter": TrebleBiQuadFilter, # TODO How to set the gain hyperparameter? 
               #"EqualizerBiQuadFilter": EqualizerBiQuadFilter, # TODO How to set the gain hyperparameter? 
               #"FiltFiltFilter": FiltFiltFilter
               }

SIMPLE_FILTERS = ["AllPassBiQuadFilter", "HighPassBiQuadFilter", "LowPassBiQuadFilter", "BandPassBiQuadFilter", "BandRejectBiQuadFilter", "BandBiQuadFilter"]

BUTTER_FILTERS = ["LFilterFilter", "FiltFiltFilter"]   

#FILTER_TYPE = 'AllPassBiQuadFilter'
N_FFT_lst = [2048, 1024]
HOP_LENGTH = 128
N_MELS = 512 
N_MFCC = 128 
THRESHOLD = [100, 150, 300, 500, 1000, 2000, 5000, 10000]

butter_order = 4
butter_filter_type = 'lowpass'

filter_trend_correction = True
transformation_type = "Avg_Mel"

N_FFT_lst = [1024, 2048]
THRESHOLD = [300, 500, 150]
cutoff = None 

closed_world = True 



assert transformation_type in ["Avg_MFCC", "Avg_Mel", "Avg_Spec"], "Transformation type not supported! Must be in ['Avg_MFCC', 'Avg_Mel'. 'Avg_Spec']"

def main():
    reference_data = None
    reference_name = None
    
    
    
    #global FILTER_TYPE
    for FILTER_TYPE in NEW_FILTERS.keys():
        if FILTER_TYPE == 'SG':
            if THRESHOLD[0] == 1:
                FILTER_TYPE = 'SG_nonstationary'
                non_stat_flg = True
            else:
                FILTER_TYPE = 'SG_stationary'
                non_stat_flg = False
        data_dir_path = 'avg_freq/' + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE
        plot_path = "plots_newfilters_" + transformation_type + "/" + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE 
        audio_path = "audio_examples/" + LENGTH_AUDIO + "/" + FILTER_TYPE 

            
         
        for N_FFT in N_FFT_lst:
            for thr in THRESHOLD:
                
                # Load all fingerprints 
                all_fingerprints_paths = []
                all_trends_paths = [] 
                counter = 0 
                for path_csv in TRAIN_CSV: 
                    path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
                    name = os.path.splitext(os.path.basename(path))[0]    

                    folder_name = name.replace("_train", "")
                    fingerprint_path = f"{data_dir_path}/{folder_name}/{thr}_{N_FFT}_{N_MELS}_{N_MFCC}_trend={filter_trend_correction}_{transformation_type}_fingerprint.pickle"        
                    all_fingerprints_paths.append(fingerprint_path)
                    
                    trend_path = f"{data_dir_path}/{folder_name}/{thr}_{N_FFT}_{N_MELS}_{N_MFCC}_trend={filter_trend_correction}_{transformation_type}_trend.pickle"
                    all_trends_paths.append(trend_path)
                    
                acc_results = {}
                if FILTER_TYPE in BUTTER_FILTERS:
                    acc_path = f'{FILTER_TYPE}_{butter_order}_{butter_filter_type}_{transformation_type}_acc/closed_world/acc_thr={thr}_nmels={N_MELS}_nmfcc={N_MFCC}_nfft={N_FFT}_hoplen={HOP_LENGTH}_trend={filter_trend_correction}.xlsx'
                else:
                    acc_path = f'{FILTER_TYPE}_{transformation_type}_acc/closed_world/acc_thr={thr}_nmels={N_MELS}_nmfcc={N_MFCC}_nfft={N_FFT}_hoplen={HOP_LENGTH}_trend={filter_trend_correction}.xlsx'
                if os.path.exists(acc_path):
                    pass # TODO usually do not recompute the AUCs
                    #continue 
                
                if FILTER_TYPE in ['SG_nonstationary', 'SG_stationary']:
                    audio_filter = SpectralGating(sample_rate=SAMPLE_RATE, nonstationary=non_stat_flg, transf=Spectrogram(n_fft=N_FFT).to('cuda'))
                elif FILTER_TYPE in ['SFA', 'SFA-Spec']:
                    quadratic_sfa = Pipeline([("Expansion", PolynomialFeatures(degree=4)),
                                                ("SFA", SFA(n_components=thr))])
                    audio_filter = SFA_filter(sample_rate=SAMPLE_RATE, transf=Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'), 
                                    sfa_transformer=quadratic_sfa, component=thr)
                    if FILTER_TYPE == 'SFA-Spec':
                        audio_filter.name = "SFA-Spec"
                    #audio_filter = SFA_filter(sample_rate=SAMPLE_RATE, transf=Spectrogram(n_fft=N_FFT).to('cuda'), 
                    #                sfa_transformer=SFA(n_components=thr), pf=PolynomialFeatures(degree=4), component=thr)
                elif FILTER_TYPE in ['MA', 'MA-Spec']:
                    audio_filter = MovingAverage(sample_rate=SAMPLE_RATE, window_size=thr, transf=Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'))
                    if FILTER_TYPE == 'MA-Spec':
                        audio_filter.name = "MovingAverage-Spec"
                elif FILTER_TYPE == 'MorFil':
                    audio_filter = MorphologicalFilter(sample_rate=SAMPLE_RATE, transf=Spectrogram(n_fft=N_FFT).to('cuda'), 
                                                        n_fft=N_FFT, threshold=thr, amp=10, invers= GriffinLim(n_fft=N_FFT).to('cuda'))
                elif FILTER_TYPE == 'MarFil':
                    audio_filter = MarrasFilter(transf=Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'))
                elif FILTER_TYPE in ['LowFreq', 'HighFreq', 'HighFreq-Spec']:
                    with open(path, newline='') as f:
                        reader = csv.reader(f)
                        audio_sample_path = next(reader)
                    # print(audio_sample_path)
                    signal, sr = torchaudio.load(audio_sample_path[0])
                    spect_fun = Spectrogram(n_fft=N_FFT).to('cuda')
                    spect_signal = spect_fun(signal.to('cuda'))
                    total_bins = spect_signal.shape[1]
                    if FILTER_TYPE == 'LowFreq':
                        low_freq_flg = True
                        cutoff_ptg = thr
                    elif FILTER_TYPE in ['HighFreq', 'HighFreq-Spec']:
                        low_freq_flg = False
                        cutoff_ptg = 100-thr
                    audio_filter = LowHighFreq(sample_rate=SAMPLE_RATE, freq_bins=total_bins, threshold=cutoff_ptg, low_freq_flg=low_freq_flg, 
                                        transf=Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'), 
                                        invers= GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'))
                    if FILTER_TYPE == 'HighFreq-Spec':
                        audio_filter.name = "HighFreq-Spec"
                elif FILTER_TYPE == 'Baseline':
                    audio_filter = BaselineFilter(transf=Spectrogram(n_fft=N_FFT).to('cuda'))
                    
                elif FILTER_TYPE == 'GriffinLim':
                    audio_filter = GriffinLimFilter(to_spec=Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to('cuda'), 
                                                    n_fft=N_FFT, 
                                                    hop_length=HOP_LENGTH)
                elif FILTER_TYPE in SIMPLE_FILTERS: 
                    # Note: thr is here the cutoff_freq or the central_freq
                    audio_filter = NEW_FILTERS[FILTER_TYPE](SAMPLE_RATE, thr)
                    melkwargs = {"n_fft": N_FFT, 
                                "n_mels": N_MELS,
                                "hop_length": HOP_LENGTH}
                elif FILTER_TYPE in BUTTER_FILTERS:
                    audio_filter = NEW_FILTERS[FILTER_TYPE](SAMPLE_RATE, butter_order=butter_order, butter_freq=thr, filter_type=butter_filter_type)
                    melkwargs = {"n_fft": N_FFT, 
                                "n_mels": N_MELS,
                                "hop_length": HOP_LENGTH}
                    
                if transformation_type == 'Avg_MFCC':
                    transformation = WaveformToAvgMFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, melkwargs=melkwargs)
                elif transformation_type == 'Avg_Mel':
                    transformation = WaveformToAvgMel(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                elif transformation_type == 'Avg_Spec':
                    transformation = WaveformToAvgSpec(n_fft=N_FFT, hop_length=HOP_LENGTH)
                    
                wrapper = ClosedWorldFingerprintingWrapper(filter=audio_filter, 
                                                            transformation=transformation, 
                                                            name=name, 
                                                            fingerprint_paths=all_fingerprints_paths,
                                                            trend_paths=all_trends_paths,
                                                            filter_trend_correction=filter_trend_correction)


                folder_name = name.replace("_train", "")
                if FILTER_TYPE in BUTTER_FILTERS:
                    folder_name = f"{folder_name}_{butter_order}_{butter_filter_type}"
                os.makedirs(f"{data_dir_path}/{folder_name}", exist_ok=True)
                os.makedirs(f"{plot_path}/{folder_name}", exist_ok=True)
                # skip_flag = False
                
                fingerprint_path = f"{data_dir_path}/{folder_name}/{thr}_{N_FFT}_{N_MELS}_{N_MFCC}_trend={filter_trend_correction}_{transformation_type}_fingerprint.pickle"
                # trend path is only accessed if filter_trend_correction==True
                trend_path = f"{data_dir_path}/{folder_name}/{thr}_{N_FFT}_{N_MELS}_{N_MFCC}_trend={filter_trend_correction}_{transformation_type}_trend.pickle"

               
                    
                wrapper.train() 
      
                    
                outputs = {}
                outputs_list = []
                counter = 0
                labels = []
                for test_path in TEST_CSV: 
                    test_file = f"{CSV_PATH}/{LENGTH_AUDIO}/{test_path}"
                    name_test = os.path.splitext(os.path.basename(test_file))[0] 
                    print(name_test, thr, N_FFT)
                    audio_test_ds = AudioDataSet(test_file, 
                                        target_sample_rate=SAMPLE_RATE,
                                        train_nrows=TEST_SAMPLE,
                                        device='cuda'
                                        )
                    
                    
                    output = wrapper.forward(audio_test_ds)
                    outputs[name_test] = output
                    outputs_list.append(output)
                    labels.append( [counter]*len(output))
                    counter += 1 

                outputs_list = np.concatenate(outputs_list)
                labels = np.concatenate(labels)
                acc = accuracy_score(labels, outputs_list)
                print(f"Accuracy: {acc}")
                
                conf_matrix = confusion_matrix(labels, outputs_list)
                disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(outputs.keys()))
                disp.plot() 
                disp.ax_.set_title(f"Accuracy = {acc}")
                path_confusion_matrix = f"closed_world_acc/confusion_matrix"
                os.makedirs(path_confusion_matrix, exist_ok=True)
                plt.savefig(f"{path_confusion_matrix}/thr={thr}-nfft={N_FFT}-transformation={transformation_type}-filter-trend-corr={filter_trend_correction}.png")
        


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 python mp_statistics.py /USERSPACE/DATASETS/LJSpeech-1.1/wavs,ljspeech -a 4 -s
