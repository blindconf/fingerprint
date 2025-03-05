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
from src.fingerprinting import OracleFilter, FingerprintingWrapper, OracleMFCCFilter, OracleLFCCFilter
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
# TODO Why is ljspeech_melgan_train excluded? Fix this when running the final version! 
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
FILTER_TYPE = 'Oracle_MFCC_Avg' #'Oracle_MFCC_Avg'
FETAURES = 'STFT'

# TODO: I don't need these here!
THRESHOLD = [5, 10, 50, 250]#, 5, 10, 15]
# MorFil:       [1, 5, 10, 15] 
# MA:           [2, 5, 10, 15]
# LowHighFreq:  [2, 5, 10, 20]
# SFA:          [1, 2, 3, 4]
THRESHOLD = [1, 2, 3, 4] # SFA
THRESHOLD = [5]
SAMPLE_RATE = 22050

# TODO: All Experiments were conducted on smaller sample sizes.
TRAINING_SAMPLE = 10480
TEST_SAMPLE = 2620

KEEP_PERCENTAGE = [0., 1.0] # How many Frequencies to keep. [0., 1.] keeps all (from 0% to 100% percentile)
REWEIGHT = False # Reweight differences pointwise based on the original signal. Did not help!
SPECTROGRAM = "Mel" # Or "" for standard Spectrogram, Only Relevant for FILTER_TYPE in ['Oracle_Spec', 'Oracle']. 

# MFCC parameters: 
# Big study: 
N_FFT_lst = [2048] #[512, 1024, 2048, 4096]  
N_MFFCS = [128] #[128, 256, 512, 1024]
N_MELS = [512] #[128, 256, 512]
HOP_LENGTHS = [128] #[None, 64, 128] 



def main():
    global FILTER_TYPE
    if FILTER_TYPE == 'SG':
        if THRESHOLD[0] == 1:
            FILTER_TYPE = 'SG_nonstationary'
            non_stat_flg = True
        else:
            FILTER_TYPE = 'SG_stationary'
            non_stat_flg = False
    data_dir_path = f'avg_freq{SPECTROGRAM}/' + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE
    plot_path = f"plots{SPECTROGRAM}/" + LENGTH_AUDIO + "/" + FETAURES + "/" + FILTER_TYPE 
    audio_path = "audio_examples/" + LENGTH_AUDIO + "/" + FILTER_TYPE 
    
    
    ##########
    # This is new and specific to MFCC! 
    for N_MFFC in N_MFFCS:
        for N_MEL in N_MELS:
            if N_MFFC > N_MEL:
                continue
            ######
            for N_FFT in N_FFT_lst:
                for hop_length in HOP_LENGTHS:
                    # TODO: Remove this for the final version. Be careful when using different sample sizes!
                    #if os.path.exists(f'{FILTER_TYPE}_aucs/auc_nmels={N_MEL}_nmfcc={N_MFFC}_nfft={N_FFT}_hoplen={hop_length}.xlsx'):
                    #    print(f'{FILTER_TYPE}_aucs/auc_nmels={N_MEL}_nmfcc={N_MFFC}_nfft={N_FFT}_hoplen={hop_length}.xlsx already exists!')
                    #    continue
                    auc_results = {}
                    for path_csv in TRAIN_CSV:
                        path = f"{CSV_PATH}/{LENGTH_AUDIO}/{path_csv}"
                        name = os.path.splitext(os.path.basename(path))[0]    
                        
                        if SPECTROGRAM=="Mel":
                            spec_transform = MelSpectrogram(n_fft=N_FFT, 
                                                            hop_length=hop_length, 
                                                            sample_rate=SAMPLE_RATE,
                                                            n_mels=N_MEL).to("cuda")
                        else:
                            spec_transform = Spectrogram(n_fft=N_FFT, hop_length=hop_length, sample_rate=SAMPLE_RATE).to("cuda")

                        if FILTER_TYPE in ['Oracle_MFCC', 'Oracle_MFCC_Avg']:
                            audio_filter = OracleMFCCFilter(sample_rate=SAMPLE_RATE,
                                                            n_mfcc=N_MFFC,
                                                            melkwargs={
                                                                "n_fft": N_FFT,
                                                                "n_mels": N_MEL,
                                                                "hop_length": hop_length,   
                                                            },
                                                            path_real_dir="/USERSPACE/DATASETS/LJSpeech-1.1/wavs/")
                            if FILTER_TYPE == 'Oracle_MFCC_Avg':
                                audio_filter.name = 'Oracle_MFCC_Avg'
                        elif FILTER_TYPE == 'Oracle_LFCC':
                            audio_filter = OracleLFCCFilter(sample_rate=SAMPLE_RATE,
                                                            n_lfcc=256,
                                                            speckwargs={
                                                                "n_fft": N_FFT,
                                                                #"win_length": ..., 
                                                                #"hop_length": ....
                                                            },
                                                            path_real_dir="/USERSPACE/DATASETS/LJSpeech-1.1/wavs/")
                        else:
                            audio_filter = OracleFilter(transf=spec_transform, 
                                                        path_real_dir="/USERSPACE/DATASETS/LJSpeech-1.1/wavs/")
                        if FILTER_TYPE == 'Oracle_Spec':
                            audio_filter.name = 'Oracle_Spec'
                        wrapper = FingerprintingWrapper(filter=audio_filter, 
                                                        transformation=spec_transform,
                                                        name=name,
                                                        keep_percentage=KEEP_PERCENTAGE,
                                                        reweight = REWEIGHT,)
                        audio_ds = AudioDataSet(path, 
                                                target_sample_rate=SAMPLE_RATE,
                                                train_nrows=TRAINING_SAMPLE,
                                                device='cuda'
                                                )
                        

                        folder_name = name.replace("_train", "")
                        os.makedirs(f"{data_dir_path}/{folder_name}", exist_ok=True)
                        os.makedirs(f"{plot_path}/{folder_name}", exist_ok=True)
                        # TODO: As above. Recompute every time to prevent bug with different sample sizes!
                        if not False: #os.path.isfile(f"{data_dir_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_reweight={REWEIGHT}_fingerprint.pickle"):   
                            print("======================================")
                            print(f"Processing {name}!")
                            print("======================================")
                            wrapper.train(audio_ds)
                            fingerprint_path = f"{data_dir_path}/{folder_name}/nmels={N_MEL}_nmfcc={N_MFFC}_hoplen={hop_length}_nfft={N_FFT}_{KEEP_PERCENTAGE}_fingerprint.pickle"
                            with open(fingerprint_path, 'wb') as f:
                                pickle.dump(wrapper.fingerprint, f)
                            os.makedirs(f"{audio_path}/{folder_name}", exist_ok=True)

                            if FILTER_TYPE in ['LowFreq', 'HighFreq', 'Oracle', 'Oracle_MFCC_Avg']:
                                plot_finger_freq(SAMPLE_RATE, wrapper.fingerprint.cpu(), wrapper.name, f"{plot_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_reweight={REWEIGHT}_{name}.pdf")
                            elif FILTER_TYPE in ['Baseline', 'MarFil', 'GriffinLim', 'HighFreq-Spec', 'MA-Spec', 'SFA-Spec', 'Oracle_Spec', 'Oracle_MFCC', 'Oracle_LFCC']:
                                # Plot the average Spectrogram, i.e., the fingerprint
                                try:
                                    plt.imshow(wrapper.fingerprint_db.T.cpu())
                                except:
                                    plt.imshow(wrapper.fingerprint.T.cpu())
                                plt.savefig(f"{plot_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_reweight={REWEIGHT}_{name}.pdf")
                            else:
                                plot_difference(SAMPLE_RATE, wrapper.spect_filter_avg.cpu(), wrapper.name + '_filtered', wrapper.spectrograms_avg.cpu(), wrapper.name,
                                    f"{plot_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_reweight={REWEIGHT}_{name}.pdf")
                                if not os.path.isdir(f"{audio_path}/{folder_name}") or True:   
                                    save_audio_files(audio_ds, wrapper, f"{audio_path}/{folder_name}", f"{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}")
                            

                        if os.path.isfile(fingerprint_path):
                            with open(fingerprint_path, 'rb') as f:
                                print("fingerprint: ", fingerprint_path)
                                wrapper.fingerprint = pickle.load(f)
                        else:
                            print("Fingerprint missing! : ", fingerprint_path)
                            exit()
                        
                            
                        outputs = {}      
                        for test_path in TEST_CSV: 
                            test_file = f"{CSV_PATH}/{LENGTH_AUDIO}/{test_path}"
                            name_test = os.path.splitext(os.path.basename(test_file))[0] 
                            print(name_test, hop_length, N_FFT, REWEIGHT)
                            audio_test_ds = AudioDataSet(test_file, 
                                                target_sample_rate=SAMPLE_RATE,
                                                train_nrows=TEST_SAMPLE,
                                                device='cuda'
                                                )
                            
                            
                            output = wrapper.forward(audio_test_ds)
                            outputs[name_test] = output

                        auc_results[f"{folder_name}-{N_FFT}"] = []
                        for key_dict in outputs.keys():
                            if name.replace("_train", "") != key_dict.replace("_test", ""):
                                labels = [1] * len(outputs[name.replace("_train", "_test")]) + [0] * len(outputs[key_dict])
                                auc = roc_auc_score(labels, outputs[name.replace("_train", "_test")] + outputs[key_dict])
                                print(f"{auc} {hop_length}_{KEEP_PERCENTAGE}_{N_FFT} reweight={REWEIGHT}_{name} vs {key_dict}")
                                hist_plot(f"{plot_path}/{folder_name}/{hop_length}_{KEEP_PERCENTAGE}_{N_FFT}_reweight={REWEIGHT}_{key_dict}.png", outputs[name.replace("_train", "_test")], 
                                            name, outputs[key_dict], key_dict, auc)
                                
                                key_test_against = f"{N_FFT}_{key_dict}"
                                auc_results[f"{folder_name}-{N_FFT}"].append({'vs_model': key_test_against, 'AUC': auc})

                    # Create Excel file with AUC results
                    os.makedirs(f"{FILTER_TYPE}_aucs", exist_ok=True)
                    with pd.ExcelWriter(f'{FILTER_TYPE}_aucs/auc_nmels={N_MEL}_nmfcc={N_MFFC}_nfft={N_FFT}_hoplen={hop_length}.xlsx') as writer:
                        for method, data in auc_results.items():
                            # Convert the list of dictionaries to a DataFrame
                            df = pd.DataFrame(data)
                            # Write the DataFrame to an Excel sheet
                            df.to_excel(writer, sheet_name=method, index=False)

if __name__ == "__main__":
    main()
