"""
Spoofed Speech via vocoder fingerprint script.
A single-model attribution approach to recognize the source (algorithm) of deepfake utterances.
(https://arxiv.org/abs/...) # Pending arxiv submission
"""
import argparse
import librosa
import numpy as np
import torch
import pickle
import os
from src.audio_dataLoader import AudioDataSet, create_data_loader, collate_fn
from src.fingerprinting import FingerprintingWrapper, WaveformToAvgSpec, WaveformToAvgMel, WaveformToAvgMFCC # LFilterFilter, BandBiQuadFilter, TrebleBiQuadFilter, EqualizerBiQuadFilter, FiltFiltFilter, , , 
from src.fingerprinting import EncodecFilter 
from src.utils import get_auc_path, get_caching_paths, plot_finger_freq, hist_plot # plot_difference, save_audio_files, , , , 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import csv
import torchaudio
import pandas as pd 
from encodec import EncodecModel
from src.filters import OracleFilter, filter_fn
import math
import random


TRAIN_CSV_DICT = {"jsut": [
                "JSUT_multi_band_melgan_train.csv",
                "JSUT_parallel_wavegan_train.csv",
                ],
                "ljspeech": ["ljspeech_melgan_train.csv",        
                            # "ljspeech_parallel_wavegan_train.csv",
                            #"ljspeech_multi_band_melgan_train.csv",
                            #"ljspeech_melgan_large_train.csv",
                            #"ljspeech_full_band_melgan_train.csv",
                            #"ljspeech_hifiGAN_train.csv",
                            "ljspeech_waveglow_train.csv",
                            #"ljspeech_avocodo_train.csv",
                            #"ljspeech_bigvgan_train.csv",
                            #"ljspeech_lbigvgan_train.csv",
                            #"ljspeech_fast_diff_tacotron_train.csv",
                            #"ljspeech_pro_diff_train.csv",
                            ]
                            }

TEST_CSV_DICT = {"jsut": ["JSUT_multi_band_melgan_test.csv",
                "JSUT_parallel_wavegan_test.csv",
                "JSUT_test.csv" # Original data has a fs of 48000 -> resample to 24000
                ], 
                "ljspeech": ["ljspeech_melgan_test.csv",
                # "ljspeech_parallel_wavegan_test.csv",
                # "ljspeech_multi_band_melgan_test.csv",
                # "ljspeech_melgan_large_test.csv",
                # "ljspeech_full_band_melgan_test.csv",
                # "ljspeech_hifiGAN_test.csv",
                "ljspeech_waveglow_test.csv",
                # "ljspeech_avocodo_test.csv",
                # "ljspeech_bigvgan_test.csv",
                # "ljspeech_lbigvgan_test.csv",                
                # "ljspeech_fast_diff_tacotron_test.csv",
                # "ljspeech_pro_diff_test.csv",
                # "LJSpeech_test.csv"
                ]
                }

# Iterate Experiments through all filters in NEW_FILTERS
FILTERS = {"low_pass_filter": filter_fn,
            "EncodecFilter": EncodecFilter,
            }

# I separate features because they require a different treatment (ButterFilters have more parameters)
SIMPLE_FILTERS_2 = ["low_pass_filter", "high_pass_filter", "band_pass_filter", "band_stop_filter"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["ljspeech", "jsut"], default="ljspeech")
    
    # Filter parameters 
    parser.add_argument("--filter-type", choices=FILTERS.keys(), default="EncodecFilter", help="Type of filter to apply to the audio signal.")
    parser.add_argument("--filter-param", type=float, default=24, help="Parameter of the filter.")
    parser.add_argument("--scorefunction", choices=["mahalanobis", "correlation"], default="correlation", help="Type of scoring function to use.")
   
    # Paths 
    parser.add_argument("--csv-dir", type=str, default="csv_files") 
    parser.add_argument("--num-train", type=int, default=None, help="Number of training samples. If not specified, take full train set of the corpus.")   
    parser.add_argument("--num-test", type=int, default=None, help="Number of test samples. If not specified, take full test set of the corpus.")
    parser.add_argument("--transformation", choices=["Avg_MFCC", "Avg_Mel", "Avg_Spec"], 
                        default="Avg_Spec",
                        help="Type of transformation to apply to the audio signal.")
    parser.add_argument("--nfft", type=int, default=2048, help="Number of FFT points for creating the Spectrograms.")
    parser.add_argument("--hop-len", type=int, default=128, help="Hop length for creating the Spectrograms.")
    parser.add_argument("--nmels", type=int, default=128, help="Number of Mel bands for creating the Mel Spectrograms.")
    parser.add_argument("--nmfcc", type=int, default=128, help="Number of MFCC coefficients for creating the MFCCs.")

    # Perturbations
    parser.add_argument("--perturbation", choices=["None", "noise", "encodec"], default="None", help="Type of perturbation to apply to the audio signal.")
    parser.add_argument("--encodec-qr", choices=["1_5", "3", "6", "12", "24"], default="1_5", help="Quantization rate for the Encodec perturbation.")
    parser.add_argument("--snr", choices=["None", "0", "5", "10", "15", "20", "25", "30", "35", "40"], default="None", help="Level of additive noise based on SNR.")

    # Details
    parser.add_argument("--cutoff", type=int, default=None, help="Cutoff first frequency bins for fingerprinting.")
    # parser.add_argument("--length-audio", choices=["full", "short"], default="full")
    parser.add_argument("--trend-correction", action="store_true", help="Correct the filter trend.")
    parser.add_argument("--batchsize", type=int, default=1, help="Adjust batch size as needed.")
    parser.add_argument("--batchsamples", type=int, default=240000, help="Each audio signal is padded or trimmed to the same length.")
    parser.add_argument("--encodec-samplewise", action="store_true", help="Encodec reencoding is applied samplewise. Strangely, the output is different depending on batch-wise or sample-wise computations.")
    # Parse the arguments
    args = parser.parse_args()
     # Validate or restrict filter-param based on filter-type
    if args.filter_type == "EncodecFilter" and  args.filter_param not in [1.5, 3, 6, 12, 24]:
        parser.error("For EncodecFilter, --filter-param must be one of [1.5, 3, 6, 12, 24].")
    elif args.filter_type in ["low_pass_filter", "high_pass_filter"] and args.filter_param not in [-1, -500, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.5]:
        parser.error("For low_pass_filter and high_pass_filter, --filter-param must be of [-1, -500, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.5].")
    elif args.filter_type in ["band_stop_filter", "band_pass_filter"] and args.filter_param not in [1_10_5, 500_10_5, 1-10, 2-9, 3-8, 4-7, 5-6]:
        parser.error("For low_pass_filter and high_pass_filter, --filter-param must be of [1_10_5, 500_10_5, 1-10, 2-9, 3-8, 4-7, 5-6]}.")
    
    args.sample_rate = 24000 if args.corpus == "jsut" else 22050
    args.train_csv = TRAIN_CSV_DICT[args.corpus]
    args.test_csv = TRAIN_CSV_DICT[args.corpus]
    if args.num_train is None:
        args.num_train = 4000 if args.corpus == "jsut" else 10480
    if args.num_test is None:
        args.num_test = 1000 if args.corpus == "jsut" else 2620
    if args.scorefunction == "mahalanobis" and args.num_train*2 < args.nfft:
        sys.exit(f"The sample size is too small. Consider reducing the nfft value ({args.nfft}) or increasing the number of training samples ({args.num_train}).")
    return args

def main(args):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_CSV = TRAIN_CSV_DICT[args.corpus]
    TEST_CSV = TEST_CSV_DICT[args.corpus]

    data_dir_path = 'avg_freq/' + args.filter_type

    if args.filter_type == "EncodecFilter":
        data_dir_path += f"-compute_samplewise={args.encodec_samplewise}"
        
    auc_results = {}
    auc_path = get_auc_path(args) 
    os.makedirs(os.path.dirname(auc_path), exist_ok=True)
    # Set a global seed for reproducibility
    # SEED = 42
    # torch.manual_seed(SEED)
    # Dictionary to store all FingerprintingWrapper instances keyed by path_csv name
    wrappers_dict = {}

    for path_csv in TRAIN_CSV:
        path = f"{args.csv_dir}/{path_csv}"
        name = os.path.splitext(os.path.basename(path))[0]    
                
        if args.filter_type in SIMPLE_FILTERS_2: 
            x = []
            if isinstance(args.filter_param, float): 
                args.filter_param = int(args.filter_param) if args.filter_param.is_integer() else args.filter_param # Convert to int and then to string
            file_in = open(f"filter_coefs/{args.filter_type}/{args.filter_param}khz.txt", 'r')
            for y in file_in.read().split('\n'):
                x.append(float(y))
            coef = torch.tensor(x)
            audio_filter = FILTERS[args.filter_type](1, coef, args.filter_type)
        elif args.filter_type == 'EncodecFilter':
            encodec_model = EncodecModel.encodec_model_24khz() # ToDo @Matias also for JSUT? - Matias: Yes!
            bandwidth = args.filter_param
            audio_filter = EncodecFilter(encodec_model, bandwidth, computations_samplewise=args.encodec_samplewise, device="cuda")
        if args.transformation == 'Avg_MFCC':
            melkwargs = {"n_fft": args.nfft, 
                        "n_mels": args.nmels,
                        "hop_length": args.hop_len}
            transformation = WaveformToAvgMFCC(sample_rate=args.sample_rate, n_mfcc=args.nmfcc, melkwargs=melkwargs)
        elif args.transformation == 'Avg_Mel':
            transformation = WaveformToAvgMel(sample_rate=args.sample_rate, n_mels=args.nmels, n_fft=args.nfft, hop_length=args.hop_len)
        elif args.transformation == 'Avg_Spec':
            transformation = WaveformToAvgSpec(n_fft=args.nfft, hop_length=args.hop_len)
            
        wrapper = FingerprintingWrapper(filter=audio_filter, 
                                        transformation=transformation, 
                                        name=name, 
                                        filter_trend_correction=args.trend_correction, 
                                        scoring=args.scorefunction)

        folder_name = name.replace("_train", "")
        caching_paths = get_caching_paths(cache_dir=data_dir_path, method_name=folder_name, args=args)
        fingerprint_path = caching_paths['fingerprint']
        invcov_path = caching_paths['invcov']
        trend_path = caching_paths['trend']

        if os.path.isfile(fingerprint_path):
            with open(fingerprint_path, 'rb') as f:
                wrapper.fingerprint = pickle.load(f)
        if args.trend_correction:
            with open(trend_path, 'rb') as f:
                wrapper.trend = pickle.load(f)
        if args.scorefunction == 'mahalanobis':
            with open(invcov_path, 'rb') as f:
                wrapper.invcov = pickle.load(f)
        # Add the wrapper to the dictionary with the path_csv name as the key
        wrappers_dict[name] = wrapper
    
    outputs = {}
    labels_all = []
    # Determine how many samples to take from each fake model (excluding real speech)
    equal_k = math.ceil(args.num_test / (len(TEST_CSV) - 1))
    # Process authentic speech (LJSpeech_test.csv)
    real_test_file = f"{args.csv_dir}/LJSpeech_test.csv"
    real_test_name = "LJSpeech_test"

    audio_test_ds = AudioDataSet(real_test_file, 
                                 target_sample_rate=args.sample_rate,
                                 train_nrows=args.num_test,
                                 snr=args.snr,
                                 device=device
                                 )
    audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn)
    real_output = {}
    for wrapper in wrappers_dict.values():
        real_output[wrapper.name] = wrapper.forward(audio_test_dataloader, cutoff=args.cutoff)
        #break
    #print(real_output[wrapper.name])
    # Store the maximum scores for real speech
    real_scores = []
    for sample_scores in zip(*real_output.values()):
        real_scores.append(max(sample_scores).cpu())  # Take the maximum score across all wrappers
    outputs[real_test_name] = real_scores
    labels_all.extend([1] * len(real_scores))  # Label '1' for real speech
    # Process fake speech (other vocoders)
    for test_path in TEST_CSV:
        if "LJSpeech_test.csv" in test_path:
            print("Entro: ", test_path)
            continue  # Skip the real speech file as we've already processed it

        test_file = f"{args.csv_dir}/{test_path}"
        name_test = os.path.splitext(os.path.basename(test_file))[0]
        
        # Sample fake audio (args.num_test / (len(TEST_CSV) - 1)) from each vocoder model
        # New random seed to ensure shuffling every time the DataLoader is called
        audio_test_ds = AudioDataSet(test_file, 
                                     target_sample_rate=args.sample_rate,
                                     train_nrows=equal_k,
                                     snr=args.snr,
                                     device=device
                                     )
        audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=True, collate_fn=collate_fn)

        fake_output = {}
        for wrapper in wrappers_dict.values():
            fake_output[wrapper.name] = wrapper.forward(audio_test_dataloader, cutoff=args.cutoff)
            #print(fake_output[wrapper.name])
        # Store the maximum scores for fake speech
        fake_scores = []
        for sample_scores in zip(*fake_output.values()):
            fake_scores.append(max(sample_scores).cpu())  # Take the maximum score across all wrappers

        outputs[name_test] = fake_scores
        labels_all.extend([0] * len(fake_scores))  # Label '0' for fake speech

    # Flatten all the outputs and compute AUROC
    all_scores = []
    for key in outputs:
        print(key, len(outputs[key]))
        all_scores.extend(outputs[key])
    auc_all = roc_auc_score(labels_all, all_scores)
    print(f"AUROC: {1-auc_all} {args.filter_param}_{args.nfft} - Comparing real vs fake speech")

if __name__ == "__main__":
    args = parse_args()
    main(args)
