"""
Spoofed Speech via vocoder fingerprint script.
A single-model attribution approach to recognize the source (algorithm) of deepfake utterances.
(https://arxiv.org/abs/2411.14013) # Arxiv submission
"""
import argparse
import torch
import pickle
import os
from src.audio_dataLoader import AudioDataSet, collate_fn # create_data_loader
from src.fingerprinting import FingerprintingWrapper, WaveformToAvgSpec, WaveformToAvgMel, WaveformToAvgMFCC # LFilterFilter, BandBiQuadFilter, TrebleBiQuadFilter, EqualizerBiQuadFilter, FiltFiltFilter, , , 
from src.filters import OracleFilter, filter_fn
from src.fingerprinting import EncodecFilter 
from src.utils import get_auc_path, get_caching_paths, plot_finger_freq, hist_plot, generate_csv_files, generate_csv_files_noise, get_csv_dict # plot_difference, save_audio_files, , , , 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd 
from encodec import EncodecModel
import math
# import librosa
# import numpy as np
# from torchaudio.functional import detect_pitch_frequency, spectral_centroid
# from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC, GriffinLim
# from src.fingerprinting import SFA_filter, SpectralGating, MovingAverage, MorphologicalFilter, MarrasFilter, LowHighFreq, BaselineFilter, GriffinLimFilter
# from src.fingerprinting import AllPassBiQuadFilter, HighPassBiQuadFilter, LowPassBiQuadFilter, BandPassBiQuadFilter, BandRejectBiQuadFilter
# import csv
# import torchaudio
# from sklearn.pipeline import Pipeline
# import sys

FAKE_DATA_DICT = {"jsut": [
                "jsut_hnsf",
                "jsut_multi_band_melgan",
                "jsut_parallel_wavegan",
                ],
                "ljspeech": ["ljspeech_hnsf",
                            "ljspeech_melgan",
                            "ljspeech_parallel_wavegan",
                            "ljspeech_multi_band_melgan",
                            "ljspeech_melgan_large",
                            "ljspeech_full_band_melgan",
                            "ljspeech_hifiGAN",
                            "ljspeech_avocodo",
                            "ljspeech_bigvgan",
                            "ljspeech_lbigvgan",
                            "ljspeech_waveglow",
                            "ljspeech_fast_diff_tacotron",
                            "ljspeech_pro_diff"
                            ]
                            }

vocoders_paper = ["ljspeech_melgan_train.csv", 'ljspeech_melgan_test.csv',
                    "ljspeech_full_band_melgan_train.csv", "ljspeech_full_band_melgan_test.csv",
                    "ljspeech_lbigvgan_train.csv", "ljspeech_lbigvgan_test.csv"]

'''
TRAIN_CSV_DICT = {"jsut": [
                "JSUT_multi_band_melgan_train.csv",
                "JSUT_parallel_wavegan_train.csv",
                ],
                "ljspeech": ["ljspeech_melgan_train.csv",
                            "ljspeech_parallel_wavegan_train.csv",
                            "ljspeech_multi_band_melgan_train.csv",
                            "ljspeech_melgan_large_train.csv",
                            "ljspeech_full_band_melgan_train.csv",
                            "ljspeech_hifiGAN_train.csv",
                            "ljspeech_waveglow_train.csv",
                            "ljspeech_avocodo_train.csv",
                            "ljspeech_bigvgan_train.csv",
                            "ljspeech_lbigvgan_train.csv",
                            "ljspeech_fast_diff_tacotron_train.csv",
                            "ljspeech_pro_diff_train.csv"
                            ]
                            }
# "ljspeech_fast_diff_train.csv"

TEST_CSV_DICT = {"jsut": ["JSUT_multi_band_melgan_test.csv",
                "JSUT_parallel_wavegan_test.csv",
                "JSUT_test.csv" # Original data has a fs of 48000 -> resample to 24000
                ], 
                "ljspeech": ["ljspeech_melgan_test.csv",
                "ljspeech_parallel_wavegan_test.csv",
                "ljspeech_multi_band_melgan_test.csv",
                "ljspeech_melgan_large_test.csv",
                "ljspeech_full_band_melgan_test.csv",
                "ljspeech_hifiGAN_test.csv",
                "ljspeech_waveglow_test.csv",
                "ljspeech_avocodo_test.csv",
                "ljspeech_bigvgan_test.csv",
                "ljspeech_lbigvgan_test.csv",                
                "ljspeech_fast_diff_tacotron_test.csv",
                "ljspeech_pro_diff_test.csv",
                "LJSpeech_test.csv"
                ]
                }
'''
# "ljspeech_fast_diff_test.csv",

# Iterate Experiments through all filters in NEW_FILTERS
FILTERS = {# "AllPassBiQuadFilter": AllPassBiQuadFilter,
           # "HighPassBiQuadFilter": HighPassBiQuadFilter,
           # "LowPassBiQuadFilter": LowPassBiQuadFilter,
           # "band_stop_filter": filter_fn,
           # "band_pass_filter": filter_fn,
           # "high_pass_filter": filter_fn,
            "low_pass_filter": filter_fn,
           #  "BandPassBiQuadFilter": BandPassBiQuadFilter, # BandPass has led to the best results!
           # "BandRejectBiQuadFilter": BandRejectBiQuadFilter,
           # "LFilterFilter": LFilterFilter,
           # "BandBiQuadFilter": BandBiQuadFilter,
           # "TrebleBiQuadFilter": TrebleBiQuadFilter, # TODO How to set the gain hyperparameter? 
           # "EqualizerBiQuadFilter": EqualizerBiQuadFilter, # TODO How to set the gain hyperparameter? 
           # "FiltFiltFilter": FiltFiltFilter,
            "EncodecFilter": EncodecFilter,
            "Oracle": OracleFilter
            }

# I separate features because they require a different treatment (ButterFilters have more parameters)
SIMPLE_FILTERS = ["AllPassBiQuadFilter", "HighPassBiQuadFilter", "LowPassBiQuadFilter", "BandPassBiQuadFilter", "BandRejectBiQuadFilter", "BandBiQuadFilter"]
SIMPLE_FILTERS_2 = ["low_pass_filter", "high_pass_filter", "band_pass_filter", "band_stop_filter"]
BUTTER_FILTERS = ["LFilterFilter", "FiltFiltFilter"]   

# Only relevant for Butterworth filters
butter_order = 4
butter_filter_type = 'lowpass'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["ljspeech", "jsut"], default="ljspeech")
    parser.add_argument("--real-data-path", help="Directory of real audio")
    parser.add_argument("--fake-data-path", help="Directory of fake audio")
    parser.add_argument("--vocoders", choices=["all", "paper"], default="paper")
    
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
    parser.add_argument("--seed", type=int, default=40, help="Default seed 40.")

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
    args.deterministic = 1 if args.filter_type == "Oracle" else None
    args.shuffle = False if args.filter_type == "Oracle" else True
        
    args.sample_rate = 24000 if args.corpus == "jsut" else 22050
    # args.train_csv = TRAIN_CSV_DICT[args.corpus]
    # args.test_csv = TEST_CSV_DICT[args.corpus]
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
    """
    if args.filter_type == 'SG':
        if args.filter_param == 1:
            args.filter_type = 'SG_nonstationary'
            non_stat_flg = True
        else:
            args.filter_type = 'SG_stationary'
            non_stat_flg = False
    """   
    data_dir_path = f"fingerprints/{args.corpus}/{args.seed}/{args.filter_type}"
    plot_path = f"plots/{args.corpus}/{args.seed}/{args.transformation}{args.scorefunction}/{args.filter_type}"
    audio_path = f"audio_examples/{args.corpus}/{args.seed}/{args.filter_type}"
    if args.filter_type == "EncodecFilter":
        data_dir_path += f"-compute_samplewise={args.encodec_samplewise}"
        plot_path += f"-compute_samplewise={args.encodec_samplewise}"
        audio_path += f"-compute_samplewise={args.encodec_samplewise}"
        
    print("======================================")
    print(f"Fingerprinting method: {args.scorefunction}!")
    print("======================================")

    auc_results = {}
    snr_name = os.path.basename(os.path.normpath(args.fake_data_path))
    last_three_chars = snr_name[-3:]
    auc_path = get_auc_path(args, last_three_chars) 
    # print(auc_path)
    os.makedirs(os.path.dirname(auc_path), exist_ok=True)
    """
    if os.path.exists(auc_path):
        pass # TODO usually do not recompute the AUCs
        #continue 
    """
    # Set a global seed for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    
    if args.perturbation == "noise":
        dir_data_files = f"{args.csv_dir}/{args.corpus}/{SEED}/{args.perturbation}/{last_three_chars}"
        if not os.path.exists(dir_data_files):
            os.makedirs(dir_data_files, exist_ok=True)
            generate_csv_files_noise(
                dir_data_files, 
                args.real_data_path,
                args.fake_data_path,
                FAKE_DATA_DICT[args.corpus], 
                args.seed)
    else:
        dir_data_files = f"{args.csv_dir}/{args.corpus}/{SEED}"
        if not os.path.exists(dir_data_files):
            os.makedirs(dir_data_files, exist_ok=True)
            generate_csv_files(
                    dir_data_files, 
                    args.real_data_path,
                    args.fake_data_path,
                    FAKE_DATA_DICT[args.corpus], 
                    args.seed)
    # print(Asfasfd)
    # print(dir_data_files)

    TRAIN_CSV, TEST_CSV = get_csv_dict(dir_data_files, args.perturbation)
    # print(TRAIN_CSV, TEST_CSV)
    # TRAIN_CSV = TRAIN_CSV_DICT[args.corpus]
    # TEST_CSV = TEST_CSV_DICT[args.corpus]
    if args.vocoders == "paper":
        # Remove files that exactly match any vocoder in vocoders_paper
        TRAIN_CSV = [file for file in TRAIN_CSV if file not in vocoders_paper]
        TEST_CSV = [file for file in TEST_CSV if file not in vocoders_paper]

    for path_csv in TRAIN_CSV:
        path = f"{args.csv_dir}/{args.corpus}/{SEED}/train/{path_csv}"
        name = os.path.splitext(os.path.basename(path))[0]    
        if args.corpus == 'ljspeech' and name == "ljspeech_hnsf_train":
            args.num_train = 10479
        elif args.corpus == 'ljspeech':
            args.num_train = 10480
        """
        if args.filter_type in ['SG_nonstationary', 'SG_stationary']:
            audio_filter = SpectralGating(sample_rate=args.sample_rate, nonstationary=non_stat_flg, transf=Spectrogram(n_fft=args.nfft).to('cuda'))
        elif args.filter_type in ['SFA', 'SFA-Spec']:
            quadratic_sfa = Pipeline([("Expansion", PolynomialFeatures(degree=4)),
                                        ("SFA", SFA(n_components=args.filter_param))])
            audio_filter = SFA_filter(sample_rate=args.sample_rate, transf=Spectrogram(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'), 
                            sfa_transformer=quadratic_sfa, component=args.filter_param)
            if args.filter_type == 'SFA-Spec':
                audio_filter.name = "SFA-Spec"
            #audio_filter = SFA_filter(sample_rate=args.sample_rate, transf=Spectrogram(n_fft=args.nfft).to('cuda'), 
            #                sfa_transformer=SFA(n_components=thr), pf=PolynomialFeatures(degree=4), component=thr)
        elif args.filter_type in ['MA', 'MA-Spec']:
            audio_filter = MovingAverage(sample_rate=args.sample_rate, window_size=args.filter_param, transf=Spectrogram(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'))
            if args.filter_type == 'MA-Spec':
                audio_filter.name = "MovingAverage-Spec"
        elif args.filter_type == 'MorFil':
            audio_filter = MorphologicalFilter(sample_rate=args.sample_rate, transf=Spectrogram(n_fft=args.nfft).to('cuda'), 
                                                n_fft=args.nfft, threshold=args.filter_param, amp=10, invers= GriffinLim(n_fft=args.nfft).to('cuda'))
        elif args.filter_type == 'MarFil':
            audio_filter = MarrasFilter(transf=Spectrogram(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'))
        elif args.filter_type in ['LowFreq', 'HighFreq', 'HighFreq-Spec']:
            with open(path, newline='') as f:
                reader = csv.reader(f)
                audio_sample_path = next(reader)
            # print(audio_sample_path)
            signal, sr = torchaudio.load(audio_sample_path[0])
            spect_fun = Spectrogram(n_fft=args.nfft).to('cuda')
            spect_signal = spect_fun(signal.to('cuda'))
            total_bins = spect_signal.shape[1]
            if args.filter_type == 'LowFreq':
                low_freq_flg = True
                cutoff_ptg = args.filter_param
            elif args.filter_type in ['HighFreq', 'HighFreq-Spec']:
                low_freq_flg = False
                cutoff_ptg = 100-args.filter_param
            audio_filter = LowHighFreq(sample_rate=args.sample_rate, freq_bins=total_bins, threshold=cutoff_ptg, low_freq_flg=low_freq_flg, 
                                transf=Spectrogram(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'), 
                                invers= GriffinLim(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'))
            if args.filter_type == 'HighFreq-Spec':
                audio_filter.name = "HighFreq-Spec"
        elif args.filter_type == 'Baseline':
            audio_filter = BaselineFilter(transf=Spectrogram(n_fft=args.nfft).to('cuda'))
            
        elif args.filter_type == 'GriffinLim':
            audio_filter = GriffinLimFilter(to_spec=Spectrogram(n_fft=args.nfft, hop_length=args.hop_len).to('cuda'), 
                                            n_fft=args.nfft, 
                                            hop_length=args.hop_len)
        """
        if args.filter_type in SIMPLE_FILTERS: 
            # Note: thr is here the cutoff_freq or the central_freq
            audio_filter = FILTERS[args.filter_type](args.sample_rate, args.filter_param)
        
        if args.filter_type in SIMPLE_FILTERS_2: 
            x = []
            if isinstance(args.filter_param, float): 
                args.filter_param = int(args.filter_param) if args.filter_param.is_integer() else args.filter_param # Convert to int and then to string
            file_in = open(f"filter_coefs/{args.filter_type}/{args.filter_param}khz.txt", 'r')
            for y in file_in.read().split('\n'):
                x.append(float(y))
            coef = torch.tensor(x)
            audio_filter = FILTERS[args.filter_type](1, coef, args.filter_type)

        elif args.filter_type in BUTTER_FILTERS:
            audio_filter = FILTERS[args.filter_type](args.sample_rate, butter_order=butter_order, butter_freq=args.filter_param, filter_type=butter_filter_type)
            
        elif args.filter_type == 'EncodecFilter':
            encodec_model = EncodecModel.encodec_model_24khz() # ToDo @Matias also for JSUT? - Matias: Yes!
            bandwidth = args.filter_param
            audio_filter = EncodecFilter(encodec_model, bandwidth, computations_samplewise=args.encodec_samplewise, device="cuda")
        elif args.filter_type == 'Oracle':
            audio_filter = FILTERS[args.filter_type]
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

        dataset = AudioDataSet(annotation_file=path, 
                    target_sample_rate=args.sample_rate,
                    train_nrows=args.num_train,
                    deterministic=args.deterministic,
                    device=device
                    )
        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)

        folder_name = name.replace("_train", "")
        if args.filter_type in BUTTER_FILTERS:
            folder_name = f"{folder_name}_{butter_order}_{butter_filter_type}"
        os.makedirs(f"{data_dir_path}/{folder_name}", exist_ok=True)
        os.makedirs(f"{plot_path}/{folder_name}", exist_ok=True)
        caching_paths = get_caching_paths(cache_dir=data_dir_path, method_name=folder_name, args=args)
        fingerprint_path = caching_paths['fingerprint']
        invcov_path = caching_paths['invcov']
        trend_path = caching_paths['trend']
        
        if not os.path.isfile(fingerprint_path):    
            print("======================================")
            print(f"Processing {name}!")
            print("======================================")
            
            if args.trend_correction or args.filter_type=='Oracle':
                # raise NotImplementedError("Trend correction is not implemented yet (JSUN implementation is missing).")
                csv_train_file = "JSUT_train.csv" if args.corpus == "jsut" else "LJSpeech_train.csv"
                real_audio_ds = AudioDataSet(f"{args.csv_dir}/{csv_train_file}", 
                                target_sample_rate=args.sample_rate,
                                train_nrows=args.num_train,
                                deterministic=args.deterministic,
                                device=device
                                )
                real_audio_dataloader = DataLoader(real_audio_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)
                wrapper.train(dataloader, real_audio_dataloader)
            else:
                wrapper.train(dataloader)
            with open(fingerprint_path, 'wb') as f:
                pickle.dump(wrapper.fingerprint, f)
            if args.scorefunction == 'mahalanobis':
                with open(invcov_path, 'wb') as f:
                    pickle.dump(wrapper.invcov, f)
            if args.trend_correction:
                with open(trend_path, 'wb') as f:
                    pickle.dump(wrapper.trend, f)
            # os.makedirs(f"{audio_path}/{folder_name}", exist_ok=True) 
            
            if args.filter_type in SIMPLE_FILTERS_2:
                plot_finger_freq(args.sample_rate, wrapper.fingerprint.cpu().squeeze(0), wrapper.name, f"{plot_path}/{folder_name}/{args.filter_param}_{args.nfft}_{args.nmels}_{args.nmfcc}_trend={args.trend_correction}_cutoff={args.cutoff}_{name}.pdf")
            elif (args.filter_type in FILTERS.keys()) or (args.filter_type in ['EncodecFilter', 'Oracle']): 
                plot_finger_freq(args.sample_rate, wrapper.fingerprint.cpu().squeeze(0), wrapper.name, f"{plot_path}/{folder_name}/{args.filter_param}_{args.nfft}_{args.nmels}_{args.nmfcc}_trend={args.trend_correction}_cutoff={args.cutoff}_{name}.pdf")
            elif args.filter_type in ['Baseline', 'MarFil', 'GriffinLim', 'HighFreq-Spec', 'MA-Spec', 'SFA-Spec']:
                # Plot the average Spectrogram, i.e., the fingerprint
                try:
                    plt.imshow(wrapper.fingerprint_db.cpu())
                except:
                    plt.imshow(wrapper.fingerprint.cpu())
                plt.savefig(f"{plot_path}/{folder_name}/{args.filter_param}_{args.nfft}_{name}.pdf")
            else:
                plot_difference(args.sample_rate, wrapper.spect_filter_avg.cpu(), wrapper.name + '_filtered', wrapper.spectrograms_avg.cpu(), wrapper.name,
                    f"{plot_path}/{folder_name}/{args.filter_param}_{args.nfft}_{name}.pdf")
                if not os.path.isdir(f"{audio_path}/{folder_name}") or True:   
                    save_audio_files(audio_ds, wrapper, f"{audio_path}/{folder_name}", f"{args.filter_param}_{args.nfft}")
    # '''
        if os.path.isfile(fingerprint_path):
            with open(fingerprint_path, 'rb') as f:
                print("fingerprint: ", fingerprint_path)
                wrapper.fingerprint = pickle.load(f)
        if args.trend_correction:
            with open(trend_path, 'rb') as f:
                print("trend: ", trend_path)
                wrapper.trend = pickle.load(f)
        if args.scorefunction == 'mahalanobis':
            with open(invcov_path, 'rb') as f:
                print("invcov: ", invcov_path)
                wrapper.invcov = pickle.load(f)

        outputs = {}
        outputs_non_pairwise = {}
        if args.filter_type == 'Oracle':
            csv_test_file = "real_test.csv"# "JSUT_test.csv" if args.corpus == "jsut" else "LJSpeech_test.csv"
            real_audio_ds = AudioDataSet(f"{args.csv_dir}/{args.corpus}/{SEED}/test/{csv_test_file}", 
                            target_sample_rate=args.sample_rate,
                            train_nrows=args.num_test,
                            deterministic=args.deterministic,
                            device=device
                            )
            real_audio_dataloader = DataLoader(real_audio_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)

        for test_path in TEST_CSV:
            if args.perturbation not in ["noise", "encodec"]:
                test_file = f"{args.csv_dir}/{args.corpus}/{SEED}/test/{test_path}"
            elif args.perturbation == "noise":
                test_file = f"{dir_data_files}/{test_path}"
            elif args.perturbation == "encodec":
                test_file = f"{args.csv_dir}/{args.perturbation}/{args.encodec_qr}/{test_path}"
            name_test = os.path.splitext(os.path.basename(test_file))[0] 
            print(name_test, args.filter_param, args.nfft)
            # print(test_file)
            audio_test_ds = AudioDataSet(test_file, 
                                target_sample_rate=args.sample_rate,
                                train_nrows=args.num_test,
                                snr = args.snr,
                                deterministic=args.deterministic,
                                device=device
                                )
            audio_test_non_pairwise_ds = AudioDataSet(test_file, 
                                target_sample_rate=args.sample_rate,
                                train_nrows=math.ceil(args.num_test / (len(TEST_CSV) - 1)),
                                snr = args.snr,
                                deterministic=args.deterministic,
                                device=device
                                )
            
            # Create a generator for consistent shuffling
            generator = torch.Generator()
            generator.manual_seed(SEED)  # Set the seed
            if args.filter_type == 'Oracle':
                if name_test == 'real_test': # in ['LJSpeech_test', 'JSUT_test']):
                    # Oracle experiments already use the original test auidio signals to calculate the residuals.
                    break
                audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)
                output = wrapper.forward(audio_test_dataloader, real_audio_dataloader, cutoff=args.cutoff)
            else:                    
                audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)
                output = wrapper.forward(audio_test_dataloader, cutoff=args.cutoff)
                audio_test_non_pairwise_dl = DataLoader(audio_test_non_pairwise_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn, generator=generator)
                output_non_pairwise = wrapper.forward(audio_test_non_pairwise_dl, cutoff=args.cutoff)
                outputs_non_pairwise[name_test] = output_non_pairwise.cpu().tolist()
            outputs[name_test] = output.cpu().tolist()
            
        auc_results[f"{folder_name}-{args.filter_param}"] = []
        labels_all = [1] * len(outputs[name.replace("_train", "_test")])
        for key_dict in outputs.keys():
            if name.replace("_train", "") != key_dict.replace("_test", ""):                
                labels = [1] * len(outputs[name.replace("_train", "_test")]) + [0] * len(outputs[key_dict])
                if args.filter_type != 'Oracle':
                    # print(labels_all)
                    labels_all += [0] * len(outputs_non_pairwise[key_dict])
                if args.scorefunction == "correlation":                    
                    x_label = "Correlation"
                elif args.scorefunction == "mahalanobis":
                    # labels = [0] * len(outputs[name.replace("_train", "_test")]) + [1] * len(outputs[key_dict])
                    x_label = "Mahalanobis dist."
                auc = roc_auc_score(labels, outputs[name.replace("_train", "_test")] + outputs[key_dict])
                # print(f"{thr}_{args.nfft} {name} vs {key_dict} - AUC={auc}")
                print(f"{auc} {args.filter_param}_{args.nfft} {name} vs {key_dict}")
                save_plot = f"{plot_path}/{folder_name}/{args.scorefunction}_{args.perturbation}_{args.filter_param}_{args.nfft}_trend={args.trend_correction}_cutoff={args.cutoff}_{key_dict}.png"
                hist_plot(save_plot, outputs[name.replace("_train", "_test")], 
                            name, outputs[key_dict], key_dict, auc, x_label)
                
                key_test_against = f"{args.filter_param}_{key_dict}"
                auc_results[f"{folder_name}-{args.filter_param}"].append({'vs_model': key_test_against, 'AUC': auc})
        
        if args.filter_type != 'Oracle':
            outputs_all = []
            for key, value in outputs_non_pairwise.items():
                if key != name.replace("_train", "_test"):
                    outputs_all += value
            auc_all = roc_auc_score(labels_all, outputs[name.replace("_train", "_test")] + outputs_all)
            print(f"{auc_all} {args.filter_param}_{args.nfft} {name} vs all")
            key_test_against = f"{args.filter_param}_all"
            auc_results[f"{folder_name}-{args.filter_param}"].append({'vs_model': key_test_against, 'AUC': auc_all})
        # Create Excel file with AUC results
        # """
        with pd.ExcelWriter(auc_path) as writer:
            for method, data in auc_results.items():
                # Convert the list of dictionaries to a DataFrame
                df = pd.DataFrame(data)
                # Write the DataFrame to an Excel sheet
                df.to_excel(writer, sheet_name=method, index=False)
        # """
        
    #'''

if __name__ == "__main__":
    args = parse_args()
    main(args)

    '''
    n_fft_values = [128, 512, 1024, 2048] # [128, 512, 1024, 2048]
    # window_sizes = [128, 128, 256, 512]
    hop_lengths = [2, 64, 128, 256]
    for n_fft in n_fft_values:
        args.nfft = n_fft
        for hop_length in hop_lengths:
            torch.cuda.empty_cache()  # Releases unused cached memory
            torch.cuda.synchronize()  # Synchronize to ensure memory operations are complete
            args.hop_len = hop_length
            # print(args)
            main(args)
    '''
