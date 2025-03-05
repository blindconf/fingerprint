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
from src.utils import get_csv_dict, get_auc_path, get_caching_paths, plot_finger_freq, hist_plot # plot_difference, save_audio_files, , , , 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import csv
import torchaudio
import pandas as pd 
from encodec import EncodecModel
from src.filters import OracleFilter, filter_fn
import math
import random
from tqdm import tqdm

from scipy.spatial.distance import mahalanobis, euclidean
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
from itertools import combinations
from scipy.stats import norm
import itertools
from numpy.linalg import inv, det
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Iterate Experiments through all filters in NEW_FILTERS
FILTERS = {"low_pass_filter": filter_fn,
            "EncodecFilter": EncodecFilter,
            }

# I separate features because they require a different treatment (ButterFilters have more parameters)
SIMPLE_FILTERS_2 = ["low_pass_filter", "high_pass_filter", "band_pass_filter", "band_stop_filter"]

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

def zero_mean_unit_norm(array: torch.tensor) -> torch.tensor:
    # Calculate the mean and standard deviation along the first dimension
    array = array - array.mean(dim=-1, keepdim=True)
    return array / array.norm(dim=-1, keepdim=True)

def main(args):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.perturbation == "noise":
        dir_data_files = f"{args.csv_dir}/{args.corpus}/{args.seed}/{args.perturbation}/{last_three_chars}"
    else:
        dir_data_files = f"{args.csv_dir}/{args.corpus}/{args.seed}"
    TRAIN_CSV, TEST_CSV = get_csv_dict(dir_data_files, args.perturbation)

    if args.filter_type == "EncodecFilter":
        data_dir_path = f"/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/fingerprints/ljspeech/1/EncodecFilter-compute_samplewise=False"
    else:
        data_dir_path = f"/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/fingerprints/ljspeech/1/low_pass_filter"
        
    # Dictionary to store all FingerprintingWrapper instances keyed by path_csv name
    wrappers_dict = {}
    
    # NSF and Avo or NSF and PWG
    # 'ljspeech_hnsf_train.csv', 'ljspeech_parallel_wavegan_train.csv'
    TRAIN_CSV = ['ljspeech_avocodo_train.csv', 'ljspeech_hnsf_train.csv']
    print(TRAIN_CSV)
    for path_csv in TRAIN_CSV:
        path = f"{args.csv_dir}/{args.corpus}/{args.seed}/train/{path_csv}"
        name = os.path.splitext(os.path.basename(path))[0]    
        if args.corpus == 'ljspeech' and name == "ljspeech_hnsf_train":
            args.num_train = 10479
        elif args.corpus == 'ljspeech':
            args.num_train = 10480
                
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
        # print(wrapper.fingerprint.shape)
        # Add the wrapper to the dictionary with the path_csv name as the key
        wrappers_dict[name] = wrapper
    
    # Save the dictionary to a file named after the audio_filter
    output_file = f"{args.filter_type}_wrappers_dict.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(wrappers_dict, f)
    print(f"Residuals saved to {output_file}")
    
    with open(output_file, 'rb') as f:
        results_dict_fing = pickle.load(f)

    # Print keys and their lengths
    print("Keys and corresponding lengths in the fingerprint dictionary:")
    for key, values in results_dict_fing.items():
        print(f"{key}: {values.fingerprint.shape}")

    # Initialize an empty dictionary to store name_test and their scores
    results_dict = {}
    # 'ljspeech_hnsf_test.csv', 'ljspeech_parallel_wavegan_test.csv'
    TEST_CSV = ['ljspeech_avocodo_test.csv', 'ljspeech_hnsf_test.csv']
    print(TEST_CSV)
    for test_path in TEST_CSV:
        test_file = f"{args.csv_dir}/{args.corpus}/{args.seed}/test/{test_path}"
        name_test = os.path.splitext(os.path.basename(test_file))[0] 
        # New random seed to ensure shuffling every time the DataLoader is called
        audio_test_ds = AudioDataSet(test_file, 
                                target_sample_rate=args.sample_rate,
                                train_nrows=args.num_test,
                                snr = args.snr,
                                deterministic=args.deterministic,
                                device=device
                                )
        audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)
        scores = []
        for i in tqdm(audio_test_dataloader, desc="Evaluating fingerprint"):
            audio = i[0]
            original_audio_lengths = i[1]
            batch_sample_rate = i[2][0]
            # audio = preemphasis(i[0], 0.97)
            path = i[2]
            if audio_filter.name == "EncodecFilter":
                filtered_audio = audio_filter.forward(audio, batch_sample_rate)
            else:
                filtered_audio = audio_filter.forward(audio)            
            if audio_filter.name == "low_pass_filter":
                avg_mfcc = transformation.forward(audio, original_audio_lengths)
                filtered_avg_mfcc = transformation.forward(filtered_audio, original_audio_lengths)
                residual = avg_mfcc - filtered_avg_mfcc
                residual = zero_mean_unit_norm(residual)
                # fingerprint = self.fingerprint
                # score = mahalanobis_score(fingerprint, residual, self.invcov)
            elif audio_filter.name == "EncodecFilter":
                features = transformation.forward(audio, original_audio_lengths)
                filtered_features = transformation.forward(filtered_audio, original_audio_lengths)
                residual = features - filtered_features
                residual = zero_mean_unit_norm(residual)
                """
                if self.fingerprint.shape[1] != cutoff: 
                    fingerprint = self.fingerprint[:, :cutoff]
                    if self.scoring=="correlation":
                        fingerprint = self.zero_mean_unit_norm(self.fingerprint) # Redundant, cause the fingerprint is already normlaized. 
                else:
                    fingerprint = self.fingerprint                
                score = correlation_score(fingerprint, residual)
                """
                # print(residual.cpu().numpy(), residual.cpu().numpy().shape)
            scores.append(residual.cpu().numpy()) # Convert to NumPy for serialization
        # Store the scores in the dictionary
        results_dict[name_test] = scores

    # Save the dictionary to a file named after the audio_filter
    output_file = f"{audio_filter.name}_residuals.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Residuals saved to {output_file}")
    
    with open(output_file, 'rb') as f:
        results_dict_2 = pickle.load(f)

    # Print keys and their lengths
    print("Keys and corresponding lengths in the residuals dictionary:")
    for key, values in results_dict_2.items():
        print(f"{key}: {len(values)}")

# Load fingerprints (wrappers_dict) and test sample residuals (results_dict_2)
def load_data(res_fil):
    # Assumes wrappers_dict contains fingerprints for each vocoder
    with open(f"{res_fil}_wrappers_dict.pkl", "rb") as f:
        wrappers_dict = pickle.load(f)
    '''
    print("Keys and corresponding lengths in the fingerprint dictionary:")
    for key, values in wrappers_dict.items():
        #print(f"{key}: {wrappers_dict.fingerprint.shape}")
        print(f"{key}: {values.fingerprint.shape}")
    '''
    # Assumes results_dict_2 contains residuals for each test sample grouped by vocoder
    with open(f"{res_fil}_residuals.pkl", "rb") as f:
        results_dict_2 = pickle.load(f)
    
    '''
    print("Keys and corresponding lengths in the residuals dictionary:")
    for key, values in results_dict_2.items():
        print(f"{key}: {len(values)}")
    '''
    return wrappers_dict, results_dict_2

def mahalanobis_score(fingerprint, batch_residual, invcov):
    scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(batch_residual, batch_residual.shape)
    for i in range(batch_residual.shape[0]):
        input_residual = batch_residual[i, :, :]
        input_residual = torch.tensor(input_residual.flatten()).to(device)
        # print(input_residual.shape)
        # print(fingerprint.flatten(), fingerprint.flatten().shape)
        # print(torch.tensor(input_residual.flatten()), torch.tensor(input_residual.flatten()).shape)
        delta = input_residual - fingerprint.flatten()   
        score = torch.sqrt(torch.dot(delta, torch.matmul(invcov, delta)))
        scores.append(score.item())
    return torch.tensor(scores)

def correlation_score(fingerprint, input_residual):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Calculate the correlation scores using inner product
    # We need to remove the singleton dimensions from the fingerprint and batch elements before using torch.inner
    input_residual = torch.tensor(input_residual).squeeze(1).to(device)
    fingerprint = torch.tensor(fingerprint.squeeze(0)).to(device)
    # print(input_residual.shape, input_residual.squeeze(1).shape)
    # print(torch.tensor(fingerprint).shape, torch.tensor(fingerprint.squeeze(0)).shape)
    correlation_scores = torch.inner(input_residual, fingerprint)
    # return torch.inner(fingerprint.flatten(), input_residual.flatten())
    return correlation_scores

# Compute mean and variance of in-sample and out-of-sample residuals
def compute_residual_stats(wrappers_dict, results_dict_2):
    stats = {}
    for vocoder, wrapper in wrappers_dict.items():
        fingerprint = wrapper.fingerprint  # Fingerprint vector (65-dimensional)
        
        # Get residuals for this vocoder
        residuals = np.array(results_dict_2[vocoder.replace("_train", "_test")])  # List of residuals (N x 65)

        # Calculate distances (using Mahalanobis or other distance metric)
        if wrapper.scoring == 'mahalanobis':
            inv_cov = wrapper.invcov  # Pre-computed inverse covariance matrix
            in_sample_distances = [mahalanobis_score(fingerprint, res, inv_cov) for res in residuals]
        else:
            # Use correlation or other metric as needed
            in_sample_distances = [correlation_score(res, fingerprint) for res in residuals]

        # Calculate mean, variance, and overlap ratio
        in_mean, in_var = torch.mean(torch.stack(in_sample_distances)), torch.var(torch.stack(in_sample_distances))

        # --- Calculate pairwise out-of-sample distances ---
        pairwise_out_distributions = {}  # Store out-of-sample distances for each vocoder pair
        for other_vocoder, other_wrapper in wrappers_dict.items():
            if other_vocoder == vocoder:
                continue  # Skip comparison with itself
            
            other_fingerprint = other_wrapper.fingerprint
            other_inv_cov = other_wrapper.invcov if other_wrapper.scoring == 'mahalanobis' else None

            # Compute distances to the other vocoder's fingerprint
            if wrapper.scoring == 'mahalanobis':
                out_sample_distances = [mahalanobis_score(other_fingerprint, res, other_inv_cov) for res in residuals]
            else:
                out_sample_distances = [correlation_score(res, other_fingerprint) for res in residuals]
            # Save pairwise out-sample distances for the current vocoder pair
            # pairwise_out_distributions[other_vocoder] = out_sample_distances

            # Calculate mean and variance for out-of-sample distances
            out_mean, out_var = torch.mean(torch.stack(out_sample_distances)), torch.var(torch.stack(out_sample_distances))
            # --- Calculate overlap ratio between in-sample and out-sample distributions ---
            # Convert the list of tensors into a NumPy array
            in_samples = np.array([t.item() for t in in_sample_distances])
            out_samples = np.array([t.item() for t in out_sample_distances])
            overlap_ratio = calculate_overlap_ratio(in_samples, out_samples)
            overlap_ratio_2 = calculate_overlap_ratio_2(in_samples, out_samples)
            # Store results for this pair
            pairwise_out_distributions[other_vocoder] = {
                "mean": out_mean.item(),
                "variance": out_var.item(),
                "overlap_ratio": overlap_ratio,
                "overlap_ratio_2": overlap_ratio_2
            }
        # out_mean, out_var = np.mean(out_sample_distances), np.var(out_sample_distances)
        # overlap_ratio = calculate_overlap_ratio(in_sample_distances, out_sample_distances)

        # Save stats
        stats[vocoder] = {
            "in_sample_distances": torch.stack(in_sample_distances),
            "in_mean": in_mean.item(),
            "in_var": in_var.item(),
            "pairwise_out_distributions": pairwise_out_distributions
            # "out_mean": 0, # out_mean,
            # "out_var": 0, # out_var,
            # "overlap_ratio": 0, # overlap_ratio,
        }
        # print(vocoder, stats[vocoder], '\n')
    return stats

# Calculate overlap ratio between in-sample and out-of-sample distance distributions
def calculate_overlap_ratio(in_sample, out_sample, bins=50):
    in_hist, bin_edges = np.histogram(in_sample, bins=bins, density=True)
    out_hist, _ = np.histogram(out_sample, bins=bin_edges, density=True)
    # Calculate overlap using the minimum area between the two distributions
    overlap = np.sum(np.minimum(in_hist, out_hist) * np.diff(bin_edges))
    return overlap

def calculate_overlap_ratio_2(in_sample_distances, out_sample_distances):
    """
    Calculate the overlap ratio between two distributions using kernel density estimation (KDE).
    """
    # Estimate probability density functions using KDE
    kde_in = gaussian_kde(in_sample_distances)
    kde_out = gaussian_kde(out_sample_distances)

    # Define a range to evaluate the distributions
    min_val = min(min(in_sample_distances), min(out_sample_distances))
    max_val = max(max(in_sample_distances), max(out_sample_distances))
    evaluation_points = np.linspace(min_val, max_val, 1000)

    # Calculate the PDFs
    pdf_in = kde_in(evaluation_points)
    pdf_out = kde_out(evaluation_points)

    # Compute overlap area (minimum of the two PDFs at each point)
    overlap_area = np.sum(np.minimum(pdf_in, pdf_out)) * (evaluation_points[1] - evaluation_points[0])  # Trapezoidal integration

    return overlap_area

# Compare fingerprints across vocoders
def compute_fingerprint_similarities(wrappers_dict):
    vocoders = list(wrappers_dict.keys())
    # fingerprints = np.array([wrappers_dict[v].fingerprint for v in vocoders])
    fingerprints = [wrappers_dict[v].fingerprint for v in vocoders]
    # Compute pairwise distances between fingerprints (Euclidean)
    similarities = {}
    for i, vocoder1 in enumerate(vocoders):
        for j, vocoder2 in enumerate(vocoders):
            if i >= j:  # Avoid duplicate calculations
                continue
            sim = euclidean(fingerprints[i].cpu().squeeze().numpy(), fingerprints[j].cpu().squeeze().numpy())
            similarities[(vocoder1, vocoder2)] = sim
    # print(similarities)
    return similarities

def visualize_results(stats, fingerprints_similarities, save_dir):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Determine the number of vocoders to plot
    num_vocoders = len(stats)

    # Create subplots (6 plots per row)
    num_cols = 6
    num_rows = (num_vocoders // num_cols) + (1 if num_vocoders % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 15))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Plot residual distributions (in-sample and out-of-sample)
    for idx, (vocoder, stat) in enumerate(stats.items()):
        ax = axes[idx]        

        # In-sample and out-of-sample distributions
        in_sample_distances = stat['in_sample_distances']
        # Flatten the tensor to 1D for the histogram
        in_sample_distances = in_sample_distances.flatten().cpu().numpy()  # Flatten and convert to numpy array
        
        in_mean = stat['in_mean']
        in_var = stat['in_var']
        # print(in_sample_distances)
        # print(in_mean)
        # print(in_var)
        pairwise_out_distributions = stat['pairwise_out_distributions']

        # Plot histogram for in-sample distances
        ax.hist(in_sample_distances, bins=50, alpha=0.5, label='In-sample', color='blue')
        
        # Overlay pairwise out-of-sample distributions
        for other_vocoder, out_dist in pairwise_out_distributions.items():
            other_vocoder_label = other_vocoder.replace("_train", "").replace("ljspeech_", "")
            ax.hist(out_dist['mean'], alpha=0.5, label=f"{other_vocoder_label}-{out_dist['mean']:.2f})", color='orange', density=True)

        # Set title, labels, and legend
        ax.set_title(f"Residual Distributions for {vocoder}")
        if args.filter_type == "EncodecFilter":
            ax.set_xlabel('Correlation')
        else:
            ax.set_xlabel('Distance')
        ax.set_ylabel('Normalized N of instances')
        ax.legend()

    # Remove any unused axes (in case there are fewer vocoders than subplots)
    for i in range(num_vocoders, len(axes)):
        fig.delaxes(axes[i])        

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_dir, "residual_distributions_all_vocoders.pdf"))
    plt.close()  # Close the plot to avoid memory issues with many plots

    # Plot pairwise fingerprint similarities (mean and overlap ratio)
    vocoder_pairs = list(fingerprints_similarities.keys())
    vocoder_pairs_label = []
    for p in vocoder_pairs:
        p_one = p[0].replace("_train", "").replace("ljspeech_", "")
        p_two = p[1].replace("_train", "").replace("ljspeech_", "")
        vocoder_pairs_label.append(f"{p_one}-{p_two}")
    # print(fingerprints_similarities.values())
    # means = [pairwise['mean'] for pairwise in fingerprints_similarities.values()]
    means = [pairwise for pairwise in fingerprints_similarities.values()]

    # Plot means of pairwise fingerprint similarities
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(means)), means, alpha=0.7, label='Mean Similarity (Euclidean Distance)')
    
    plt.xticks(range(len(vocoder_pairs)), vocoder_pairs_label, rotation=90)
    plt.title("Pairwise Fingerprint Similarities (Mean Euclidean Distance)")
    plt.ylabel('Euclidean Distance')
    plt.xlabel('Vocoder Pair')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairwise_fingerprint_similarities_mean.pdf"))
    plt.close()

    # Create a new figure for pairwise overlap ratios
    vocoder_pairs = []
    overlap_ratios = []
    
    for vocoder, stat in stats.items():
        for other_vocoder, pairwise_stat in stat['pairwise_out_distributions'].items():
            vocoder_label = vocoder.replace("_train", "").replace("ljspeech_", "")
            other_vocoder_label = other_vocoder.replace("_train", "").replace("ljspeech_", "")
            vocoder_pairs.append(f"{vocoder_label}-{other_vocoder_label}")
            overlap_ratios.append(pairwise_stat['overlap_ratio_2'])

    # Plot pairwise overlap ratios
    plt.figure(figsize=(18, 6))
    x_positions = np.arange(len(overlap_ratios)) * 1.5  # Multiply by a spacing factor (e.g., 1.5)
    plt.bar(range(len(overlap_ratios)), overlap_ratios, alpha=0.7, color='green')
    plt.xticks(range(len(overlap_ratios)), vocoder_pairs, rotation=90)
    plt.title("Pairwise Overlap Ratios")
    plt.ylabel('Overlap Ratio')
    plt.xlabel('Vocoder Pair')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairwise_overlap_ratios.pdf"))
    plt.close()

# Helper function to generate sample data from mean and variance
def generate_sample_data(mean, variance, size=1000):
    return np.random.normal(loc=mean, scale=np.sqrt(variance), size=size)

def in_sample_ratio(data):
    # Extracting in-sample data
    samples = {}
    for model, info in data.items():
        in_mean = info['in_mean']
        in_var = info['in_var']
        samples[model] = generate_sample_data(in_mean, in_var)

    # Performing KS test to check for similarity
    results = []
    models = list(data.keys())
    for model1, model2 in combinations(models, 2):
        ks_stat, p_value = ks_2samp(samples[model1], samples[model2])
        results.append((model1, model2, ks_stat, p_value))

    # Display results
    for result in results:
        model1, model2, ks_stat, p_value = result
        print(f"KS Test between {model1} and {model2}: KS Statistic = {ks_stat:.4f}, P-value = {p_value:.4f}")

    # Identify similar distributions based on a threshold for p-value (e.g., p-value > 0.05 indicates similarity)
    similar_distributions = [(model1, model2) for model1, model2, ks_stat, p_value in results if p_value > 0.01]

    print("\nSimilar distributions:")
    for model1, model2 in similar_distributions:
        print(f"{model1} and {model2}")

def in_sample_ratio_2(data):
    # Extracting in-sample data
    samples = {}
    for model, info in data.items():
        samples[model] = np.array([t.item() for t in info['in_sample_distances']])
    
    # Performing KS test to check for similarity
    results = []
    models = list(data.keys())
    for model1, model2 in combinations(models, 2):
        ratio = calculate_overlap_ratio_2(samples[model1], samples[model2])
        results.append((model1, model2, ratio))

    # Display results
    for result in results:
        model1, model2, ratio = result
        print(f"Overlap ratio between {model1} and {model2}: overlap ratio = {ratio:.4f}")

# Function to fit a high-dimensional Gaussian distribution
def fit_high_dimensional_gaussian(samples):
    """
    Fits a high-dimensional Gaussian to a list of samples.

    Args:
        samples (list of arrays): List of 2620 samples, each of shape (1, 1, 65).

    Returns:
        tuple: Mean vector (shape: [65,]), Covariance matrix (shape: [65, 65]).
    """
    # Step 1: Convert samples list into a 2D array of shape (2620, 65)
    data = np.squeeze(np.array(samples), axis=(1, 2))  # Shape: (2620, 65)
    '''
    data = zero_mean_unit_norm_numpy(data)
    if args.filter_type == "EncodecFilter":
        data_components = 110
    else:
        data_components = data.shape[1]
    # Not sure why whithout the PCA I get nan values, check what this function is doing to the data as I am choosing all dimensions! 
    # I also use the StandardScaler to fit_transform() and standarized the data, which provides the results reported, not sure if I should use it. 
    pca = PCA(n_components=data_components)  # This will retain 95% of the variance
    # Fit PCA and transform the data
    data = pca.fit_transform(data)
    data = zero_mean_unit_norm_numpy(data)
    # print(scaled_data.shape)
    '''
    mean_vector = np.mean(data, axis=0)
    # Step 3: Compute covariance matrix (shape: [65, 65])
    covariance_matrix = np.cov(data, rowvar=False)
    return mean_vector, covariance_matrix

def zero_mean_unit_norm_numpy(array: np.ndarray) -> np.ndarray:
    """
    Normalize the input array to have zero mean and unit norm (L2 norm) for each row.
    
    :param array: A numpy array of shape (n_samples, n_features)
    :return: The normalized numpy array
    """
    # Calculate the mean along the last axis (axis=-1)
    array = array - np.mean(array, axis=-1, keepdims=True)
    
    # Calculate the L2 norm along the last axis (axis=-1)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    
    # Avoid division by zero in case the norm is zero
    array = array / np.maximum(norm, 1e-8)
    
    return array

# Example usage for multiple vocoders
def fit_gaussians_per_vocoder(results_dict_2):
    """
    Fits high-dimensional Gaussians for each vocoder in results_dict_2.

    Args:
        results_dict_2 (dict): Dictionary where keys are vocoder names and
                               values are lists of samples (2620 per vocoder).

    Returns:
        dict: Dictionary of Gaussian parameters (mean and covariance) per vocoder.
    """
    gaussian_params = {}
    for vocoder, samples in results_dict_2.items():
        # print(vocoder)
        # if vocoder in ["ljspeech_full_band_melgan_test", "ljspeech_avocodo_test", "ljspeech_lbigvgan_test"]:
        # if vocoder not in ["LJSpeech_test"]:
        mean, cov = fit_high_dimensional_gaussian(samples)
        print(np.max(mean), np.min(mean))
        # print(vocoder, mean.shape, cov.shape)
        gaussian_params[vocoder] = {"mean": mean, "covariance": cov}

    return gaussian_params

# Function to compute overlapping probability mass between two Gaussian distributions
def integrate_overlap(dist1, dist2):
    from scipy.integrate import quad

    # Define the overlap function: minimum of the two PDFs
    def overlap_func(x):
        return np.minimum(dist1.pdf(x), dist2.pdf(x))

    # Integrate over a reasonable range (e.g., -10 to 10 standard deviations)
    range_min = min(dist1.mean() - 10 * dist1.std(), dist2.mean() - 10 * dist2.std())
    range_max = max(dist1.mean() + 10 * dist1.std(), dist2.mean() + 10 * dist2.std())

    overlap_area, _ = quad(overlap_func, range_min, range_max)
    return overlap_area

def kl_divergence_gaussian(mean_p, cov_p, mean_q, cov_q):
    """
    Computes KL divergence D_KL(P || Q) between two multivariate Gaussians.
    
    Args:
        mean_p (np.ndarray): Mean vector of distribution P (shape: [65,]).
        cov_p (np.ndarray): Covariance matrix of distribution P (shape: [65, 65]).
        mean_q (np.ndarray): Mean vector of distribution Q (shape: [65,]).
        cov_q (np.ndarray): Covariance matrix of distribution Q (shape: [65, 65]).
    
    Returns:
        float: KL divergence D_KL(P || Q).
    """
    k = mean_p.shape[0]  # Dimensionality (65 in this case)
    
    epsilon = 1e-6  # Small regularization constant
    cov_p += epsilon * np.eye(cov_p.shape[0])
    cov_q += epsilon * np.eye(cov_q.shape[0])
    
    # Compute the terms in the KL divergence formula
    cov_q_inv = inv(cov_q)
    mean_diff = mean_q - mean_p
    trace_term = np.trace(cov_q_inv @ cov_p)
    mean_term = mean_diff.T @ cov_q_inv @ mean_diff
    if det(cov_p) <= 1e-10 or det(cov_q) <= 1e-10:
        log_det_term = 0  # Handle singular case
    else:
        log_det_term = np.log(det(cov_q) / det(cov_p))
        
    # log_det_term = np.log(det(cov_q) / det(cov_p))
    
    # KL divergence
    kl_div = 0.5 * (trace_term + mean_term - k + log_det_term)
    return kl_div

def compute_pairwise_kl(gaussian_params):
    """
    Computes pairwise KL divergences between Gaussian distributions for each vocoder.

    Args:
        gaussian_params (dict): Dictionary where keys are vocoder names and values are
                                {"mean": mean_vector, "covariance": covariance_matrix}.
    
    Returns:
        dict: Pairwise KL divergences between vocoders (both directions).
    """
    vocoders = list(gaussian_params.keys())
    pairwise_kl = {}

    # Compute KL divergence for each pair of vocoders
    for i, vocoder_p in enumerate(vocoders):
        for j, vocoder_q in enumerate(vocoders):
            if i < j:  # Avoid duplicate computations
                mean_p, cov_p = gaussian_params[vocoder_p]["mean"], gaussian_params[vocoder_p]["covariance"]
                mean_q, cov_q = gaussian_params[vocoder_q]["mean"], gaussian_params[vocoder_q]["covariance"]
                kl_pq = kl_divergence_gaussian(mean_p, cov_p, mean_q, cov_q)
                kl_qp = kl_divergence_gaussian(mean_q, cov_q, mean_p, cov_p)
                sym_kl = 0.5 * (kl_pq + kl_qp)

                pairwise_kl[(vocoder_p, vocoder_q)] = {
                    "KL(P||Q)": kl_pq,
                    "KL(Q||P)": kl_qp,
                    "Symmetric KL": sym_kl,
                }
                print(pairwise_kl)
    return pairwise_kl

def find_most_overlapping(pairwise_kl, top_n=5):
    """
    Finds the pairs of vocoders with the most overlap based on symmetric KL divergence.

    Args:
        pairwise_kl (dict): Pairwise KL divergences between vocoders (output of compute_pairwise_kl).
        top_n (int): Number of top overlapping pairs to return.
    
    Returns:
        list: Top N overlapping pairs sorted by lowest symmetric KL divergence.
    """
    # Extract pairs and their symmetric KL divergence
    overlaps = [(pair, kl_values["Symmetric KL"]) for pair, kl_values in pairwise_kl.items()]
    
    # Sort by lowest symmetric KL divergence
    overlaps.sort(key=lambda x: x[1])
    
    # Return the top N pairs
    return overlaps[:top_n]

def standardize_variance(variances):
    """
    Standardizes the variances by subtracting the mean and dividing by the standard deviation.
    
    :param variances: A 1D array of variances (diagonal of covariance matrix)
    :return: A 1D array of standardized variances
    """
    mean_variance = np.mean(variances)
    std_variance = np.std(variances)
    
    # Standardize variances
    standardized_variances = (variances - mean_variance) / std_variance
    return standardized_variances

# Function to compute variances from the covariance matrix
def get_variances(covariance_matrix):
    """
    Extract the diagonal values (variances) from the covariance matrix.
    :param covariance_matrix: A 2D numpy array (covariance matrix)
    :return: A 1D array of variances (diagonal values of covariance matrix)
    """
    return np.diagonal(covariance_matrix)

# Function to calculate pairwise variance differences for all pairs and sort by difference
def calculate_and_sort_variance_differences(models):
    """
    Calculate the variance differences for all pairwise models and sort them in descending order.
    
    :param models: A dictionary of models, each containing a 'covariance' matrix
    :return: A sorted list of tuples with model pair names and their corresponding variance difference score
    """
    # Extract the variances for all models
    model_variances = []
    for model_name, model in models.items():
        variances = get_variances(model['covariance'])
        model_variances.append((model_name, variances))
    # print(model_variances)
    # List to store all pairwise variance differences
    variance_differences = []

    # Compare each pair of models
    for i in range(len(model_variances)):
        for j in range(i + 1, len(model_variances)):
            model_i_name, variances_i = model_variances[i]
            model_j_name, variances_j = model_variances[j]
            # Calculate the absolute difference in variances
            diff = np.sum(np.abs(variances_i - variances_j))  # Sum of absolute differences in variances
            variance_differences.append(((model_i_name, model_j_name), diff))

    # Sort the pairwise differences in descending order
    variance_differences.sort(key=lambda x: x[1], reverse=True)
    
    return variance_differences

# Main execution
if __name__ == "__main__":
    args = parse_args()
    # main(args)
    wrappers_dict, results_dict_2 = load_data(args.filter_type)
    # print(results_dict_2)
    # Fit Gaussians and compute overlaps
    gaussian_params = fit_gaussians_per_vocoder(results_dict_2)
    # print(gaussian_params)
    # '''
    # Extract variances from the covariance matrices
    # Example: Assume 'models' is your dictionary of models, each with 'covariance' matrix
    variance_differences = calculate_and_sort_variance_differences(gaussian_params)
    # Print the pairwise models with their variance difference score in descending order
    print("Pairwise Models Sorted by Variance Difference (Descending):")
    for (model_pair, diff) in variance_differences:
        print(f"Models: {model_pair[0]} vs {model_pair[1]} | Variance Difference: {diff:.6f}")
    # '''
    # '''
    # Compute pairwise KL divergences
    pairwise_kl = compute_pairwise_kl(gaussian_params)
    # Find the most overlapping vocoder pairs
    most_overlapping = find_most_overlapping(pairwise_kl, top_n=4)
    # Output results
    print("Top overlapping vocoder pairs:")
    for pair, sym_kl in most_overlapping:
        print(f"Pair: {pair}, Symmetric KL: {sym_kl:.4f}")
    # Output results
    for pair, kl_values in pairwise_kl.items():
        print(f"Pair: {pair}")
        print(f"  KL(P||Q): {kl_values['KL(P||Q)']}")
        print(f"  KL(Q||P): {kl_values['KL(Q||P)']}")
        print(f"  Symmetric KL: {kl_values['Symmetric KL']}")   
    # '''  
    # stats = compute_residual_stats(wrappers_dict, results_dict_2)
    # in_sample_ratio(stats)
    # in_sample_ratio_2(stats)
    # fingerprint_similarities = compute_fingerprint_similarities(wrappers_dict)
    # visualize_results(stats, fingerprint_similarities, f"plots/{args.filter_type}")
