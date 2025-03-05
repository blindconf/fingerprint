"""
Spoofed Speech via vocoder fingerprint script.
A single-model attribution approach to recognize the source (algorithm) of deepfake utterances.
(https://arxiv.org/abs/...) # Pending arxiv submission
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import pickle
import os
from torch.utils.data import DataLoader
from src.audio_dataLoader import AudioDataSet, collate_fn
from src.fingerprinting import FingerprintingWrapper, WaveformToAvgSpec
from src.fingerprinting import EncodecFilter 
from src.filters import filter_fn, OracleFilter
from src.utils import get_caching_paths, get_csv_dict
import sys
import numpy as np 

# Latex font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

PLOT_FING = {   "ljspeech_waveglow_train" : "WaveGlow",
                "ljspeech_melgan_train" : "MelGAN",
                "ljspeech_parallel_wavegan_train" : "Parallel WaveGAN",
                "JSUT_parallel_wavegan_train" : "Parallel WaveGAN",
                "ljspeech_multi_band_melgan_train" : "Multi-Band MelGAN",
                "JSUT_multi_band_melgan_train" : "Multi-Band MelGAN",
                "ljspeech_melgan_large_train" : "MelGAN Large",
                "ljspeech_full_band_melgan_train" : "Full-Band MelGAN",
                "ljspeech_hifiGAN_train" : "HiFi GAN",                
                "ljspeech_avocodo_train" : "Avocodo",                
                "ljspeech_bigvgan_train" : "Big-V GAN",  
                "ljspeech_lbigvgan_train" : "Big-V GAN Large",             
                "real_train" : "Real audio", 
                "ljspeech_fast_diff_tacotron_train" : "FastDiff",
                "ljspeech_pro_diff_train" : "ProDiff",
                "ljspeech_hnsf_train" : "NSF"
                }

vocoders_paper = ["ljspeech_melgan_train.csv", 'ljspeech_melgan_test.csv',
                    "ljspeech_full_band_melgan_train.csv", "ljspeech_full_band_melgan_test.csv",
                    "ljspeech_lbigvgan_train.csv", "ljspeech_lbigvgan_test.csv"]

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

remove_vocoder = ["ljspeech_fast_diff_tacotron_train", "ljspeech_avocodo_train", "ljspeech_hifiGAN_train", "ljspeech_multi_band_melgan_train", 
                    "ljspeech_parallel_wavegan_train", "ljspeech_bigvgan_train"]
# ProDiff_train, ljspeech_hnsf_train, ljspeech_melgan_large_train, ljspeech_waveglow_train

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
    parser.add_argument("--nmels", type=int, default=512, help="Number of Mel bands for creating the Mel Spectrograms.")
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
    # args.test_csv = TRAIN_CSV_DICT[args.corpus]
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

    SEED = args.seed
    dir_data_files = f"{args.csv_dir}/{args.corpus}/{SEED}"
    TRAIN_CSV, TEST_CSV = get_csv_dict(dir_data_files)

    if args.vocoders == "paper":
        # Remove files that exactly match any vocoder in vocoders_paper
        TRAIN_CSV = [file for file in TRAIN_CSV if file not in vocoders_paper]
        TEST_CSV = [file for file in TEST_CSV if file not in vocoders_paper]

    # print(TRAIN_CSV)

    data_dir_path = f"fingerprints/{args.corpus}/{args.seed}/{args.filter_type}"
    plot_path = f"plots/{args.corpus}/{args.seed}/{args.transformation}{args.scorefunction}/{args.filter_type}"

    if args.filter_type == "EncodecFilter":
        data_dir_path += f"-compute_samplewise={args.encodec_samplewise}"
        plot_path += f"-compute_samplewise={args.encodec_samplewise}"
    if args.filter_type == "low_pass_filter":
        args.filter_param = int(args.filter_param)
    # Create subplots
    # fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(9, 2))  # 2 rows, 5 columns
    cont =  len(TRAIN_CSV)
    if args.corpus == "jsut":
        rows = 2
        colms = 1
        width = 4
        height = 4
    elif args.corpus == "ljspeech":
        rows = 1 # 5
        colms = 4 # 2
        width = 12 # 6
        height = 3 # 10
    fig, axes = plt.subplots(nrows=rows, ncols=colms, figsize=(width, height)) 
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Define letters for captions
    letters = [chr(i) for i in range(ord('a'), ord('a') + 10)]
    cont = 0
    # '''
    if args.filter_type == "EncodecFilter":
        for i, path_csv in enumerate(TRAIN_CSV):
            path = f"{args.csv_dir}/{args.corpus}/{SEED}/train/{path_csv}"
            name = os.path.splitext(os.path.basename(path))[0] 
            if args.corpus == 'ljspeech' and name == "ljspeech_hnsf_train":
                args.num_train = 10479
            elif args.corpus == 'ljspeech':
                args.num_train = 10480
            if name in remove_vocoder:
                continue
            print(name)

            wrapper = FingerprintingWrapper(name=name, 
                                            scoring=args.scorefunction)

            folder_name = name.replace("_train", "")
        
            if args.filter_type in BUTTER_FILTERS:
                folder_name = f"{folder_name}_{butter_order}_{butter_filter_type}"
            
            caching_paths = get_caching_paths(cache_dir=data_dir_path, method_name=folder_name, args=args)
            fingerprint_path = caching_paths['fingerprint']
            invcov_path = caching_paths['invcov']
            trend_path = caching_paths['trend']
            # print(fingerprint_path)
            if os.path.isfile(fingerprint_path):
                with open(fingerprint_path, 'rb') as f:
                    # print("fingerprint: ", fingerprint_path)
                    wrapper.fingerprint = pickle.load(f)
            if args.trend_correction:
                with open(trend_path, 'rb') as f:
                    # print("trend: ", trend_path)
                    wrapper.trend = pickle.load(f)
            if args.scorefunction == 'mahalanobis':
                with open(invcov_path, 'rb') as f:
                    # print("invcov: ", invcov_path)
                    wrapper.invcov = pickle.load(f)

            outputs = {}

            # Fingerprint
            ref_data = wrapper.fingerprint.cpu().squeeze(0) 
            # print(ref_data.shape)
            ref_data_normalized = ref_data

            num_freqs = ref_data_normalized.shape[0]
            x_khz = np.arange(num_freqs)
        
            # plot ref
            axes[cont].bar(x=x_khz, height=ref_data_normalized, color="#2D5B68")
            # Set x and y labels
            axes[cont].set_xlabel("Frequency bins", fontsize=12, labelpad=1)
            # Set the limits for the y-axis
            if args.filter_type == "EncodecFilter":
                axes[cont].set_ylim(-0.2, 0.2)

            # Set y-label only for the first plot in each row
            if args.corpus == "jsut":
                axes[cont].set_ylabel("Standardized\naverage energy (dB)", fontsize=12)
            else:
                if cont % 5 == 0: # cont % 2 == 0:
                    axes[cont].set_ylabel("Standardized\naverage energy (dB)", fontsize=12)
                else:
                    axes[cont].set_yticklabels([])  # Hide y-tick labels for other plots
            axes[cont].tick_params(axis='both', which='major', labelsize=12)
            # Add caption below the x-axis
            axes[cont].text(0.5, -0.30, f"({letters[cont]}) {PLOT_FING[wrapper.name]}", ha='center', va='top', transform=axes[cont].transAxes, fontsize=12)

            cont += 1
        
        fig.tight_layout(pad=2.0)
        fig.savefig(f"fig_paper_fingerprints.pdf", dpi=300, transparent=True, bbox_inches='tight', format='pdf')
        plt.close()

    else:
        target_vocoder = "ljspeech_pro_diff_test"
        x = []
        if isinstance(args.filter_param, float): 
            args.filter_param = int(args.filter_param) if args.filter_param.is_integer() else args.filter_param # Convert to int and then to string
        file_in = open(f"filter_coefs/{args.filter_type}/{args.filter_param}khz.txt", 'r')
        for y in file_in.read().split('\n'):
            x.append(float(y))
        coef = torch.tensor(x)
        audio_filter = FILTERS[args.filter_type](1, coef, args.filter_type)

        transformation = WaveformToAvgSpec(n_fft=args.nfft, hop_length=args.hop_len)
            
        wrapper = FingerprintingWrapper(filter=audio_filter, 
                                        transformation=transformation, 
                                        name=target_vocoder, 
                                        filter_trend_correction=args.trend_correction, 
                                        scoring=args.scorefunction)
        folder_name = target_vocoder.replace("_test", "")
        caching_paths = get_caching_paths(cache_dir=data_dir_path, method_name=folder_name, args=args)
        fingerprint_path = caching_paths['fingerprint']
        invcov_path = caching_paths['invcov']
        trend_path = caching_paths['trend']

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
        # print(TEST_CSV)
        for test_path in TEST_CSV:
            # break
            test_file = f"{args.csv_dir}/{args.corpus}/{SEED}/test/{test_path}"
            name_test = os.path.splitext(os.path.basename(test_file))[0] 
            name_test = name_test.replace("test", "train")
            # print(args.deterministic)
            audio_test_ds = AudioDataSet(test_file, 
                                target_sample_rate=args.sample_rate,
                                train_nrows=args.num_test,
                                snr = args.snr,
                                deterministic=args.deterministic,
                                device=device
                                )
            # Create a generator for consistent shuffling
            generator = torch.Generator()
            generator.manual_seed(SEED)  # Set the seed
            audio_test_dataloader = DataLoader(audio_test_ds, batch_size=args.batchsize, shuffle=args.shuffle, collate_fn=collate_fn)
            output = wrapper.forward(audio_test_dataloader, cutoff=args.cutoff)
            outputs[name_test] = output.cpu().tolist()
            # break
            # print(outputs)

        # Colors for the histograms
        colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'salmon', 'tab:cyan']
        # Define the figure size for a single column (e.g., 3.25 inches wide and 2.5 inches tall)
        fig_width = 5.25  # in inches
        fig_height = 3.5  # in inches
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        ax1.set_xlabel("Mahalanobis distance", size=12)
        ax1.set_ylabel('Normalized N° of instances', size=12)
        color_idx = 0
        # Lists to hold legend entries
        known_vocoder_labels = []
        known_vocoder_colors = []
        unknown_vocoder_labels = []
        unknown_vocoder_colors = []

        for test_path in TEST_CSV:
            test_file = f"{args.csv_dir}/{args.corpus}/{SEED}/test/{test_path}"
            name_test = os.path.splitext(os.path.basename(test_file))[0] 
            name_test = name_test.replace("test", "train")
            # print(name_test)
        
            if name_test == target_vocoder.replace("test", "train"):
                # print(name_test)
                hist_color = 'tab:blue'  # Specific color for WGlow
                label = PLOT_FING[name_test]
                ax1.hist([-x for x in outputs[name_test]], bins=100, histtype='bar', color=hist_color, alpha=1, label=label, density=True)
                known_vocoder_labels.append(label)
                known_vocoder_colors.append(hist_color)
            else:
                hist_color = colors[color_idx]
                color_idx += 1
                label = PLOT_FING[name_test]
                ax1.hist([-x for x in outputs[name_test]], bins=100, histtype='bar', color=hist_color, alpha=1, label=label, density=True)
                unknown_vocoder_labels.append(label)
                unknown_vocoder_colors.append(hist_color)
        
        # Create legend handles and labels
        # Neutral handle for "Known vocoder"
        # known_vocoder_handle = plt.Line2D([0], [0], color='tab:blue', lw=0)  # No line for category
        known_vocoder_handle = [plt.Line2D([0], [0], color=color, lw=2, markersize=10) for color in known_vocoder_colors]

        # Create handles for each unknown vocoder color
        unknown_vocoder_handles = [plt.Line2D([0], [0], color=color, lw=2, markersize=10) for color in unknown_vocoder_colors]

        # Create handles for the "Unknown vocoders" label
        known_vocoder_label_handle = plt.Line2D([0], [0], color='white', lw=0)  # No line for category
        unknown_vocoder_label_handle = plt.Line2D([0], [0], color='white', lw=0)  # No line for category

        # Set the legend entries
        legend_labels = ["Target\nvocoder:"]
        legend_handles= [known_vocoder_label_handle]
        legend_handles.extend(known_vocoder_handle)
        legend_labels.extend(known_vocoder_labels)
        # legend_labels = ["Known\nvocoder:\n" + ", ".join(known_vocoder_labels)]
        # legend_handles = [known_vocoder_handle]

        # Add handles and labels for unknown vocoders
        legend_labels.append("Unknown\nsources:")
        legend_handles.append(unknown_vocoder_label_handle)
        legend_handles.extend(unknown_vocoder_handles)
        legend_labels.extend(unknown_vocoder_labels)

        # Create and customize the legend
        legend = ax1.legend(handles=legend_handles, labels=legend_labels, frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1), prop={'size': 8.7}) #, handletextpad=1.0, columnspacing=1.0)
        # legend = ax1.legend(handles=legend_handles, labels=legend_labels, frameon=False, loc='upper right', prop={'size': 8.7}) #, handletextpad=1.0, columnspacing=1.0)
        
        # Adjust legend line width and length
        #for handle in legend.get_lines():
        #    handle.set_linewidth(2)  # Adjust width if necessary

        # Set the size of the axis ticks
        ax1.tick_params(axis='both', which='major', labelsize=8.7)

        # Adjust text alignment
        for text in legend.get_texts():
            text.set_ha('left')  # Align text to the left to make room for the colored boxes on the right

        # fig.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])  # Adjust padding to accommodate the legend
        fig.tight_layout(pad=2.0)
        plt.savefig("fig_paper_mahalanobis.pdf", dpi=300, transparent=True, bbox_inches='tight', format='pdf')

        plt.show()
        plt.close()
        
if __name__ == "__main__":
    args = parse_args()
    # print(args)
    main(args)
