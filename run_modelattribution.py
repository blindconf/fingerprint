"""
Spoofed Speech via audio fingerprint script.

This script implements a single-model attribution approach to recognize 
the source (algorithm) of deepfake utterances via audio fingerprinting.

Paper (Arxiv submission): https://github.com/blindconf/fingerprint/
"""

import argparse
import sys
import logging
from src.fingerprinting.filters import OracleFilter, filter_fn, EncodecFilter
import torch
import os
from src.datasets.utility_2 import load_or_construct_datasets, get_caching_paths, plot_finger_freq, hist_plot, get_auc_path
from encodec import EncodecModel
from src.fingerprinting.fingerprinting import WaveformToAvgSpec, FingerprintingWrapper
import numpy as np
from src.training.invariables import DATASETS
from torch.utils.data import DataLoader
from src.fingerprinting.audio_dataLoader import AudioDataSet, collate_fn
import pickle
from sklearn.metrics import roc_auc_score
import pandas as pd 
from statistics import mean


# Setup logger
# Initialize the logger for standardized console output across the script.
logging.basicConfig(
    level=logging.INFO,  # Set default logging level (can be changed to DEBUG for more details)
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dictionary of available filters mapped to their respective implementations.
# This dictionary connects filter names from the command line to actual functions/classes.
# It's used to dynamically instantiate the correct filter.
FILTERS = {
    "EncodecFilter": EncodecFilter,
    "Oracle": OracleFilter,
    "band_stop_filter": filter_fn,
    "band_pass_filter": filter_fn,
    "high_pass_filter": filter_fn,
    "low_pass_filter": filter_fn,
}

def parse_args():
    """
    Parse command-line arguments provided by the user.
    """
    parser = argparse.ArgumentParser()

    # Dataset selection: ljspeech, jsut, or asvspoof
    parser.add_argument("--corpus", choices=["ljspeech", "jsut", "asvspoof", "codecfake"], default="ljspeech")
    parser.add_argument("--real-data-path", help="Directory of real audio")
    parser.add_argument("--fake-data-path", help="Directory of fake audio")
    parser.add_argument("--protocol-path", type=str, help="Path to ASVspoof protocol file (required for asvspoof corpus)")

    # Filter configuration
    parser.add_argument("--filter-type", choices=FILTERS.keys(), default="low_pass_filter", help="Type of filter to apply to the audio signal.")
    parser.add_argument("--filter-param", type=str, default="1", help="Parameter of the filter.")
    parser.add_argument("--scorefunction", choices=["mahalanobis", "correlation"], default="mahalanobis", help="Type of scoring function to use.")

    # Data and processing paths
    # parser.add_argument("--csv-dir", type=str, default="csv_files")
    parser.add_argument("--nfft", type=int, default=128, help="Number of FFT points for creating the Spectrograms.")
    parser.add_argument("--hop-len", type=int, default=2, help="Hop length for creating the Spectrograms.")
    parser.add_argument("--seed", type=int, default=40, help="Default seed 40.")
    parser.add_argument("--cutoff", type=int, default=None, help="Cutoff first frequency bins for fingerprinting.")

    # Additional processing details
    parser.add_argument("--trend-correction", action="store_true", help="Correct the filter trend.")
    parser.add_argument("--batchsize", type=int, default=24, help="Adjust batch size as needed.")
    parser.add_argument("--batchsamples", type=int, default=240000, help="Each audio signal is padded or trimmed to the same length.")
    parser.add_argument("--encodec-samplewise", action="store_true", help="Encodec reencoding is applied samplewise. Strangely, the output is different depending on batch-wise or sample-wise computations.")
    parser.add_argument("--encodec-qr", choices=["1_5", "3", "6", "12", "24"], default="1_5", help="Quantization rate for the Encodec perturbation.")
    parser.add_argument("--plot-flg", action="store_true", help="If set, generates residual plots and histogram fingerprints for diagnostic analysis.")
    parser.add_argument("--corruption-type", choices=[0, 1, 2], type=int, default=0, help="Evaluate under lossy conditions: 0 = no corruption, 1 = reverberation, 2 = MP3 compression.")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="It compresses or dilates the given impulse response.")

    args = parser.parse_args()

    # Validate and convert filter parameters based on filter type
    # This section ensures correct parameters are passed based on the filter type.
    logger.info("Validating filter parameters...")
    if args.filter_type == "EncodecFilter":
        args.filter_param = float(args.filter_param)
        if args.filter_param not in [1.5, 3, 6, 12, 24]:
            parser.error("For EncodecFilter, --filter-param must be one of [1.5, 3, 6, 12, 24].")
    elif args.filter_type in ["low_pass_filter", "high_pass_filter"] and args.filter_param not in ["-1", "-500", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10.5"]:
        parser.error("For low_pass_filter and high_pass_filter, --filter-param must be one of [-1, -500, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.5].")
    elif args.filter_type in ["band_stop_filter", "band_pass_filter"] and args.filter_param not in ["1_10_5", "500_10_5", "1-10", "2-9", "3-8", "4-7", "5-6"]:
        parser.error("For band_stop_filter and band_pass_filter, --filter-param must be one of [1_10_5, 500_10_5, 1-10, 2-9, 3-8, 4-7, 5-6].")

    if args.filter_type in ["low_pass_filter", "high_pass_filter"]:
        args.filter_param = float(args.filter_param)

    args.deterministic = 1 if args.filter_type == "Oracle" else None
    args.shuffle = False if args.filter_type == "Oracle" else True

    # Assign sample rates based on corpus
    # Different corpora use different sample rates natively, which must be respected for audio fidelity.
    if args.corpus == "jsut":
        args.sample_rate = 24000
    elif args.corpus == "ljspeech":
        args.sample_rate = 22050
    else:
        args.sample_rate = 16000

    logger.info(f"Corpus: {args.corpus}, Sample Rate: {args.sample_rate}")

    return args

def main(args):
    """
    The main function where the experiment pipeline should be implemented.
    """
    logger.info("Starting main pipeline...")
    # Set the device and seed to ensure reproducibility and GPU support where available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set a global seed for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    
    # Load or construct training, validation, and test datasets.
    train_df, validate_df, test_df, real_audio_train_df, real_audio_validate_df, real_audio_test_df = load_or_construct_datasets(args)
    # Prepare directories for storing outputs such as fingerprints and plots.
    data_dir_path = f"fingerprints/{args.corpus}/{args.seed}/{args.filter_type}"
    plot_path = f"plots/{args.corpus}/{args.seed}/Avg_Spec/{args.scorefunction}/{args.filter_type}/{args.corruption_type}"
    if args.filter_type == "EncodecFilter":
        data_dir_path += f"-compute_samplewise={args.encodec_samplewise}"
        plot_path += f"-compute_samplewise={args.encodec_samplewise}"
        
    logger.info("======================================")
    logger.info(f"Fingerprinting method: {args.scorefunction}!")
    logger.info("======================================")

    auc_results = {}
    auc_path = get_auc_path(args)
    os.makedirs(os.path.dirname(auc_path), exist_ok=True)

    # Instantiate the appropriate audio filter based on the argument.
    if args.filter_type == 'EncodecFilter':
        encodec_model = EncodecModel.encodec_model_24khz() 
        bandwidth = args.filter_param
        audio_filter = FILTERS[args.filter_type](encodec_model, bandwidth, computations_samplewise=args.encodec_samplewise, device=device)
    elif args.filter_type == 'Oracle':
        audio_filter = FILTERS[args.filter_type]
    elif args.filter_type in ['band_stop_filter', 'band_pass_filter', 'high_pass_filter', 'low_pass_filter']:
        x = []
        if isinstance(args.filter_param, float): 
            # Convert to int and then to string
            args.filter_param = int(args.filter_param) if args.filter_param.is_integer() else args.filter_param 
        file_in = open(f"spectral_filter_coefs/{args.filter_type}/{args.filter_param}khz.txt", 'r')
        for y in file_in.read().split('\n'):
            x.append(float(y))
        coef = torch.tensor(x)
        audio_filter = FILTERS[args.filter_type](1, coef, args.filter_type)
    # Create a transformation to convert waveforms into averaged spectrograms.
    # This is the core feature representation for fingerprinting.
    transformation = WaveformToAvgSpec(n_fft=args.nfft, hop_length=args.hop_len)

    # Assume train_df is already defined and loaded
    attack_labels = sorted(train_df["label"].unique())
    attack_test_labels = sorted(test_df["label"].unique())

    fingerprints = {}
    CORPUS_DICT_REVERSE = {v: k for k, v in DATASETS[args.corpus].items()}
    finger_folder = f"trained_models/fingerprint/{args.corpus}/{args.seed}/{args.filter_type}"
    os.makedirs(finger_folder, exist_ok=True)

    num_train_data_dict = {}
    # Loop over each attack label (each fake model) to train a dedicated fingerprint.
    for label in attack_labels:
        os.makedirs(f"{finger_folder}/{CORPUS_DICT_REVERSE[label]}", exist_ok=True)
        attack_df = train_df[train_df["label"] == label] # .sample(n=25, random_state=42)
        # Sanity check for Mahalanobis scoring function
        args.num_train = len(attack_df)
        num_train_data_dict[label] = args.num_train
        if args.scorefunction == "mahalanobis" and args.num_train * 2 < args.nfft:
            logger.error("The sample size is too small for Mahalanobis scoring.")
            sys.exit(f"The sample size is too small. Consider reducing the nfft value ({args.nfft}) or increasing the number of training samples ({args.num_train}).")

        wrapper = FingerprintingWrapper(filter=audio_filter, 
                                        transformation=transformation, 
                                        name=args.filter_type, 
                                        filter_trend_correction=args.trend_correction, 
                                        scoring=args.scorefunction)

        dataset = AudioDataSet(
            annotation_df=attack_df[["path"]],  # Only keep the 'path' column
            target_sample_rate=args.sample_rate,
            train_nrows=False,
            deterministic=args.deterministic,
            device=device
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=args.shuffle,
            collate_fn=collate_fn
        )
        
        # os.makedirs(f"{plot_path}/{folder_name}", exist_ok=True)
        caching_paths = get_caching_paths(cache_dir=finger_folder, args=args, target_model=CORPUS_DICT_REVERSE[label])
        fingerprint_path = caching_paths['fingerprint']# .strip()
        print(fingerprint_path)
        # If fingerprint is not precomputed, train it now and optionally cache it.
        if not os.path.isfile(fingerprint_path): 
            logger.info("======================================")
            logger.info(f"Processing {CORPUS_DICT_REVERSE[label]}!")
            logger.info("======================================")
            
            if args.trend_correction or args.filter_type=='Oracle':
                real_audio_ds = AudioDataSet(
                                annotation_df=real_audio_train_df[["path"]],  
                                target_sample_rate=args.sample_rate,
                                train_nrows=len(real_audio_train_df),
                                deterministic=args.deterministic,
                                device=device
                                )
                real_audio_dataloader = DataLoader(
                                        real_audio_ds,
                                        batch_size=args.batchsize,
                                        shuffle=args.shuffle,
                                        collate_fn=collate_fn
                                    )
                wrapper.train(dataloader, real_audio_dataloader)
            else:
                wrapper.train(dataloader)
            
            with open(fingerprint_path, 'wb') as f:
                pickle.dump(wrapper.fingerprint, f)
            if args.scorefunction == 'mahalanobis':
                invcov_path = caching_paths['invcov']
                with open(invcov_path, 'wb') as f:
                    pickle.dump(wrapper.invcov, f)
            if args.trend_correction:
                trend_path = caching_paths['trend']
                with open(trend_path, 'wb') as f:
                    pickle.dump(wrapper.trend, f)
            if args.plot_flg:
                plot_folder = f"plots/fingerprint/residuals/{args.corpus}/{args.seed}/{args.filter_type}"      
                os.makedirs(plot_folder, exist_ok=True)
                plot_path = f"{plot_folder}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_ntrain={args.num_train}_model={CORPUS_DICT_REVERSE[label]}_corrtype={args.corruption_type}.pdf"
                plot_finger_freq(args.sample_rate, wrapper.fingerprint, wrapper.name, plot_path)
    
    # Evaluate each trained fingerprint against all test models (cross-model comparison).
    for label in attack_labels:
        args.num_train = num_train_data_dict[label]
        caching_paths = get_caching_paths(cache_dir=finger_folder, args=args, target_model=CORPUS_DICT_REVERSE[label])
        fingerprint_path = caching_paths['fingerprint']
        if os.path.isfile(fingerprint_path):
            with open(fingerprint_path, 'rb') as f:
                wrapper.fingerprint = pickle.load(f)
            if args.trend_correction:
                trend_path = caching_paths['trend']
                with open(trend_path, 'rb') as f:
                    wrapper.trend = pickle.load(f)
            if args.scorefunction == 'mahalanobis':
                invcov_path = caching_paths['invcov']
                with open(invcov_path, 'rb') as f:
                    wrapper.invcov = pickle.load(f)

            outputs = {}
            pd.set_option('display.max_colwidth', None)  # Shows full column content

            # Prepare dataloader for real audio, used as a negative class or baseline.
            real_audio_ds = AudioDataSet(
                annotation_df=real_audio_test_df[["path"]].sort_values(by="path"),  
                target_sample_rate=args.sample_rate,
                train_nrows=len(real_audio_test_df),
                deterministic=args.deterministic,
                device=device
                )
            real_audio_dataloader = DataLoader(
                                    real_audio_ds,
                                    batch_size=args.batchsize,
                                    shuffle=args.shuffle,
                                    collate_fn=collate_fn
                                    )

            for label_test in attack_test_labels:
                
                test_df_ = test_df[test_df["label"] == label_test] 
                dataset_test = AudioDataSet(
                    annotation_df=test_df_[["path"]],  # Only keep the 'path' column
                    target_sample_rate=args.sample_rate,
                    train_nrows=len(test_df_),
                    deterministic=args.deterministic,
                    device=device
                )

                audio_test_dataloader = DataLoader(
                                        dataset_test,
                                        batch_size=args.batchsize,
                                        shuffle=args.shuffle,
                                        collate_fn=collate_fn
                                        )

                if args.filter_type == 'Oracle':
                    output = wrapper.forward(audio_test_dataloader, real_audio_dataloader, cutoff=args.cutoff)
                else:    
                    output = wrapper.forward(audio_test_dataloader, cutoff=args.cutoff, corruption_type=args.corruption_type, scale_factor=args.scale_factor)
                outputs[label_test] = output.cpu().tolist()
                
            # '''
            if args.filter_type != 'Oracle':
                output = wrapper.forward(real_audio_dataloader, cutoff=args.cutoff)
                outputs[0] = output.cpu().tolist()
            # '''
            auc_results[f"{CORPUS_DICT_REVERSE[label]}"] = []

            for key_dict in outputs.keys():
                if label != key_dict:                
                    labels = [1] * len(outputs[label]) + [0] * len(outputs[key_dict])
                    auc = roc_auc_score(labels, outputs[label] + outputs[key_dict])
                    logger.info(f"AUC: {auc:.4f} | Filter Param: {args.filter_param}, NFFT: {args.nfft} | {CORPUS_DICT_REVERSE[label]} vs {CORPUS_DICT_REVERSE[key_dict]}")

                    key_test_against = f"{CORPUS_DICT_REVERSE[key_dict]}"
                    auc_results[f"{CORPUS_DICT_REVERSE[label]}"].append({'vs_model': key_test_against, 'AUC': auc})

                    # If enabled, plot histogram showing score distributions for each pair of models.
                    if args.plot_flg:
                        plot_folder = f"plots/fingerprint/histograms/{args.corpus}/{args.seed}/{args.filter_type}"
                        os.makedirs(plot_folder, exist_ok=True)
                        plot_path = f"{plot_folder}/param={args.filter_param}_score={args.scorefunction}_nfft={args.nfft}_hoplen={args.hop_len}_trend={args.trend_correction}_ntrain={args.num_train}_corrtype={args.corruption_type}_models={CORPUS_DICT_REVERSE[label]}_vs_{CORPUS_DICT_REVERSE[key_dict]}.png"
                        if args.scorefunction == "correlation":                    
                            x_label = "Correlation"
                        else:
                            x_label = "Mahalanobis dist."
                        hist_plot(plot_path, outputs[label], CORPUS_DICT_REVERSE[label], outputs[key_dict], CORPUS_DICT_REVERSE[key_dict], auc, x_label)

            auc_values = [entry['AUC'] for entry in auc_results[f"{CORPUS_DICT_REVERSE[label]}"]]
            auc_all = mean(auc_values)
            auc_results[f"{CORPUS_DICT_REVERSE[label]}"].append({'vs_model': 'Avg.', 'AUC': auc_all})

            # missing using real and then adding all and trying the oracle! 

            # Create Excel file with AUC results
            # Save AUC results across all comparisons to an Excel file for later review.
            # '''
            with pd.ExcelWriter(auc_path) as writer:
                for method, data in auc_results.items():
                    # Convert the list of dictionaries to a DataFrame
                    df = pd.DataFrame(data)
                    # Write the DataFrame to an Excel sheet
                    df.to_excel(writer, sheet_name=method, index=False)
            # '''                    
        else:
            continue

    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
