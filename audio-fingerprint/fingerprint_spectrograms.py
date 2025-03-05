<<<<<<< HEAD
from src.data import read_pickle_spectrogram_data
from src.utils import spec_to_mel_spec, mel_spec_to_spec
from src.fingerprinting import HighPassFilter, FingerprintingWrapper, MorphologicalFilter, MarrasFilter
import matplotlib.pyplot as plt
=======
from src.fingerprinting import HighPassFilter, FingerprintingWrapper
from src.audio_dataLoader import AudioDataSet
>>>>>>> results_1907
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
from sklearn.metrics import roc_auc_score
<<<<<<< HEAD

from argparse import ArgumentParser



n_train = 2000
n_test = 500
my_gan = "melgan"
paths = {"melgan": "/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_melgan",
         "hifiGAN": "/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_hifiGAN",
         "waveglow": "/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_waveglow"}


paths_test_other = ["/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_hifiGAN",
                    "/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_waveglow"]
=======
# New imports
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import recognizer.tools as tools
import csv
from scipy.io.wavfile import read as read_wav
import librosa
import matplotlib.pyplot as plt
from argparse import ArgumentParser


N_TRAIN = 2000
N_TEST = 500
MY_GAN = "melgan"
# Generate audio features during training
PATHS = {"melgan": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_melgan.csv",
         "hifiGAN": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_hifiGAN.csv",
         "waveglow": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_waveglow.csv",
         "pwg": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_parallel_wavegan.csv",
         "mb_melgan": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_multi_band_melgan.csv",
         "fb_melgan": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_full_band_melgan.csv",
         "ljspeech": "/USERSPACE/pizarm5k/audio_fingerprint/csv_files/ljspeech_full_band_melgan.csv"
         }

#PATHS_TEST_OTHER = ["/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_hifiGAN",
#                    "/USERSPACE/pizarm5k/audio_fingerprint/spectrogram_data/ljspeech_waveglow"]

REFORMAT = "crop" # "crop" or "pad_zeros"
# Feature extraction parameters
FE_TYPE = 'spectrogram'
N_FFT = 2048
HOP_LENGTH = 64
WIN_LENGTH = None
N_MELS = None
N_MFCC = None
N_LPCC = None
BATCH_SIZE = 64
NUM_SAMPLES = 22050
>>>>>>> results_1907

os.makedirs("spectrogram_plots", exist_ok=True)
os.makedirs("plots", exist_ok=True)

<<<<<<< HEAD
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--filter", type=str, choices=["morphological", "highpass", "marras"], default="highpass")
    parser.add_argument("--threshold-highpass", type=float, default=9.5, help="Threshold for highpass filter")
    parser.add_argument("--reformat", type=str, choices=["crop", "pad_zeros"], help="Spectograms are of different size. Crop or pad with zeros to make consistent?", default="crop")
    parser.add_argument("--threshold-morphological", type=float, default=0.2, help="Threshold for binary thresholding in morphological filter")
    parser.add_argument("--use_mel", action="store_true", help="Use mel spectrograms instead of linear spectrograms")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()

    path_train = paths[my_gan]
    spectrogram_data_train = read_pickle_spectrogram_data(path_train)[:n_train]
    spectrograms_train = []

    counter_plots = 0
    min_num_time_bins = 100000
    max_num_time_bins = 0
    # process training data
    for spectrogram, _, _ in spectrogram_data_train:
        if counter_plots < 5:
            # just for visualization
            plt.imshow(spectrogram.T)
            plt.xlabel("time frame")
            plt.ylabel("frequency bin")
            plt.savefig(f"spectrogram_plots/spectrogram_{counter_plots}.png")

            plt.clf()
            mel_spec = spec_to_mel_spec(spectrogram)
            plt.imshow(mel_spec.T)
            plt.xlabel("time frame")
            plt.ylabel("frequency bin")
            plt.savefig(f"spectrogram_plots/mel_spec_{counter_plots}.png")
        counter_plots += 1

        if args.use_mel:
            spectrogram = spec_to_mel_spec(spectrogram)
        spectrograms_train.append(spectrogram)


        if spectrogram.shape[0] < min_num_time_bins:
            min_num_time_bins = spectrogram.shape[0]
        if spectrogram.shape[0] > max_num_time_bins:
            max_num_time_bins = spectrogram.shape[0]

    # process test data:
    spectrograms_test = {}
    for name in paths.keys():
        path = paths[name]
        spectrogram_data = read_pickle_spectrogram_data(path)[n_train:n_train + n_test]
        spectrograms_test[name] = []
        for spectrogram, _, _ in spectrogram_data:
            if args.use_mel:
                spectrogram = spec_to_mel_spec(spectrogram)
            spectrograms_test[name].append(spectrogram)

            if spectrogram.shape[0] < min_num_time_bins:
                min_num_time_bins = spectrogram.shape[0]
            if spectrogram.shape[0] > max_num_time_bins:
                max_num_time_bins = spectrogram.shape[0]

    for j in range(len(spectrograms_train)):
        spectrogram = spectrograms_train[j]
        if args.reformat == "crop":
            spectrograms_train[j] = spectrogram[:min_num_time_bins, :] # crop only to the beginning of the spectrogram
        elif args.reformat == "pad_zeros":
            spectrograms_train[j] = np.pad(spectrogram, ((0, max_num_time_bins - spectrogram.shape[0]), (0, 0)), mode="constant", constant_values=0)

    for name in spectrograms_test.keys():
        spectrogram_test = spectrograms_test[name]
        for j in range(len(spectrogram_test)):
            spectrogram = spectrogram_test[j]
            if args.reformat == "crop":
                spectrogram_test[j] = spectrogram[:min_num_time_bins, :] # crop only to the beginning of the spectrogram
            elif args.reformat == "pad_zeros":
                spectrogram_test[j] = np.pad(spectrogram, ((0, max_num_time_bins - spectrogram.shape[0]), (0, 0)), mode="constant", constant_values=0)
        spectrograms_test[name] = spectrogram_test


    spectrograms_train = np.asarray(spectrograms_train)
    for name in spectrograms_test.keys():
        spectrograms_test[name] = np.asarray(spectrograms_test[name])
=======
def create_data_loader(train_data, batch_size):
    train_dL = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dL

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--filter", type=str, choices=["morphological", "highpass", "marras"], default="highpass")
    parser.add_argument("--threshold-highpass", type=float, default=9.5, help="Threshold for highpass filter")
    parser.add_argument("--reformat", type=str, choices=["crop", "pad_zeros"], help="Spectograms are of different size. Crop or pad with zeros to make consistent?", default="crop")
    parser.add_argument("--threshold-morphological", type=float, default=0.2, help="Threshold for binary thresholding in morphological filter")
    parser.add_argument("--use_mel", action="store_true", help="Use mel spectrograms instead of linear spectrograms")
    args = parser.parse_args()
    return args

>>>>>>> results_1907

if __name__=="__main__":
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

<<<<<<< HEAD
    if args.filter=="highpass":
        audio_filter = HighPassFilter(num_frequency_bins=num_frequency_bins, threshold=args.threshold_highpass)
        threshold = args.threshold_highpass
    elif args.filter=="morphological":
        audio_filter = MorphologicalFilter(threshold=args.threshold_morphological, mel_spec_to_spec=args.use_mel) # morphological filter is defined for linear spectrograms
        threshold = args.threshold_morphological
    elif args.filter=="marras":
        audio_filter = MarrasFilter() # Defeault settings for now..
    wrapper = FingerprintingWrapper(filter=audio_filter)

    # test morphological filter:
    if args.filter=="morphological":
        for j in range(5):
            data = spectrograms_train[j]
            filtered = audio_filter.forward(data)
            plt.clf()
            plt.imshow(filtered.T)
            plt.xlabel("time frame")
            plt.ylabel("frequency bin")
            plt.savefig(f"spectrogram_plots/{args.filter}-{j}-use_mel={args.use_mel}-threshold={threshold}.png")
            plt.clf()
            plt.imshow(data.T)
            plt.xlabel("time frame")
            plt.ylabel("frequency bin")
            plt.savefig(f"spectrogram_plots/use_mel={args.use_mel}-unfiltered_{j}.png")

    train_loader = DataLoader(spectrograms_train, batch_size=64, shuffle=True)
    wrapper.train(train_loader)

    # testing
    plt.clf()
    outputs = {}
    for name in spectrograms_test.keys():
        output = wrapper.forward(torch.tensor(spectrograms_test[name]))
        plt.hist(output, label=name, alpha=0.4)
        outputs[name] = output

    plt.legend()
    plt.xlabel("Fingerprinting score")
    plt.title("Fingerprinting Score Histograms")
    plt.savefig(f"plots/my={my_gan}-filter={audio_filter.name}-use_mel={args.use_mel}-threshold={threshold}-reformat={args.reformat}.png")

    # compute aucs:
    for name in spectrograms_test.keys():
        if name != my_gan:
            labels = [1] * len(outputs[my_gan]) + [0] * len(outputs[name])
            auc = roc_auc_score(labels, outputs[my_gan] + outputs[name])
            print(f"{my_gan} vs {name} - use_mel={args.use_mel} - threshold={threshold} - reformat={args.reformat}:")
            print(f"AUC={auc}")
=======
    path_train = PATHS[MY_GAN]
    speech_train = []
    counter = 0
    # process training data
    
    with open(PATHS[MY_GAN], newline='') as f:
        reader = csv.reader(f)
        while counter < N_TRAIN:
            row1 = next(reader) 
            # Load audio
            speech_wave, sr = torchaudio.load(row1[0])     
            speech_train.append(speech_wave.to(device))
            counter += 1

    print(f"Sampling rate of {sr}")
    
    # process test data:
    speech_test = {}
    speech_name = {}
    for name in PATHS.keys():
        path = PATHS[name]
        speech_test[name] = []
        counter = 0
        with open(path, newline='') as f:
            test_lines = f.readlines()[N_TRAIN:N_TRAIN + N_TEST]
            for wav_file in test_lines:
                # Load audio
                speech_wave, sr_test = torchaudio.load(wav_file.strip())     
                speech_test[name].append(speech_wave.to(device))
                counter += 1
                # print(f"GA model {name} Sampling rate of {sr}")
                # break
        assert sr_test == sr, "Different sampling rate between data sets!"
        
    if REFORMAT == "crop":
        all_data = []
        all_data += speech_train
        for name in PATHS.keys():
            all_data += speech_test[name]
        print(f"Total number of data {len(all_data)}")
        for i in all_data:
            if NUM_SAMPLES < i.shape[1]:
                NUM_SAMPLES = i.shape[1]
        fe_transformation = torchaudio.transforms.MFCC(sample_rate=sr,
                                                        n_mfcc=256,
                                                        melkwargs={
                                                            "n_fft": 2048,
                                                            "n_mels": 256,
                                                            "hop_length": 512,
                                                            "mel_scale": "htk",
                                                        },
                                                    ).to(device)
        min_num_time_bins, max_num_time_bins, num_frequency_bins = tools.crop_bounderies(all_data,
                                                                                        FE_TYPE, 
                                                                                        fe_transformation,
                                                                                        device
                                                                                        )                                                                                        
        print(f"Min bin {min_num_time_bins}")
        print(f"Max bin {max_num_time_bins}")
        print(f"Frequency resolution {num_frequency_bins}")
    
    for threshold in [8, 9, 9.5, 9.8]: # @Matias This means 80%, 90%,... of the lowest frequencies are removed
        if args.filter=="highpass":
            audio_filter = HighPassFilter(num_frequency_bins=num_frequency_bins, threshold=args.threshold_highpass)
            threshold = args.threshold_highpass
        elif args.filter=="morphological":
            audio_filter = MorphologicalFilter(threshold=args.threshold_morphological, mel_spec_to_spec=args.use_mel) # morphological filter is defined for linear spectrograms
            threshold = args.threshold_morphological
        elif args.filter=="marras":
            audio_filter = MarrasFilter() # Defeault settings for now..

        audio_filter = HighPassFilter(num_frequency_bins=num_frequency_bins, threshold=threshold, transformation=fe_transformation, time_bound=min_num_time_bins)
        wrapper = FingerprintingWrapper(filter=audio_filter)

        audio_dL = AudioDataSet(PATHS[MY_GAN], 
                                # fe_transformation,
                                sr,
                                NUM_SAMPLES, 
                                N_TRAIN,
                                None, 
                                device
                                )
        train_loader = create_data_loader(audio_dL, BATCH_SIZE)
        wrapper.train(train_loader)

        # testing
        plt.clf()
        outputs = {}
        for name in PATHS.keys():
            audio_dL = AudioDataSet(PATHS[name], 
                                # fe_transformation,
                                sr,
                                NUM_SAMPLES, 
                                N_TRAIN,
                                N_TEST,
                                device
                                )
            test_loader = create_data_loader(audio_dL, N_TEST)
            output = wrapper.forward(next(iter(test_loader)))
            # print(output[:10])
            plt.hist(output, label=name, alpha=0.4)
            outputs[name] = output
        
        plt.legend()
        plt.xlabel("Fingerprinting score")
        plt.title("Fingerprinting Score Histograms")
        plt.savefig(f"plots/my={MY_GAN}-threshold={threshold}-REFORMAT={REFORMAT}.png")
        
        # compute aucs:
        for name in PATHS.keys():
            if name != MY_GAN:
                labels = [1] * len(outputs[MY_GAN]) + [0] * len(outputs[name])
                auc = roc_auc_score(labels, outputs[MY_GAN] + outputs[name])
                print(f"{MY_GAN} vs {name} - threshold={threshold} - REFORMAT={REFORMAT}:")
                print(f"AUC={auc}")
    
>>>>>>> results_1907
