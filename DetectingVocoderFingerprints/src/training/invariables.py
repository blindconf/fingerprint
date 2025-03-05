from torch import device
from torch.cuda import is_available

# directory where to save models and logs
URL_DIR_TO_SAVE_MODELS_AND_LOGS = r'/home/hessos4l/Downloads/DetectingVocoderFingerprints/trained_models/'

# Set up device for CUDA
DEV = device("cuda:1")
DEVICE_IDS = [1, 0, 2]

FINGERPRINT_DIR = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_seed_files/fingerprints"

# dataset Paths
FAKE_AUDIO_DIR = "/USERSPACE/DATASETS/WaveFake"
REAL_AUDIO_DIR = "/USERSPACE/DATASETS/LJSpeech-1.1/wavs"
URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/fake_audio"
URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/real_audio"
URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/mix_audio"

# Path to the folder containing mean and std files
MEAN_STD_FOLDER_DIR = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/mean_std_stats"

# Others

MULTI_CLASS_10_LABELS = [
"ljspeech_avocodo",
"ljspeech_bigvgan",
"ljspeech_fast_diff_tacotron",
"ljspeech_hifiGAN",
"ljspeech_hnsf",
"ljspeech_melgan_large",
"ljspeech_multi_band_melgan",
"ljspeech_parallel_wavegan",
"ljspeech_pro_diff",
"ljspeech_waveglow"
]

MULTI_CLASS_13_LABELS = [
    "ljspeech_avocodo",
    "ljspeech_bigvgan",
    "ljspeech_fast_diff_tacotron",
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_hnsf",
    "ljspeech_lbigvgan",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_pro_diff",
    "ljspeech_waveglow"
]

CLASSES = {
    "binary-10": MULTI_CLASS_10_LABELS,
    "binary-13": MULTI_CLASS_13_LABELS,
    "multiclass-10": MULTI_CLASS_10_LABELS,
    "multiclass-13": MULTI_CLASS_13_LABELS
}

CSV_DIR_DEST = {
    "real_audio": URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES,
    "fake_audio": URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES,
    "mix_audio": URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES
}

CSV_DIR_SRC = {
    "real_audio": REAL_AUDIO_DIR,
    "fake_audio": FAKE_AUDIO_DIR
}

BINARY_CLASS_LABELS = ["real audio", "fake audio"]


TARGET_SAMPLE_RATE = {
    "resnet": 16000,
    "se-resnet": 16000,
    "x-vector": 16000,
    "lcnn": 16000,
    "vfd-resnet": 24000,
    "fingerprints": 22050
}

BATCH_SIZE = {
    "resnet": 256,
    "se-resnet": 256,
    "x-vector": 256,
    "lcnn": 256,
    "vfd-resnet":256,
    "fingerprints": 64
}
