from torch import device
from torch.cuda import is_available

# directory where to save models and logs
URL_DIR_TO_SAVE_MODELS_AND_LOGS = r'trained_models/'

# Set up device for CUDA
DEV = device("cuda:1")
DEVICE_IDS = [1, 0, 2]

FINGERPRINT_DIR = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_seed_files/fingerprints"

# dataset Paths
# FAKE_AUDIO_DIR = "/USERSPACE/DATASETS/WaveFake"
# REAL_AUDIO_DIR = "/USERSPACE/DATASETS/LJSpeech-1.1/wavs"
# URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/fake_audio"
# URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/real_audio"
# URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES = "/home/hessos4l/Downloads/DetectingVocoderFingerprints/csv_dir/mix_audio"

# Path to the folder containing mean and std files
MEAN_STD_FOLDER_DIR = "/mean_std_stats"

# Others
LJSPEECH = {
    'Real': 0,
    'ljspeech_avocodo': 1,
    'ljspeech_bigvgan': 2,
    'ljspeech_fast_diff_tacotron': 3,
    'ljspeech_hifiGAN': 4,
    'ljspeech_hnsf': 5,
    'ljspeech_melgan_large': 6,
    'ljspeech_multi_band_melgan': 7,
    'ljspeech_parallel_wavegan': 8,
    'ljspeech_pro_diff': 9,
    'ljspeech_waveglow': 10
}

JSUT = {
    'Real': 0,
    'jsut_multi_band_melgan': 1,
    'jsut_parallel_wavegan' : 2,
    }

ASVSPOOF = {
    'Real': 0,
    'A01': 1,
    'A02': 2,
    'A03': 3,
    'A04': 4,
    'A05': 5,
    'A06': 6,
    'A07': 7,
    'A08': 8,
    'A09': 9,
    'A10': 10,
    'A11': 11,
    'A12': 12,
    'A13': 13,
    'A14': 14,
    'A15': 15,
    'A16': 16,
    'A17': 17,
    'A18': 18,
    'A19': 19
}

DATASETS = {
    # "binary-ljspeech": MULTI_CLASS_10_LABELS,
    # "binary-jsut": MULTI_CLASS_13_LABELS,
    # "binary-asvspoof": MULTI_CLASS_13_LABELS,
    "ljspeech": LJSPEECH,
    "jsut": JSUT,
    "asvspoof": ASVSPOOF
    # "multiclass-jsut": MULTI_CLASS_jsut, 
    # "multiclass-asvspoof": MULTI_CLASS_asvspoof
}

'''
CSV_DIR_DEST = {
    "real_audio": URL_DIR_TO_SAVE_REAL_AUDIO_CSV_FILES,
    "fake_audio": URL_DIR_TO_SAVE_FAKE_AUDIO_CSV_FILES,
    "mix_audio": URL_DIR_TO_SAVE_MIX_AUDIO_CSV_FILES
}
'''
'''
CSV_DIR_SRC = {
    "real_audio": REAL_AUDIO_DIR,
    "fake_audio": FAKE_AUDIO_DIR
}
'''

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
