import pickle
import numpy as np

def read_pickle_spectrogram_data(path):
    """
    Returns a list of triplets (spectrogram, sampling rate, filename)
    :param path:
    :return:
    """
    with open(path, "rb") as f:
        list_of_spectrogram_data = pickle.load(f)

    return list_of_spectrogram_data

