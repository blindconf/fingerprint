# import...
import math
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import torchaudio.transforms as T


def hz_to_mel(x):
    """
    Converts a frequency given in Hz into the corresponding Mel frequency.

    :param x: input frequency in Hz.
    :return: frequency in mel-scale.
    """
    # TODO implement this method
    return 2595 * np.log10(1 + x / 700)


# Task 4.2
def mel_to_hz(x):
    """
    Converts a frequency given in Mel back into the linear frequency domain in Hz.

    :param x: input frequency in mel.
    :return: frequency in Hz.
    """
    # TODO implement this method
    return 700 * (10**(x / 2595) - 1)


# 10 seconds long, x samples, getting the samples
def sec_to_samples(x, sampling_rate):
    """
    Converts continuous time to sample index.

    :param x: scalar value representing a point in time in seconds.
    :param sampling_rate: sampling rate in Hz.
    :return: sample index.
    """
    return round(x * sampling_rate)


def next_pow2(x):
    """
    Returns the next power of two for any given positive number.

    :param x: scalar input number.
    :return: next power of two larger than input number.
    """
    # Exponent of next higher power of 2
    # TODO implement this method
    return 0 if x==0 else math.ceil((math.log(abs(x), 2)))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    Returns the total number of frames for a given signal length with corresponding window and hop sizes.

    :param signal_length_samples: total number of samples.
    :param window_size_samples: window size in samples.
    :param hop_size_samples: hop size (frame shift) in samples.
    :return: total number of frames.
    """
    shared_frame = window_size_samples - hop_size_samples
    return math.ceil((signal_length_samples - shared_frame) / (window_size_samples - shared_frame))


def plot_filterbank(data):    
    x = np.linspace(0, data.shape[1], data.shape[1])
    fig, ax = plt.subplots()  
    ax.set_title('24-filters Mel Filterbank')
    ax.plot(x, data.T)
    # fig.savefig('Filterbank.png')
    plt.show()
    pass

def plotting(frames):
    start = 0
    end = frames.shape[1] * 1 / 16000
    t = np.linspace(start, end, frames.shape[1])    
    # Create two subplots and unpack the output array immediately
    fig, axs = plt.subplots(4)
    fig.subplots_adjust(hspace=.5)
    plt.xlabel('Time in seconds', fontsize=9)

    for i in range(frames.shape[0]):
        y = frames[i, :]
        axs[i].plot(t, y, linewidth=1)
        plt.setp(axs[i].get_xticklabels(), fontsize=7)
        plt.setp(axs[i].get_yticklabels(), fontsize=7)
        axs[i].set_xlim(0, end)
        axs[i].grid()  
    
    axs[0].set_title('First 4 successive frames (normalized between -1 and 1)', fontsize=10)
    plt.show()
    pass


def spectrogram_plot(features_dB, x_axis, y_axis, y_label, save_file):
    fig = plt.figure(1)
    # plt.imshow(np.transpose(features_dB), extent=[0, x_axis, 0, y_axis], origin='lower', aspect='auto')
    plt.imshow(np.transpose(features_dB), origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel('Time in seconds')
    plt.ylabel(y_label)
    fig.savefig(save_file)
    plt.show()
    pass


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

def crop_bounderies(all_data, fe_type='spectrogram', transformation=None, device='cpu'):
    min_num_time_bins = 100000
    max_num_time_bins = 0
    num_frequency_bins = 0

    # delta and deltadeltas
    deltas_function = T.ComputeDeltas()
    for i in all_data:
        spec = transformation(i)
        if fe_type == 'mfccdeltas':
            # add delta and delta deltas
            deltas = deltas_function(spec)
            deltadeltas = deltas_function(deltas)
            spec = torch.cat((spec, deltas, deltadeltas), 1)

        if spec.shape[2] < min_num_time_bins:
            min_num_time_bins = spec.shape[2]
        if spec.shape[2] > max_num_time_bins:
            max_num_time_bins = spec.shape[2]
    num_frequency_bins = spec.shape[1]
    return min_num_time_bins, max_num_time_bins, num_frequency_bins
