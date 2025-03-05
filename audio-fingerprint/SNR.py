import sys
import torch
import csv
import torchaudio
import numpy as np
import scipy.io.wavfile as wav
import wave
import os 
import re


def newWER(x, y):
    x = x.split()
    y = y.split()
    n = len(x)
    m = len(y)
    k = min(n, m)
    d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)
    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            if (x[i - 1] == y[j - 1]):
                d[i][j] = d[i - 1][j - 1]
            else:
                S = d[i - 1][j - 1] + 1
                I = d[i][j - 1] + 1
                D = d[i - 1][j] + 1
                d[i][j] = min(S, I, D)
    return d[k][k] * 1.0 / k

def WER(x, y):
	x = x.split()
	y = y.split()
	n = len(x)
	m = len(y)
	print(n)
	print(m)
	d = np.zeros((n + 1) * (m + 1), dtype = np.uint8).reshape(n + 1, m + 1)

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	print(d[n][m])
	return d[n][m] * 1.0 / n

def newCER(x, y):
    # Convert the list into a string without spaces
    x = x.replace(" ", "")
    y = y.replace(" ", "")
    n = len(x)
    m = len(y)
    k = min(n, m)
    d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)

    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            if (x[i - 1] == y[j - 1]):
                d[i][j] = d[i - 1][j - 1]
            else:
                S = d[i - 1][j - 1] + 1
                I = d[i][j - 1] + 1
                D = d[i - 1][j] + 1
                d[i][j] = min(S, I, D)
	
    return d[k][k] * 1.0 / k

def CER(x, y):
	x = x.replace(" ", "")
	y = y.replace(" ", "")
	n = len(x)
	m = len(y)
	d = np.zeros((n + 1) * (m + 1), dtype = np.uint8).reshape(n + 1, m + 1)

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			if (x[i - 1] == y[j - 1]):
				d[i][j] = d[i - 1][j - 1]
			else:
				S = d[i - 1][j - 1] + 1
				I = d[i][j - 1] + 1
				D = d[i - 1][j] + 1
				d[i][j] = min(S, I, D)
	
	return d[n][m] * 1.0 / n

def numel(array):
	# Number of elements in an array
    s = array.shape
    n = 1
    for i in range(len(s)):
        n *= s[i]
    return n
    
def snrseg(noisy, clean, fs, tf=0.05):
    '''
    Segmental SNR computation. Does NOT support VAD (voice activity dection) or Interpolation (at the moment). Corresponds to the mode 'wz' in
    the original Matlab implementation.

    SEG = mean(10*log10(sum(Ri^2)/sum((Si-Ri)^2))

    '''
    snmax = 100
    noisy = noisy.squeeze()
    clean = clean.squeeze()
    if clean.shape[0] != noisy.shape[0]:
        print("ERROR SHAPE! ")
    nr = min(clean.shape[0], noisy.shape[0])
    kf = round(tf * fs)
    ifr = np.arange(kf, nr, kf)
    ifl = int(ifr[len(ifr)-1])
    nf = numel(ifr)
    ef = np.sum(np.reshape(np.square((noisy[:ifl] - clean[:ifl]), dtype='float32'), (kf, nf), order='F'), 0)
    rf = np.sum(np.reshape(np.square(clean[:ifl], dtype='float32'), (kf, nf), order='F'), 0)
    em = ef == 0
    rm = rf == 0
    snf = 10 * np.log10((rf + rm) / (ef + em))
    snf[rm] = -snmax
    snf[em] = snmax
    temp = np.ones(nf)
    vf = temp == 1
    seg = np.mean(snf[vf])
    return seg

def read_csv_file(file_path):
    """Read a CSV file and return a list of its rows."""
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]  # Assuming each row has a single file path

def extract_filename(file_path):
    """Extracts the file name without extension and anything after '_'."""
    # Get the file name without the directory path
    base_name = os.path.basename(file_path)
    
    # Remove the extension
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Extract everything before the first underscore
    result = re.split(r'_', name_without_ext)[0]
    
    return result

def modify_path(file_path, insert_folder):
    """Inserts the folder after 'wavefake_noise/' in the given path."""
    # Split the path into head and tail
    head, tail = os.path.split(file_path)
    
    # Check if 'wavefake_noise' is in the path and add the new folder after it
    if 'wavefake_noise' in head:
        modified_path = head.replace('wavefake_noise', f'wavefake_noise/{insert_folder}')
        new_path = os.path.join(modified_path, tail)
        return new_path
    else:
        return file_path  # Return the original if the pattern is not found

def Seg_SNR(source, target, snr_level):
    rdr_src = read_csv_file(source)
    rdr_tgt = read_csv_file(target)
    if len(rdr_src) != len(rdr_tgt):
        print("The files have different numbers of rows.")
        return

    count = 0
    db_neg = 0
    snr_seg, snr_cw, snr_cw_16bits = [], [], []
    for i, (src, tgt) in enumerate(zip(rdr_src, rdr_tgt), start=1):
        # print(extract_filename(src))
        # print(extract_filename(tgt))
        if extract_filename(src) == extract_filename(tgt):    
            # Test with your example path
            file_path = '/USERSPACE/DATASETS/Encodec/wavefake_noise/ljspeech_melgan/LJ030-0129_gen.wav'
            insert_folder = 'high_snr_value_value'

            tgt = modify_path(tgt, f'high_snr_{snr_level}_{snr_level}')
                    
            data_adv, _ = torchaudio.load(tgt)
            data_src, _ = torchaudio.load(src)

            # Calculate the power of the signal and noise
            signal_power = np.mean(data_src[0].numpy() ** 2)
            diff = data_adv[0].numpy() - data_src[0].numpy()
            noise_power = np.mean(diff ** 2)

            # Calculate SNR in decibels (dB)
            snr = 10 * np.log10(signal_power / noise_power)
            snr_cw.append(snr)
        else:
            print("ERROR")
        count += 1
    
    return np.mean(snr_cw), count

if __name__ == "__main__":
    TEST_CSV = [
                    "ljspeech_melgan_test.csv",
                    "ljspeech_parallel_wavegan_test.csv",
                    "ljspeech_multi_band_melgan_test.csv",
                    "ljspeech_melgan_large_test.csv",
                    "ljspeech_full_band_melgan_test.csv",
                    "ljspeech_hifiGAN_test.csv",
                    "ljspeech_waveglow_test.csv",
                    "ljspeech_avocodo_test.csv",
                    "ljspeech_bigvgan_test.csv",
                    "ljspeech_lbigvgan_test.csv",
                    "LJSpeech_test.csv"
                    ]
    snr = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    for test_csv in TEST_CSV:
        print(test_csv)
        for j in snr:    
            clean_speech = 'csv_files/full/'+ test_csv
            noise_speech = 'csv_files/full/noise/' + test_csv 
            # SNR_SEG, C&W SNR:
            SNR, total = Seg_SNR(clean_speech, noise_speech, snr_level=j)
            # print("{}: SNR {:.2f} total samples {}".format(j, SNR, total))
            print(SNR)
