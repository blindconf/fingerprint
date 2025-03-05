from os import listdir
from os.path import isfile, join
import csv
import random
import glob
import os
import torchaudio
from pathlib import Path


onlyfiles = [f for f in glob.glob("/USERSPACE/DATASETS/LJSpeech-1.1/wavs/*.*")]
# print(onlyfiles)
random.Random(4).shuffle(onlyfiles)
train_data = onlyfiles[:int(len(onlyfiles)*0.8)]
test_data = onlyfiles[int(len(onlyfiles)*0.8):]

main_path = '/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/csv_files'
if not os.path.isfile(f"{main_path}/{'LJSpeech_train.csv'}"):
    with open(f"{main_path}/{'LJSpeech_train.csv'}", 'w') as print_to:
        writer = csv.writer(print_to)
        for i in train_data:
            writer.writerow([i])

if not os.path.isfile(f"{main_path}/{'LJSpeech_test.csv'}"):
    with open(f"{main_path}/{'LJSpeech_test.csv'}", 'w') as print_to:
        writer = csv.writer(print_to)
        for i in test_data:
            writer.writerow([i])

flac_files = glob.glob("/USERSPACE/DATASETS/WaveFake/ljspeech_melgan/*.*", recursive=True)
print("Tot audio files {}".format(len(flac_files)))
with open(f"{main_path}/{'sb_ljspeech_melgan.csv'}", 'w') as print_to:
    writer = csv.writer(print_to)
    writer.writerow(["ID", "duration", "wav"])
    for idk, utterance in enumerate(flac_files):
        # print(idk)
        utt_info = torchaudio.info(utterance)
        length = utt_info.num_frames / utt_info.sample_rate
        data = [[idk, length, utterance]]
        # print("length", utt_info.sample_rate, utt_info.num_frames / utt_info.sample_rate)
        writer.writerow(data[0])
    # break
'''
flac_files = glob.glob("/USERSPACE/DATASETS/LJSpeech-1.1/wavs/*.*", recursive=True)
sb_LJSpeech.csv

flac_files = glob.glob("/USERSPACE/DATASETS/WaveFake/ljspeech_melgan/*.*", recursive=True)
sb_ljspeech_melgan.csv

'''

'''
#generated
ljspeech_hifiGAN
# _gen
ljspeech_parallel_wavegan
ljspeech_multi_band_melgan
ljspeech_melgan_large
ljspeech_melgan
ljspeech_full_band_melgan
# same
ljspeech_waveglow
'''