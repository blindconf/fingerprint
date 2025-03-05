from speechbrain.lobes.augment import EnvCorrupt
import os
import glob
import torchaudio
import torch
import csv
'''
"ljspeech_fast_diff_tacotron": "ljspeech_fast_diff_tacotron_test.csv",
"ljspeech_hnsf": "ljspeech_hnsf_test.csv",
"ljspeech_pro_diff": "ljspeech_pro_diff_test.csv",
"ljspeech_melgan": "ljspeech_melgan_test.csv",
"ljspeech_parallel_wavegan": "ljspeech_parallel_wavegan_test.csv",
"ljspeech_multi_band_melgan": "ljspeech_multi_band_melgan_test.csv",
"ljspeech_melgan_large": "ljspeech_melgan_large_test.csv",
"ljspeech_full_band_melgan": "ljspeech_full_band_melgan_test.csv",
"ljspeech_hifiGAN": "ljspeech_hifiGAN_test.csv",
"ljspeech_waveglow": "ljspeech_waveglow_test.csv",
"ljspeech_avocodo": "ljspeech_avocodo_test.csv",
"ljspeech_bigvgan": "ljspeech_bigvgan_test.csv",
"ljspeech_lbigvgan": "ljspeech_lbigvgan_test.csv",
"LJSpeech": "LJSpeech_test.csv",
"jsut_multi_band_melgan": "JSUT_multi_band_melgan_test.csv",
"jsut_parallel_wavegan": "JSUT_parallel_wavegan_test.csv",
"jSut": "JSUT_test.csv"
'''
# 
if __name__ == "__main__":
    # '''
    datasets = {
        "jsut_hnsf": "jsut_hnsf_test.csv",

                }

    high_snr = 0 # 15
    low_snr = 0
    # print(os.path.abspath(EnvCorrupt.__file__))
    corrupter = EnvCorrupt(
    openrir_folder =  "/USERSPACE/DATASETS/LibriSpeech/",
    babble_prob= 0.0, 
    reverb_prob= 0.0, 
    noise_prob= 1.0, 
    noise_snr_low= low_snr, 
    noise_snr_high= high_snr
    ).to('cuda')
    CSV_PATH = "/USERSPACE/pizarm5k/audio_fingerprint/audio-fingerprint/csv_files/jsut/1/test"
    
    for dataset_type in datasets.keys():
        output_path = f"/USERSPACE/DATASETS/Encodec/wavefake_noise/high_snr_{low_snr}_{high_snr}/{dataset_type}"
        print(output_path)
        os.makedirs(output_path, exist_ok=True)
        csv_file_path = f"{CSV_PATH}/{datasets[dataset_type]}"
        print(csv_file_path)

        # Open the CSV file and read its contents
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            
            # Loop through each row in the CSV file
            for row in reader:
                # Assuming each row contains a single directory path
                directory_path = row[0]
                name = os.path.splitext(os.path.basename(directory_path))[0] 
                if not os.path.isfile(f"{output_path}/{name}.wav"):
                    # print("error!: ", j)
                    wav, sr = torchaudio.load(directory_path)
                    wav = wav.to('cuda')
                    wav_len = torch.tensor([1]).to('cuda')
                    wavs_noise = corrupter(wav, wav_len)
                    # print(f"{output_path}/{name}.wav")
                    torchaudio.save(f"{output_path}/{name}.wav", wavs_noise.cpu(), sample_rate=sr)
                else:
                    pass
                    # print(j)
    # '''
    '''
    import matplotlib.pyplot as plt

    # Data
    snr = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Using 45 to plot "No Noise" at the end of the x-axis
    auroc_mahalanobis = [0.678, 0.715, 0.758, 0.805, 0.847, 0.886, 0.921, 0.949, 0.970, 0.998]
    auroc_correlation = [0.683, 0.722, 0.766, 0.811, 0.853, 0.892, 0.924, 0.949, 0.966, 0.983]

    # Plotting
    fig_width = 4.5  # in inches, suitable for one column
    fig_height = 2.0  # in inches, suitable for one column
    plt.figure(figsize=(fig_width, fig_height))

    plt.plot(snr, auroc_mahalanobis, marker='o', linestyle='-', color="#2D5B68", label='Low-pass filter', markersize=4, linewidth=1)
    plt.plot(snr, auroc_correlation, marker='o', linestyle='-', color="#D2A497", label='EnCodec filter', markersize=4, linewidth=1)

    # Customizing the plot
    plt.xlabel('SNR (dB)', size=9)
    plt.ylabel('Average AUROC', size=9)
    plt.ylim(0.5, 1.02)

    # Adjust x-ticks and labels
    x_labels = [0, 5, 10, 15, 20, 25, 30, 35, 40, "No Noise"]
    plt.xticks(ticks=snr, labels=x_labels, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=7)

    # Customize legend
    plt.legend(frameon=False, loc='lower right', prop={'size':9})

    # Display the plot
    plt.tight_layout(pad=1.0)
    plt.savefig("fig_paper_snr.pdf", dpi=300, transparent=True, bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()
    '''

    # plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.5), prop={'size':10})
    # ax = plt.gca()
    # Hide x-axis labels
    # ax.set_xticklabels([])