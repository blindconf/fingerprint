import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T

def generate_spectrogram(audio_file, output_filename, transform_type="mel", target_sample_rate=24000):
    """
    Loads an audio file, applies a spectrogram transformation (STFT, Mel, or LFCC), 
    and saves the spectrogram as an image with correctly labeled frequency bins.

    Args:
        audio_file (str): Path to the audio file.
        output_filename (str): Path to save the spectrogram image.
        transform_type (str): Type of spectrogram ("mel", "lfcc", "stft"). Default is "mel".
        target_sample_rate (int): Target sample rate for resampling. Default is 24000 Hz.
    """

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Define the Spectrogram Transformation
    if transform_type == "mel":
        n_fft = 2048
        transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=300,
            win_length=1200,
            n_mels=80,
            f_min=0,
            f_max=12000,
            window_fn=torch.hamming_window
        )
    elif transform_type == "lfcc":
        n_fft = 512
        transform = T.LFCC(
            n_filter=20,
            n_lfcc=60,
            speckwargs={
                "n_fft": n_fft,
                "win_length": int(0.025 * target_sample_rate),
                "hop_length": int(0.01 * target_sample_rate)
            }
        )
    elif transform_type == "stft":
        n_fft = 2048
        transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=300,
            win_length=1200,
            power=2,
            window_fn=torch.hamming_window
        )
    else:
        raise ValueError("Invalid transform_type. Choose from 'mel', 'lfcc', or 'stft'.")

    # Apply the transformation
    spectrogram = transform(waveform)

    # Convert to Log Scale (for better visualization)
    spectrogram = torch.log1p(spectrogram)

    # Convert tensor to NumPy for plotting
    spectrogram_np = spectrogram[0].detach().cpu().numpy()

    # Compute actual frequency values for y-axis labels
    num_bins = spectrogram_np.shape[0]  # Number of frequency bins
    freq_bins = np.linspace(0, target_sample_rate / 2, num_bins)  # Frequency values (Hz)

    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='viridis')

    # Set frequency labels
    plt.yticks(
        np.linspace(0, num_bins - 1, num=6),  # Choose 6 evenly spaced ticks
        labels=np.round(np.linspace(0, target_sample_rate / 2, num=6), 1)  # Convert to Hz
    )

    plt.colorbar(label="Log Power")
    plt.title(f"{transform_type.upper()} Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency (Hz)")  # Label in Hz
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

# Example Usage
audio_file = "/USERSPACE/DATASETS/WaveFake/ljspeech_full_band_melgan/LJ001-0096.wav"
output_filename = "spectrogram.png"
generate_spectrogram(audio_file, output_filename, transform_type="stft")  # Use "stft", "mel", or "lfcc"
print("Spectrogram saved!")
