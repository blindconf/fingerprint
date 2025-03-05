
    fe_transformation = torchaudio.transforms.Spectrogram(n_fft=512,
                                                        hop_length=64
                                                        ).to(device)
                                                        
    mel_spectrogram = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=1024,
                win_length=None,
                hop_length=512,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                n_mels=128,
                mel_scale="htk",
            )
    mfcc_transform = T.MFCC(
                sample_rate=sr,
                n_mfcc=256,
                melkwargs={
                    "n_fft": 2048,
                    "n_mels": 256,
                    "hop_length": 512,
                    "mel_scale": "htk",
                },
            )
    lfcc_transform = T.LFCC(
                sample_rate=sr,
                n_lfcc=256,
                speckwargs={
                    "n_fft": 2048,
                    "win_length": None,
                    "hop_length": 512,
                },
            )

