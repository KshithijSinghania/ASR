import librosa
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

def extract_logmel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


import os

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

CMVN_MEAN = np.load(ASSETS_DIR / "mel_mean.npy")
CMVN_STD = np.load(ASSETS_DIR / "mel_std.npy")


def extract_logmel_normalized(audio_path):
    log_mel = extract_logmel(audio_path)
    return (log_mel - CMVN_MEAN[:, None]) / CMVN_STD[:, None]

