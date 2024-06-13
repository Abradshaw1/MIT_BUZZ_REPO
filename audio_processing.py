# audio_processing.py
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score
from datetime import datetime
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file_paths_and_labels(base_dir):
    file_paths = []
    labels = []

    datasets = [
        ('negative_dataset_1sec', 0),
        ('positive_dataset_1sec', 1)
    ]

    for dataset, label in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.mp3'):
                    file_paths.append(file_path)
                    labels.append(label)

    return file_paths, labels

def load_and_resample_audio(file_path, target_sr=48000):
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        y = np.require(y, requirements=['W'])
        return y, sr
    except Exception as e:
        print(f"Error loading or processing audio: {e}")
        return None, None

def extract_mfcc(y, sr, n_mfcc=30, n_mels=128, fmin=150, fmax=8000, pre_emphasis_coeff=0.97, frame_length=0.025, frame_stride=0.01, n_fft=2048):
    try:
        y_preemphasized = np.append(y[0], y[1:] - pre_emphasis_coeff * y[:-1])
        frame_length_samples = int(frame_length * sr)
        frame_stride_samples = int(frame_stride * sr)
        frames = librosa.util.frame(y_preemphasized, frame_length=frame_length_samples, hop_length=frame_stride_samples).T
        frames = np.require(frames, requirements=['W'])
        frames *= np.hamming(frame_length_samples)
        mel_spectrogram = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=n_fft, hop_length=frame_stride_samples, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfcc = librosa.feature.mfcc(S=mel_spectrogram_db, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
        
        return np.require(mfcc.T, requirements=['W'])
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

class MFCCDataset(Dataset):
    def __init__(self, file_paths, labels, n_mfcc=30):
        self.file_paths = file_paths
        self.labels = labels
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        y, sr = load_and_resample_audio(file_path)
        if y is None or sr is None:
            return None, None
        mfcc_features = extract_mfcc(y, sr, n_mfcc=self.n_mfcc)
        if mfcc_features is None:
            return None, None
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).transpose(0, 1).to(device)
        label_tensor = torch.tensor(label, dtype=torch.float32).to(device)
        return mfcc_tensor, label_tensor
