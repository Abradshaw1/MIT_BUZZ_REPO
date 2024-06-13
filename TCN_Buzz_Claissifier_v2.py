#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_tcn import TCN
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#for saving and loading already trained model
# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the model
# def load_model(model, path):
#     model.load_state_dict(torch.load(path))
#     model.to(device)


# Load audio file paths and labels
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

base_dir = r'/home/gridsan/abradshaw/MIT Buzz'

# Get file paths and labels
file_paths, labels = get_file_paths_and_labels(base_dir)

# Collect first five files for each label category
negative_files = [(path, label) for path, label in zip(file_paths, labels) if label == 0][:5]
positive_files = [(path, label) for path, label in zip(file_paths, labels) if label == 1][:5]

print("First five files in negative dataset (label 0):")
for file_path, label in negative_files:
    print(f"File: {file_path}, Label: {label}")

print("\nFirst five files in positive dataset (label 1):")
for file_path, label in positive_files:
    print(f"File: {file_path}, Label: {label}")

print(f"Found {len(file_paths)} audio files.")


# In[2]:


def load_and_resample_audio(file_path, target_sr=48000):
    """
    Load and resample audio files, then convert stereo to mono by averaging channels.
    There are more complex methods here like channel summing, selecting a single channel or combining but this can be explored later
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=False)  # Load as stereo
        if y.ndim > 1:
            y = np.mean(y, axis=0)  # Convert stereo to mono by averaging the channels
        y = np.require(y, requirements=['W'])  # Ensure the array is writable    
        return y, sr
    except Exception as e:
        print(f"Error loading or processing audio: {e}")
        return None, None
#consider changing window length to 50ms and stride length to 25ms(1200samples, and 40 frames), right now I am at  15ms overlap and (720 sample and 98 frames)
def extract_mfcc(y, sr, n_mfcc=30, n_mels=128, fmin=150, fmax=8000, pre_emphasis_coeff=0.97, frame_length=0.025, frame_stride=0.01, n_fft=2048):
    """
    extract mfccs, all of these parameters can be changed, PEC 0.97 chosen for 48khz data, FL, FS are common and NFFT is good reoultion medium for 48kz
    There are more complex methods here like channel summing, selecting a single channel, or combining but this can be explored later
    possible finetunning: n_mfcc values (20, 30, 40) and n_mels (64, 128, 256), 
    """
    try:
        y_preemphasized = np.append(y[0], y[1:] - pre_emphasis_coeff * y[:-1])   # Pre-emphasis to amplify frequencies    
        frame_length_samples = int(frame_length * sr) #0.025 * 4800=1200 consider changing
        frame_stride_samples = int(frame_stride * sr)#0.01 * 4800=480 consider changing
        frames = librosa.util.frame(y_preemphasized, frame_length=frame_length_samples, hop_length=frame_stride_samples).T
        frames = np.require(frames, requirements=['W']) # Ensure frames are writable
        frames *= np.hamming(frame_length_samples) # Apply a Hamming window to each frame
        mel_spectrogram = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=n_fft, hop_length=frame_stride_samples, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) # Compute Mel spectrogram
        mfcc = librosa.feature.mfcc(S=mel_spectrogram_db, sr=sr, n_mfcc=n_mfcc) # Compute MFCCs from the Mel spectrogram
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6) # Normalize MFCCs to have zero mean and unit variance common for stable models and normailzation
        
        return np.require(mfcc.T, requirements=['W'])
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None


file_paths, labels = get_file_paths_and_labels(base_dir)

# Iterate through all file paths and process each file
for file_path in file_paths:
    # Convert stereo to mono and load audio
    y, sr = load_and_resample_audio(file_path)

    # Extract MFCCs
    if y is not None:
        mfcc_features = extract_mfcc(y, sr)
        if mfcc_features is not None:
            print(f"Extracted MFCCs for file {file_path}. Shape: {mfcc_features.shape}")
        else:
            print(f"Failed to extract MFCCs for file {file_path}.")
    else:
        print(f"Failed to load audio for file {file_path}.")


# In[3]:


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
        # Transpose MFCC features to [num_channels, sequence_length]
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).transpose(0, 1).to(device)
        label_tensor = torch.tensor(label, dtype=torch.float32).to(device)
        return mfcc_tensor, label_tensor

print("Creating dataset...")
dataset = MFCCDataset(file_paths, labels)
print("Dataset created")


# In[4]:


# Split the data into training, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.20, random_state=42, stratify=labels)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.25, random_state=42, stratify=temp_labels) # 0.25 * 0.20 = 0.05

# Create datasets
def create_dataloader(file_paths, labels, n_mfcc=30, batch_size=32, shuffle=True, num_workers=4):
    dataset = MFCCDataset(file_paths, labels, n_mfcc=n_mfcc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

train_loader = create_dataloader(train_files, train_labels)
val_loader = create_dataloader(val_files, val_labels, shuffle=False)
test_loader = create_dataloader(test_files, test_labels, shuffle=False)


# In[5]:


class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, causal, use_norm, activation):
        super(TCNModel, self).__init__()
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            use_norm=use_norm,
            activation=activation
        )
        self.bn = nn.BatchNorm1d(num_channels[-1]) #normalize the activations across the batch for the last channel.
        self.fc = nn.Linear(num_channels[-1], 1) #maps the output of the TCN to a single output neuron
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu') #Initializes the weights of the fully connected layer using Kaiming normalization

    def forward(self, x):
        x = self.tcn(x)
        x = self.bn(x)
        x = torch.mean(x, dim=2)  # Mean pooling instead of adaptive_avg_pool1d, mean pooling across time dimension reducing sequence length to a single value per feature channel
        x = self.fc(x)
        return x

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCNModel(
    num_inputs=30,  #higer better for quick more noise buzz soung compared to speech
    num_channels=[16, 32, 64, 128], #gradual is  important to avoid  gradient explosion
    kernel_size=5, #large enough to capture the context but not too large to lose detailed information
    dropout=0.4,#regularization mabye kick this down
    causal=False, #amke true for real-time cases
    use_norm='weight_norm',
    activation='relu'
).to(device)

criterion = nn.BCEWithLogitsLoss()  # Binary classification, sigmoid layer and binary cross-entropy loss in a single class.
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[6]:


def train(model, dataloader, criterion, optimizer, device, num_epochs=20, clip_value=1.0):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for mfcc_batch, labels_batch in dataloader:
            if mfcc_batch is None:
                continue
            mfcc_batch, labels_batch = mfcc_batch.to(device), labels_batch.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            outputs = model(mfcc_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")

model_save_path = os.path.join(base_dir, 'saved_tcn_model.pth')
save_model(model, 'saved_tcn_model.pth') #CHANGE MODEL PATH


# In[7]:


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for mfcc_batch, labels_batch in dataloader:
            if mfcc_batch is None:
                continue
            mfcc_batch, labels_batch = mfcc_batch.to(device), labels_batch.unsqueeze(-1).to(device)
            outputs = model(mfcc_batch)
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Acc: {accuracy:.4f}")
    return avg_loss, accuracy


# In[8]:


def manual_validation(file_paths, labels, preds, base_dir, num_samples=100):
    # Separate correct predictions, false positives, and false negatives
    correct_files = []
    false_positives = []
    false_negatives = []

    #create folder and file to print results
    output_dir = os.path.join(base_dir, 'manual_validation_results')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'validation_results.txt')
    
    for i, (label, pred) in enumerate(zip(labels, preds)):
        file_path = file_paths[i]
        if label == pred:
            correct_files.append((file_path, pred, label))
        elif label == 0 and pred == 1:
            false_positives.append((file_path, pred, label))
        elif label == 1 and pred == 0:
            false_negatives.append((file_path, pred, label))
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    
    with open(output_file_path, 'w') as f:
        f.write(f"Cross-validation Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}\n")
        f.write(f"Total Correct: {len(correct_files)}\n")
        f.write(f"Total False Positives: {len(false_positives)}\n")
        f.write(f"Total False Negatives: {len(false_negatives)}\n\n")
        
        sample_correct = random.sample(correct_files, min(len(correct_files), num_samples))
        sample_false_positives = random.sample(false_positives, min(len(false_positives), num_samples))
        sample_false_negatives = random.sample(false_negatives, min(len(false_negatives), num_samples))
        
        f.write("Sampled Correct Predictions:\n")
        for fp, pred, label in sample_correct:
            f.write(f"File: {fp}, Prediction: {pred}, True Label: {label}\n")
        
        f.write("\nSampled False Positives:\n")
        for fp, pred, label in sample_false_positives:
            f.write(f"File: {fp}, Prediction: {pred}, True Label: {label}\n")
        
        f.write("\nSampled False Negatives:\n")
        for fp, pred, label in sample_false_negatives:
            f.write(f"File: {fp}, Prediction: {pred}, True Label: {label}\n")

    print(f"Validation results saved to {output_file_path}")


# In[ ]:


# Run
num_epochs = 20

# Training the model
train(model, train_loader, criterion, optimizer, device, num_epochs)

# Evaluating the model on validation set
evaluate(model, val_loader, criterion, device)

# Load the trained model weights
# model_save_path = os.path.join(base_dir, 'saved_tcn_model.pth')
# load_model(model, 'tcn_model.pth')

# Evaluating the model on test set
test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)

# Manual validation
manual_validation(test_loader.dataset.file_paths, test_labels, test_preds, base_dir, num_samples=100)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# In[1]:


# #Run
# num_epochs = 20

# # Training the model
# train(model, train_loader, criterion, optimizer, device, num_epochs)

# # Evaluating the model on validation set
# evaluate(model, val_loader, criterion, device)

# # Evaluating the model on test set
# test_loss, test_acc = evaluate(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# In[ ]:




