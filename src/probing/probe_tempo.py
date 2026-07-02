from configs.config import PATHS  # centralised config
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import librosa
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm

# --- 1. Define the Linear Probe Class ---
# This was missing in your previous run, causing the NameError.
class LinearProbe(nn.Module):
    def __init__(self):
        super(LinearProbe, self).__init__()
        # MERT-v1-95M/330M usually provides 768 or 1024 dims. 
        # Since your previous runs used 1024, we stay with that.
        self.linear = nn.Linear(1024, 1) 
    
    def forward(self, x):
        return self.linear(x)

# --- 2. Setup Device and Data Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
embeddings_dict = torch.load(str(PATHS.mert_pooled_embeddings), weights_only=True)

# --- 3. Extract Ground Truth Tempo ---
print("Extracting Ground Truth Tempo from audio...")
tempo_labels = {}
for music_id in tqdm(embeddings_dict.keys()):
    path = os.path.join(AUDIO_DIR, f"{music_id}.mp3")
    if os.path.exists(path):
        try:
            y_audio, sr = librosa.load(path, sr=None)
            # Calculate BPM - extracting scalar to avoid DeprecationWarning
            tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
            tempo_labels[music_id] = float(np.mean(tempo))
        except Exception as e:
            continue

# --- 4. Prepare and Standardize Data ---
X, y = [], []
for music_id, tempo in tempo_labels.items():
    X.append(embeddings_dict[music_id].squeeze().numpy())
    y.append(tempo)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

# STANDARDIZE THE TARGETS (Crucial for a meaningful R2 score)
scaler = StandardScaler()
y_scaled = torch.tensor(scaler.fit_transform(y.numpy()), dtype=torch.float32)

# Split Scaled Data
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# --- 5. Initialize and Train Model ---
model = LinearProbe().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) 
criterion = nn.MSELoss()

print(f"Training Scaled Probe on {device}...")
model.train()
for epoch in range(200): 
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {epoch_loss/len(train_loader):.4f}")

# --- 6. Results and Evaluation ---
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu()
    r2 = r2_score(y_test.numpy(), y_pred.numpy())
    
    print(f"\n--- Improved Music Theory Probing Result ---")
    print(f"Standardized Tempo R2 Score: {r2:.4f}")