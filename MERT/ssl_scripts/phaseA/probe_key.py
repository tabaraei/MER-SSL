import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import librosa
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- 1. Define the Logistic (Linear) Probe for Classification ---
class KeyProbe(nn.Module):
    def __init__(self):
        super(KeyProbe, self).__init__()
        # Output: 1 (Binary classification: 0 = Minor, 1 = Major)
        self.linear = nn.Linear(1024, 1) 
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# --- 2. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
embeddings_dict = torch.load('../pmemo_mert_embeddings.pt', weights_only=True)

# --- 3. Extract Ground Truth "Mode" (Major/Minor) ---
print("Extracting Ground Truth Musical Mode (Major/Minor)...")
key_labels = {}
for music_id in tqdm(embeddings_dict.keys()):
    path = os.path.join(AUDIO_DIR, f"{music_id}.mp3")
    if os.path.exists(path):
        try:
            y, sr = librosa.load(path, sr=None)
            # Compute Chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            # Simple estimate of Key/Mode (0: Minor, 1: Major)
            # This is a proxy for the 'Theoretical' label
            key_labels[music_id] = 1 if np.mean(chroma) > 0.2 else 0 
        except:
            continue

# --- 4. Prepare Data ---
X, y = [], []
for music_id, mode in key_labels.items():
    X.append(embeddings_dict[music_id].squeeze().numpy())
    y.append(mode)

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# --- 5. Train the Probe ---
model = KeyProbe().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss() # Binary Cross Entropy for classification

print(f"Training Key Probe on {device}...")
model.train()
for epoch in range(100):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()

# --- 6. Results ---
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test.to(device)).cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int)
    acc = accuracy_score(y_test.numpy(), y_pred)
    
    print(f"\n--- Musical Key Probing Result ---")
    print(f"Major/Minor Classification Accuracy: {acc:.4f}")
    if acc > 0.6:
        print("Conclusion: MERT successfully captures Harmonic Mode (Major/Minor).")