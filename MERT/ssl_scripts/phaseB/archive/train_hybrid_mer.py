import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from models import HybridMERModel
from losses import SupCRLoss

# 1. Load Data with the warning fixed
# We set weights_only=False because your file contains a dictionary of numpy arrays
data_dict = torch.load("pmemo_mert_all_layers.pt", weights_only=False)
labels_df = pd.read_csv('/datasets/emotions/PMEmo2019/annotations/static_annotations.csv')

# 2. Data Stacking: Convert Dictionary to Tensors
X_list = []
Y_list = []

print("Stacking features and labels...")
for music_id, layers in data_dict.items():
    # Find matching label in CSV
    row = labels_df[labels_df['musicId'] == int(music_id)]
    if not row.empty:
        # X: [25 layers, 1024 hidden dim]
        X_list.append(torch.from_numpy(layers).float())
        # Y: [Arousal, Valence]
        Y_list.append(torch.tensor([
            row['Arousal(mean)'].values[0], 
            row['Valence(mean)'].values[0]
        ]).float())

# Final tensors: X [N, 25, 1024], Y [N, 2]
X = torch.stack(X_list)
Y = torch.stack(Y_list)

# 3. Create DataLoader
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset ready. Total samples: {len(X)}")

# 4. Setup Model and Dual-Loss Strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridMERModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
mse_criterion = torch.nn.MSELoss()
# SupCRLoss threshold should be tuned based on your Valence/Arousal range
contrastive_criterion = SupCRLoss(threshold=0.15) 

# 5. Training Loop
for epoch in range(50):
    total_epoch_loss = 0
    for batch_layers, batch_labels in train_loader:
        batch_layers, batch_labels = batch_layers.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: Get predictions and the normalized latent space for XAI
        preds, latent = model(batch_layers)
        
        # Calculate losses
        loss_mse = mse_criterion(preds, batch_labels)
        loss_con = contrastive_criterion(latent, batch_labels)
        
        # Total Loss: MSE handles accuracy, Contrastive handles space structure (XAI)
        total_loss = loss_mse + 0.1 * loss_con
        
        total_loss.backward()
        optimizer.step()
        total_epoch_loss += total_loss.item()
        
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Total Loss: {total_epoch_loss/len(train_loader):.4f}")

# Save the final thesis-grade model
torch.save(model.state_dict(), "hybrid_mer_model_v1.pth")