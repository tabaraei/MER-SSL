import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Setup Device & Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings_dict = torch.load('../pmemo_mert_embeddings.pt', weights_only=True)
labels_df = pd.read_csv('/datasets/emotions/PMEmo2019/annotations/static_annotations.csv')

X, y = [], []
for music_id, emb in embeddings_dict.items():
    row = labels_df[labels_df['musicId'] == int(music_id)]
    if not row.empty:
        X.append(emb.squeeze().numpy())
        # Target: [Arousal(mean), Valence(mean)]
        y.append([row['Arousal(mean)'].values[0], row['Valence(mean)'].values[0]])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 2. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 3. Define the MER "Head"
class MERRegressor(nn.Module):
    def __init__(self):
        super(MERRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Crucial for "Low-Label" generalization
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: [Arousal, Valence]
        )
    
    def forward(self, x):
        return self.net(x)

model = MERRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# 4. Training Loop
print(f"Training MER Regressor on {device}...")
for epoch in range(100):
    model.train()
    for b_x, b_y in train_loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(b_x), b_y)
        loss.backward()
        optimizer.step()

# 5. Evaluation with Publishable Metrics
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.numpy()
    
    r2_arousal = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_valence = r2_score(y_true[:, 1], y_pred[:, 1])
    
    print(f"\n--- MER Results (Phase B Baseline) ---")
    print(f"Arousal R2: {r2_arousal:.4f}")
    print(f"Valence R2: {r2_valence:.4f}")
    
torch.save(model.state_dict(), "mer_model_v1.pth")
