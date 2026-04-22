import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load full data
data = torch.load("pmemo_mert_all_layers.pt")
labels_df = pd.read_csv('/datasets/emotions/PMEmo2019/annotations/static_annotations.csv')

def train_on_layer(layer_idx):
    X, y = [], []
    for m_id, layers in data.items():
        row = labels_df[labels_df['musicId'] == int(m_id)]
        if not row.empty:
            X.append(layers[layer_idx]) # Extract specific layer
            y.append([row['Arousal(mean)'].values[0], row['Valence(mean)'].values[0]])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple Linear Probe to test "Raw Information"
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    return r2_score(y_test[:, 0], preds[:, 0]), r2_score(y_test[:, 1], preds[:, 1])

arousal_scores, valence_scores = [], []
num_layers = next(iter(data.values())).shape[0]

for i in range(num_layers):
    a_r2, v_r2 = train_on_layer(i)
    arousal_scores.append(a_r2)
    valence_scores.append(v_r2)
    print(f"Layer {i} | Arousal R2: {a_r2:.3f} | Valence R2: {v_r2:.3f}")

# Plotting the "Performance Topography"
plt.figure(figsize=(10, 6))
plt.plot(range(num_layers), arousal_scores, label='Arousal (Energy)', marker='o')
plt.plot(range(num_layers), valence_scores, label='Valence (Mood)', marker='s')
plt.title("MERT Layer-Wise Emotion Encoding (PMEmo2019)")
plt.xlabel("Transformer Layer Index")
plt.ylabel("Linear Probing R² Score")
plt.legend()
plt.grid(True)
plt.savefig("layer_performance.png")
plt.show()