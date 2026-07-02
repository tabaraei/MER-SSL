from configs.config import PATHS  # centralised config
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the embeddings and labels
print("Loading data...")
data = torch.load(str(PATHS.mert_pooled_embeddings), weights_only=True)
# Adjust this path if your CSV is located elsewhere
labels_df = pd.read_csv('/datasets/emotions/PMEmo2019/annotations/static_annotations.csv')

# 2. Match IDs and create a clean Matrix
ids = []
embeddings = []
valence = []
arousal = []

for music_id, emb in data.items():
    # Find the row in CSV matching the musicId
    row = labels_df[labels_df['musicId'] == int(music_id)]
    if not row.empty:
        ids.append(music_id)
        embeddings.append(emb.squeeze().numpy())
        valence.append(row['Valence(mean)'].values[0])
        arousal.append(row['Arousal(mean)'].values[0])

embeddings = np.array(embeddings)
print(f"Matched {len(embeddings)} songs with labels.")

# 3. Compute t-SNE
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 4. Plotting (Two plots: one for Valence, one for Arousal)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Valence Plot
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
                hue=valence, palette="coolwarm", ax=ax1)
ax1.set_title("MERT Embeddings colored by VALENCE (Blue=Sad, Red=Happy)")

# Arousal Plot
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
                hue=arousal, palette="magma", ax=ax2)
ax2.set_title("MERT Embeddings colored by AROUSAL (Dark=Calm, Bright=Energetic)")

plt.savefig("mert_emotion_clusters.png")
print("New plot saved as mert_emotion_clusters.png")
