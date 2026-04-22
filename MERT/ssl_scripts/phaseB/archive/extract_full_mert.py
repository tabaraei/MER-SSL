import torch
import librosa
import os
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

# 1. Load Model with hidden states enabled
model_name = "m-a-p/MERT-v1-330M" # or "m-a-p/MERT-v1-95M"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
SAVE_PATH = "pmemo_mert_all_layers.pt"

full_layer_data = {}

print(f"Extracting all hidden layers using {model_name}...")

with torch.no_grad():
    for file in tqdm(os.listdir(AUDIO_DIR)):
        if file.endswith(".mp3"):
            music_id = file.split(".")[0]
            path = os.path.join(AUDIO_DIR, file)
            
            # Load and process
            y, sr = librosa.load(path, sr=processor.sampling_rate)
            inputs = processor(y, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            
            # outputs.hidden_states is a tuple of 13 or 25 tensors 
            # (Input Embedding + N Transformer Layers)
            # We take the mean across the time dimension for each layer
            # Resulting shape per song: [Num_Layers, 1024]
            layers = torch.stack(outputs.hidden_states) # [Layers, Batch, Time, Hidden]
            layer_means = layers.mean(dim=2).squeeze(1).cpu().numpy() # [Layers, Hidden]
            
            full_layer_data[music_id] = layer_means

torch.save(full_layer_data, SAVE_PATH)
print(f"Saved full-stack embeddings to {SAVE_PATH}")