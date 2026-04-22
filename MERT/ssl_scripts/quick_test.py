import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa

# 1. Target the idle GPU (GPU 1)
device = torch.device("cuda:1")
print(f"Using device: {device}")

# 2. Load MERT (this will download weights automatically)
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")

# 3. Create dummy audio (1 second of silence at 24kHz)
audio = torch.randn(1, 24000).to(device)

# 4. Run inference
with torch.no_grad():
    outputs = model(audio, output_hidden_states=True)

print(f"Success! Output shape: {outputs.last_hidden_state.shape}")
