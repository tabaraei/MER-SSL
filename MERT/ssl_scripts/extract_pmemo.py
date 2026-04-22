import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa
import os
from tqdm import tqdm

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
OUTPUT_FILE = "pmemo_mert_embeddings.pt"
MODEL_NAME = "m-a-p/MERT-v1-330M"

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Model and Processor
    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    model.eval()

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    results = {}

    # 2. Process Files
    print(f"Processing {len(audio_files)} files...")
    with torch.no_grad():
        for filename in tqdm(audio_files):
            try:
                path = os.path.join(AUDIO_DIR, filename)
                # MERT requires 24000Hz sampling rate
                audio, sr = librosa.load(path, sr=24000)
                
                # Prepare input
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(DEVICE)
                
                # Get model output (hidden states)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract the last hidden state and mean pool over the time dimension
                # music_id is the filename without .mp3
                music_id = filename.split('.')[0]
                embedding = outputs.last_hidden_state.mean(dim=1).cpu()
                
                results[music_id] = embedding
                
            except Exception as e:
                print(f"Error with {filename}: {e}")

    # 3. Save the results
    torch.save(results, OUTPUT_FILE)
    print(f"\nDone! Saved {len(results)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
