import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa
import os
from tqdm import tqdm

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
OUTPUT_FILE = "phaseB/pmemo_wav2vec_all_layers.pt"
MODEL_NAME = "facebook/wav2vec2-base"  # public — no HF auth required

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Model and Processor
    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    results = {}

    # 2. Process Files
    print(f"Processing {len(audio_files)} files...")
    with torch.no_grad():
        for filename in tqdm(audio_files):
            try:
                path = os.path.join(AUDIO_DIR, filename)
                # wav2vec2 requires 16000Hz sampling rate
                audio, sr = librosa.load(path, sr=16000)

                # Prepare input
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(DEVICE)

                # Get model output (all hidden states)
                outputs = model(**inputs, output_hidden_states=True)

                # hidden_states: tuple of 13 tensors (CNN feature projection + 12
                # transformer layers), each (1, T, 768). Mean-pool over time,
                # stack to (13, 768).
                music_id = filename.split('.')[0]
                layer_embeds = [h.mean(dim=1).squeeze(0).cpu() for h in outputs.hidden_states]
                embedding = torch.stack(layer_embeds)  # (13, 768)

                results[music_id] = embedding

            except Exception as e:
                print(f"Error with {filename}: {e}")

    # 3. Save the results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(results, OUTPUT_FILE)
    print(f"\nDone! Saved {len(results)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
