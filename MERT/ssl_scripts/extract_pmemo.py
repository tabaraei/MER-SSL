"""
extract_pmemo.py — MERT all-layer embedding extractor
======================================================
Generalized: works for ANY audio folder (flat or nested) via CLI args.
No-arg invocation reproduces the original PMEmo MERT file exactly.

# PMEmo (default, unchanged):
# python extract_pmemo.py

# IADS-E (MERT):
# python extract_pmemo.py --audio_dir "phaseB/_iadse_audio" \
#                         --out_path phaseB/iadse_mert_all_layers.pt --ext wav
"""

import argparse
import glob
import os

import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import librosa
from tqdm import tqdm

# Defaults preserve original behavior
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
DEFAULT_OUT_PATH = "phaseB/pmemo_mert_all_layers.pt"
MODEL_NAME = "m-a-p/MERT-v1-330M"


def find_audio(audio_dir: str, ext: str):
    """Recursive discovery so nested datasets (IADS-E categories) work too.
    For a flat dir (PMEmo) this yields exactly the original file set."""
    files = sorted(glob.glob(os.path.join(audio_dir, "**", f"*.{ext}"), recursive=True))
    return files


def main():
    ap = argparse.ArgumentParser(description="MERT all-layer extractor")
    ap.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    ap.add_argument("--ext", default="mp3")
    ap.add_argument("--sr", type=int, default=24000)  # MERT needs 24kHz
    args = ap.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    model.eval()

    audio_files = find_audio(args.audio_dir, args.ext)
    print(f"Processing {len(audio_files)} files from {args.audio_dir} (*.{args.ext})...")
    results = {}

    with torch.no_grad():
        for path in tqdm(audio_files):
            filename = os.path.basename(path)
            try:
                audio, sr = librosa.load(path, sr=args.sr)
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs, output_hidden_states=True)

                # All 25 hidden states → (25, 1024), mean-pooled over time
                music_id = filename.rsplit(".", 1)[0]
                layer_embeds = [h.mean(dim=1).squeeze(0).cpu() for h in outputs.hidden_states]
                results[music_id] = torch.stack(layer_embeds)  # (25, 1024)

            except Exception as e:
                print(f"Error with {filename}: {e}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(results, args.out_path)
    print(f"\nDone! Saved {len(results)} embeddings to {args.out_path}")


if __name__ == "__main__":
    main()
