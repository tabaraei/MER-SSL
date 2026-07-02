"""
extract_pmemo_melspec.py — Mel-spectrogram pre-extractor (third branch)
=======================================================================
Mirrors extract_pmemo.py / extract_pmemo_wav2vec.py. The mel-spectrogram
transform has NO learnable parameters, so pre-extracting it offline is
functionally identical to computing it on-the-fly — it just lets the
TripleSSLModel reuse the exact same TensorDataset(.pt) pipeline as the
frozen SSL encoders. Only the CNN that consumes these spectrograms is
trainable (see models_triple.MelSpectrogramCNN).

Variable-length handling: PMEmo chorus clips range 13–77 s. Each clip is
reduced to a fixed CENTER 30 s window (zero-pad if shorter, symmetric
center-crop if longer) so all spectrograms stack into one tensor.

# PMEmo (default):
# python extract_pmemo_melspec.py
#
# Any folder:
# python extract_pmemo_melspec.py --audio_dir <dir> \
#        --out_path phaseB/pmemo_melspec.pt --ext mp3
"""

import argparse
import glob
import os

import torch
import torchaudio
import librosa
from tqdm import tqdm

DEFAULT_AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
DEFAULT_OUT_PATH = "phaseB/pmemo_melspec.pt"

SR = 24000          # PMEmo native rate (same as MERT)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
WINDOW_SEC = 30
TARGET_SAMPLES = WINDOW_SEC * SR  # 720_000


def find_audio(audio_dir: str, ext: str):
    """Recursive discovery (flat PMEmo dir → original file set)."""
    return sorted(glob.glob(os.path.join(audio_dir, "**", f"*.{ext}"), recursive=True))


def center_window(wave: torch.Tensor, target: int = TARGET_SAMPLES) -> torch.Tensor:
    """(samples,) → fixed-length (target,): zero-pad short, center-crop long."""
    n = wave.shape[-1]
    if n == target:
        return wave
    if n < target:
        total = target - n
        left = total // 2
        right = total - left
        return torch.nn.functional.pad(wave, (left, right))
    start = (n - target) // 2
    return wave[start:start + target]


def main():
    ap = argparse.ArgumentParser(description="Mel-spectrogram center-30s extractor")
    ap.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--out_path", default=DEFAULT_OUT_PATH)
    ap.add_argument("--ext", default="mp3")
    ap.add_argument("--sr", type=int, default=SR)
    args = ap.parse_args()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    audio_files = find_audio(args.audio_dir, args.ext)
    print(f"Processing {len(audio_files)} files from {args.audio_dir} (*.{args.ext})...")
    print(f"Fixed center window: {WINDOW_SEC}s → {TARGET_SAMPLES} samples @ {args.sr}Hz")

    results = {}
    with torch.no_grad():
        for path in tqdm(audio_files):
            filename = os.path.basename(path)
            try:
                audio, _ = librosa.load(path, sr=args.sr, mono=True)
                wave = torch.from_numpy(audio).float()
                wave = center_window(wave)
                spec = to_db(mel(wave))               # (128, T)
                music_id = filename.rsplit(".", 1)[0]
                results[music_id] = spec.contiguous()
            except Exception as e:
                print(f"Error with {filename}: {e}")

    if results:
        any_shape = next(iter(results.values())).shape
        print(f"Spectrogram shape (all): {tuple(any_shape)}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(results, args.out_path)
    print(f"\nDone! Saved {len(results)} mel-spectrograms to {args.out_path}")


if __name__ == "__main__":
    main()
