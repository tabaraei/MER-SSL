"""
extract_music_theory.py — Phase A music-theory ground-truth extractor
======================================================================
Extracts music-theoretic features from every PMEmo chorus clip with
librosa and saves them as phaseA/data/pmemo_music_theory.pt
(dict: song_id -> {feature_name: torch.Tensor}).

NOTE: the spec referenced `librosa.estimate_key()`, which does not exist
in any librosa version. Key/mode are estimated with the standard
Krumhansl–Schmuckler key-finding algorithm (Krumhansl & Kessler 1982):
correlate the mean chroma vector against the 24 rotated major/minor key
profiles and take the best match. This is the canonical, citable method.

Mirrors the existing phaseA conventions (probe_key.py / probe_tempo.py):
self-contained script, AUDIO_DIR = PMEmo chorus, librosa.load(sr=None),
tqdm progress, robust per-file try/except.

Run from phaseA/:
    python extract_music_theory.py
"""

import glob
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

AUDIO_DIR = "/datasets/emotions/PMEmo2019/chorus"
OUT_PATH = "data/pmemo_music_theory.pt"

# Krumhansl–Kessler (1982) tonal hierarchy profiles.
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key_mode(chroma_mean: np.ndarray):
    """Krumhansl–Schmuckler key finding.

    Args:
        chroma_mean: (12,) time-averaged chroma vector.
    Returns:
        (key, mode) — key in 0..11 (C..B), mode 1=major / 0=minor.
    """
    c = chroma_mean - chroma_mean.mean()
    if np.allclose(c, 0):
        return 0, 1
    best_key, best_mode, best_corr = 0, 1, -2.0
    for tonic in range(12):
        for profile, mode in ((_KK_MAJOR, 1), (_KK_MINOR, 0)):
            p = np.roll(profile, tonic)
            p = p - p.mean()
            denom = np.linalg.norm(c) * np.linalg.norm(p)
            corr = float(np.dot(c, p) / denom) if denom > 0 else -2.0
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, tonic, mode
    return best_key, best_mode


def _zeros():
    """Zero-filled feature dict (used on extraction failure)."""
    return {
        "chroma":              torch.zeros(12),
        "mode":                torch.tensor(0.0),
        "key":                 torch.tensor(0.0),
        "tempo":               torch.tensor(0.0),
        "rhythmic_stability":  torch.tensor(0.0),
        "spectral_centroid":   torch.tensor(0.0),
        "spectral_contrast":   torch.zeros(7),
        "zcr":                 torch.tensor(0.0),
    }


def extract_one(path: str) -> dict:
    y, sr = librosa.load(path, sr=None)

    chroma = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)        # (12,)
    key, mode = estimate_key_mode(chroma)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo).ravel()[0])

    tg = librosa.feature.tempogram(y=y, sr=sr)
    tg_mean = float(np.mean(tg))
    rhythmic_stability = 1.0 - (float(np.std(tg)) / tg_mean) if tg_mean else 0.0

    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_centroid_norm = cent / (sr / 2.0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)  # (7,)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return {
        "chroma":             torch.tensor(chroma, dtype=torch.float32),
        "mode":               torch.tensor(float(mode)),
        "key":                torch.tensor(float(key)),
        "tempo":              torch.tensor(tempo / 200.0),
        "rhythmic_stability": torch.tensor(float(rhythmic_stability)),
        "spectral_centroid":  torch.tensor(float(spec_centroid_norm)),
        "spectral_contrast":  torch.tensor(contrast, dtype=torch.float32),
        "zcr":                torch.tensor(float(zcr)),
    }


def main():
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    print(f"Extracting music-theory ground truth from {len(files)} clips...")

    results, n_fail = {}, 0
    for path in tqdm(files):
        song_id = os.path.basename(path).rsplit(".", 1)[0]
        try:
            results[song_id] = extract_one(path)
        except Exception as e:
            print(f"  ⚠️  {song_id}: extraction failed ({e}) — filled with zeros")
            results[song_id] = _zeros()
            n_fail += 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(results, OUT_PATH)
    print(f"\nDone! Saved {len(results)} entries to {OUT_PATH} "
          f"({n_fail} zero-filled failures)")


if __name__ == "__main__":
    main()
