"""
eda_loader.py — EDA Feature Extraction for PMEmo
=================================================
Loads per-song EDA (electrodermal activity) CSV files and extracts a
7-dimensional statistical feature vector per song:

    [mean_eda, std_eda, slope, n_peaks, mean_peak_amplitude, max_eda, high_ratio]

Features are min-max normalized across the dataset before return.
PMEmo EDA files are expected as: {music_id}_EDA.csv
"""

import os
import numpy as np
import pandas as pd


def load_eda_for_ids(music_ids, eda_dir, sr=200):
    """
    Args:
        music_ids : list of str — PMEmo music IDs in index order
        eda_dir   : path to folder containing {id}_EDA.csv files
        sr        : EDA sampling rate (PMEmo default: 200 Hz)

    Returns:
        (N, 7) float32 numpy array, normalized across the dataset
    """
    from scipy.signal import find_peaks

    def extract(eda):
        mean_eda = np.mean(eda)
        std_eda  = np.std(eda)
        t        = np.arange(len(eda)) / sr
        slope    = np.polyfit(t, eda, 1)[0] if len(eda) > 1 else 0.0
        peaks, props = find_peaks(eda, prominence=0.01, distance=sr)
        n_peaks  = len(peaks)
        mean_amp = props["prominences"].mean() if n_peaks > 0 else 0.0
        return np.array([mean_eda, std_eda, slope, n_peaks, mean_amp,
                         np.max(eda), np.mean(eda > mean_eda)], dtype=np.float32)

    features, missing = [], 0
    for mid in music_ids:
        path = os.path.join(eda_dir, f"{mid}_EDA.csv")
        if not os.path.exists(path):
            features.append(np.zeros(7, dtype=np.float32))
            missing += 1
            continue
        df  = pd.read_csv(path, header=None)
        eda = df.select_dtypes(include=[np.number]).iloc[:, 1:].mean(axis=1).values.astype(np.float32)
        features.append(extract(eda))

    arr    = np.stack(features)
    mn, mx = arr.min(0, keepdims=True), arr.max(0, keepdims=True)
    arr    = (arr - mn) / np.where(mx - mn < 1e-6, 1.0, mx - mn)

    if missing:
        print(f"  ⚠️  EDA missing for {missing}/{len(music_ids)} songs — zeros used")
    else:
        print(f"  ✅ EDA loaded for all {len(music_ids)} songs")

    return arr.astype(np.float32)
