"""
data_utils.py — Final Debugged Version
===========================================
Fixed: ValueError by implementing robust ID matching (handles float/int/string).
Fixed: Return signature to match mainB.py requirements.
Included: All quadrant and multimodal EDA functions.
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# =============================================================================
# 1. Audio Loading (Robust ID Matching)
# =============================================================================

def load_pmemo_data(feature_path: str, label_path: str):
    print(f"  Loading embeddings from: {feature_path}")
    data = torch.load(feature_path, map_location="cpu")

    print(f"  Loading labels from: {label_path}")
    df = pd.read_csv(label_path)
    
    # Identify standard columns
    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]

    # Normalize labels to [0, 1]
    def _norm(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col])
    df[va_col] = _norm(df[va_col])

    # --- MATCHING LOGIC ---
    if isinstance(data, dict):
        # Case A: Standard container dictionary
        feat_key = next((k for k in ["features", "embeddings", "hidden_states"] if k in data), None)
        
        if feat_key:
            print(f"  Standard format detected (key: '{feat_key}')")
            embeddings = data[feat_key]
            m_ids = [str(x) for x in data.get("music_ids", [])]
            id_to_idx = {mid: i for i, mid in enumerate(m_ids)}
            df["_idx"] = df[id_col].astype(str).map(id_to_idx)
            df = df.dropna(subset=["_idx"])
            X = embeddings[df["_idx"].values.astype(int)]
        else:
            # Case B: Dictionary-of-Tensors (Your specific format)
            print("  Detected 'Dictionary-of-Tensors' format. Matching IDs...")
            valid_rows = []
            valid_X = []
            
            # Robust Matcher: Handle cases where ID is 760 (int), 760.0 (float), or '760' (str)
            for _, row in df.iterrows():
                raw_id = row[id_col]
                # Convert to int then str to remove '.0' if it exists
                clean_id = str(int(raw_id)) if isinstance(raw_id, (int, float, np.number)) else str(raw_id)
                
                if clean_id in data:
                    valid_rows.append(row)
                    valid_X.append(torch.as_tensor(data[clean_id]))
            
            if not valid_X:
                sample_keys = list(data.keys())[:5]
                sample_csv = df[id_col].iloc[:5].tolist()
                print(f"  ❌ Debug: CSV IDs look like {sample_csv}")
                print(f"  ❌ Debug: .pt keys look like {sample_keys}")
                raise ValueError("No matching IDs found between CSV and .pt file.")
            
            X = torch.stack(valid_X)
            df = pd.DataFrame(valid_rows)
    else:
        # Case C: Raw Tensor
        n = min(len(data), len(df))
        X = data[:n]
        df = df.iloc[:n]

    Y = torch.tensor(df[[ar_col, va_col]].values, dtype=torch.float32)
    matched_ids = df[id_col].astype(str).tolist()
    
    print(f"  ✅ Successfully matched {len(X)} samples.")
    return X.float(), Y, matched_ids

# =============================================================================
# 2. Multimodal EDA Functions (Ensuring compatibility)
# =============================================================================

def load_eda_features(eda_dir, music_ids, sr=200):
    features = []
    found_count = 0  # To track successful matches
    
    for mid in music_ids:
        # 1. Clean ID (removes .0 from floats)
        clean_id = str(int(float(mid))) if mid.replace('.','',1).isdigit() else mid
        
        # 2. UPDATE: Match the {id}_EDA.csv pattern
        path = os.path.join(eda_dir, f"{clean_id}_EDA.csv")
        
        if not os.path.exists(path):
            # Useful for initial debugging to see which IDs are failing
            # print(f"      ⚠️  Warning: EDA file missing for ID: {clean_id} at {path}")
            features.append(np.zeros(7))
            continue
            
        found_count += 1
        df = pd.read_csv(path, header=None)
        
        # Extract signal (all columns except time)
        numeric_cols = df.select_dtypes(include=[np.number]).iloc[:, 1:]
        eda = numeric_cols.mean(axis=1).values.astype(np.float32)
        
        # 3. Statistical Feature Extraction
        mean_eda, std_eda = np.mean(eda), np.std(eda)
        t = np.arange(len(eda)) / sr
        slope = np.polyfit(t, eda, 1)[0] if len(eda) > 1 else 0.0
        peaks, props = find_peaks(eda, prominence=0.01, distance=sr)
        mean_amp = props["prominences"].mean() if len(peaks) > 0 else 0.0
        
        features.append([
            mean_eda, std_eda, slope, len(peaks), mean_amp, 
            np.max(eda), np.mean(eda > mean_eda)
        ])
    
    print(f"    📊 EDA Sync Summary: Successfully matched {found_count}/{len(music_ids)} files.")
    
    arr = np.array(features, dtype=np.float32)
    # Min-max normalize each feature
    mn, mx = arr.min(0), arr.max(0)
    return (arr - mn) / (mx - mn + 1e-8)

def load_pmemo_with_eda(feature_path, label_path, eda_dir):
    # This logic ensures Audio and EDA are perfectly aligned
    X_audio, Y, matched_ids = load_pmemo_data(feature_path, label_path)
    X_eda_np = load_eda_features(eda_dir, matched_ids)
    X_eda = torch.tensor(X_eda_np, dtype=torch.float32)
    return X_audio, X_eda, Y

# =============================================================================
# 3. Quadrant Utilities (Required by mainB.py)
# =============================================================================

QUADRANT_NAMES = {0: "HVHA (Happy)", 1: "HVLA (Calm)", 2: "LVHA (Angry)", 3: "LVLA (Sad)"}

def get_emotion_quadrant(arousal, valence):
    mid = 0.5
    if valence >= mid and arousal >= mid: return 0
    if valence >= mid and arousal < mid:  return 1
    if valence < mid and arousal >= mid:  return 2
    return 3

def add_quadrant_labels(Y: torch.Tensor) -> torch.Tensor:
    arr = Y.numpy()
    return torch.tensor([get_emotion_quadrant(a, v) for a, v in arr], dtype=torch.int64)

def quadrant_r2_breakdown(y_true, y_pred):
    from sklearn.metrics import r2_score
    q_ids = np.array([get_emotion_quadrant(a, v) for a, v in y_true])
    res = {}
    for q_idx, name in QUADRANT_NAMES.items():
        mask = q_ids == q_idx
        n = mask.sum()
        if n < 5:
            res[name] = {"R2_Arousal": None, "R2_Valence": None, "n": int(n)}
        else:
            r2 = r2_score(y_true[mask], y_pred[mask], multioutput="raw_values")
            res[name] = {"R2_Arousal": r2[0], "R2_Valence": r2[1], "n": int(n)}
    return res