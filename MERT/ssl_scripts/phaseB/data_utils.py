"""
data_utils.py — Final Debugged Version
===========================================
Fixed: ValueError by implementing robust ID matching (handles float/int/string).
Fixed: Return signature to match mainB.py requirements.
Included: All quadrant and multimodal EDA functions.
"""

import os
import re
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
# 1b. Dual-SSL Loading (MERT + wav2vec2)
# =============================================================================

def _match_dict_to_csv(data_dict: dict, df: pd.DataFrame, id_col: str) -> dict:
    """
    Reuses the robust ID-matching pattern from load_pmemo_data (handles
    ID as 760 int, 760.0 float, or '760' str).

    Returns {df_row_index: tensor} for every CSV row whose cleaned ID is a
    key in data_dict. Keyed by DataFrame index so multiple dicts can be
    intersected on common rows.
    """
    matched = {}
    for idx, raw_id in df[id_col].items():
        clean_id = str(int(raw_id)) if isinstance(raw_id, (int, float, np.number)) else str(raw_id)
        if clean_id in data_dict:
            matched[idx] = torch.as_tensor(data_dict[clean_id])
    return matched


def load_pmemo_dual_ssl(mert_path: str, w2v_path: str, label_path: str):
    """
    Loads MERT and wav2vec2 all-layer embeddings and aligns them to the
    PMEmo annotations. Only songs present in ALL THREE sources
    (MERT ∩ wav2vec ∩ CSV) are kept, in matched order.

    Returns:
        X_mert      : (N, 25, 1024)
        X_w2v       : (N, 13,  768)
        Y           : (N, 2)  [arousal, valence], min-max normalized to [0, 1]
        matched_ids : list[str] of length N
    """
    print(f"  Loading MERT embeddings from:     {mert_path}")
    mert_data = torch.load(mert_path, map_location="cpu")
    print(f"  Loading wav2vec2 embeddings from: {w2v_path}")
    w2v_data = torch.load(w2v_path, map_location="cpu")

    print(f"  Loading labels from: {label_path}")
    df = pd.read_csv(label_path)

    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]

    def _norm(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col])
    df[va_col] = _norm(df[va_col])

    mert_matched = _match_dict_to_csv(mert_data, df, id_col)
    w2v_matched  = _match_dict_to_csv(w2v_data, df, id_col)

    common_idx = [i for i in df.index if i in mert_matched and i in w2v_matched]
    if not common_idx:
        raise ValueError(
            "No overlapping IDs across MERT, wav2vec, and CSV. "
            f"MERT∩CSV={len(mert_matched)}, w2v∩CSV={len(w2v_matched)}."
        )

    X_mert = torch.stack([mert_matched[i] for i in common_idx]).float()
    X_w2v  = torch.stack([w2v_matched[i]  for i in common_idx]).float()

    df_matched  = df.loc[common_idx]
    Y           = torch.tensor(df_matched[[ar_col, va_col]].values, dtype=torch.float32)
    matched_ids = df_matched[id_col].astype(str).tolist()

    print(f"  ✅ Dual-SSL matched {len(X_mert)} samples (MERT ∩ wav2vec ∩ CSV)")
    return X_mert, X_w2v, Y, matched_ids


# =============================================================================
# 1c. IADS-E Joint-Learning Loaders (Simonetta et al. 2024 replication, SSL)
# =============================================================================

def _iadse_sound_id(stem: str) -> str:
    """IADS-E audio is named `{SoundID}_2.wav` (e.g. 0319_2, 0015_b_2).
    The label sheet keys on `SoundID` (0319, 0015_b). Strip one trailing `_2`."""
    return re.sub(r"_2$", "", stem)


def load_iadse_dual_ssl(mert_path, w2v_path, label_path,
                        valence_col, arousal_col, id_col,
                        label_min=1.0, label_max=9.0):
    """
    Loads IADS-E dual-SSL embeddings (generalized environmental sounds).

    - Embedding .pt keys are audio stems (`0319_2`); the label CSV keys on
      `SoundID` (`0319`). Keys are normalized via _iadse_sound_id before matching.
    - Labels are SAM in [label_min, label_max]; normalized to [0, 1].
    - Y is returned as [arousal, valence] to match load_pmemo_data's ordering.

    Returns:
        X_mert (N,25,1024), X_w2v (N,13,768), Y (N,2), ids (list[str]),
        domain = torch.ones(N, dtype=torch.long)   # 1 = generalized sound
    """
    print(f"  Loading IADS-E MERT from:     {mert_path}")
    mert_raw = torch.load(mert_path, map_location="cpu")
    print(f"  Loading IADS-E wav2vec from:  {w2v_path}")
    w2v_raw = torch.load(w2v_path, map_location="cpu")

    # Normalize embedding keys: audio stem -> Sound ID (keep-last; no collisions
    # in practice since every IADS-E clip is the `_2` take)
    mert_data = {_iadse_sound_id(k): v for k, v in mert_raw.items()}
    w2v_data  = {_iadse_sound_id(k): v for k, v in w2v_raw.items()}

    print(f"  Loading IADS-E labels from:   {label_path}")
    df = pd.read_csv(label_path, dtype={id_col: str})
    df = df.dropna(subset=[valence_col, arousal_col, id_col])

    def _norm(col):
        return ((col - label_min) / (label_max - label_min)).clip(0.0, 1.0)
    df = df.copy()
    df[valence_col] = _norm(df[valence_col].astype(float))
    df[arousal_col] = _norm(df[arousal_col].astype(float))

    mert_matched = _match_dict_to_csv(mert_data, df, id_col)
    w2v_matched  = _match_dict_to_csv(w2v_data, df, id_col)

    common_idx = [i for i in df.index if i in mert_matched and i in w2v_matched]
    if not common_idx:
        raise ValueError(
            "No overlapping IDs across IADS-E MERT, wav2vec, and label CSV. "
            f"MERT∩CSV={len(mert_matched)}, w2v∩CSV={len(w2v_matched)}. "
            "Did you run the IADS-E extraction commands?"
        )

    X_mert = torch.stack([mert_matched[i] for i in common_idx]).float()
    X_w2v  = torch.stack([w2v_matched[i]  for i in common_idx]).float()

    df_m = df.loc[common_idx]
    # [arousal, valence] to match load_pmemo_data
    Y = torch.tensor(df_m[[arousal_col, valence_col]].values, dtype=torch.float32)
    ids = df_m[id_col].astype(str).tolist()
    domain = torch.ones(len(ids), dtype=torch.long)

    print(f"  ✅ IADS-E matched {len(X_mert)} sounds (MERT ∩ wav2vec ∩ CSV)")
    return X_mert, X_w2v, Y, ids, domain


def _quadrant_counts(Y: torch.Tensor) -> dict:
    arr = Y.numpy()
    counts = {name: 0 for name in QUADRANT_NAMES.values()}
    for a, v in arr:
        counts[QUADRANT_NAMES[get_emotion_quadrant(a, v)]] += 1
    return counts


def _print_coverage(tag: str, Y: torch.Tensor):
    c = _quadrant_counts(Y)
    n = max(len(Y), 1)
    cells = " | ".join(f"{k.split()[0]}={v} ({100*v/n:.0f}%)" for k, v in c.items())
    print(f"    {tag:<14} N={len(Y):<4} | {cells}")


def build_joint_dataset(pmemo_mert, pmemo_w2v, pmemo_labels_csv,
                        iadse_mert,  iadse_w2v,  iadse_labels_csv,
                        iadse_valence_col, iadse_arousal_col, iadse_id_col,
                        k=1.0, p=1.0, seed=42):
    """
    Simonetta mixing: keep round(p * |PMEmo|) music samples (domain 0) and
    round(k * |IADS-E|) sound samples (domain 1); k, p ∈ [0, 1].

    Returns:
        X_mert, X_w2v, Y, domain, source_tags
        source_tags : list[str] like 'pmemo:760' / 'iadse:0319' (debugging)

    Prints per-dataset and combined Russell-quadrant coverage so V-A plane
    complementarity is visible before training.
    """
    rng = np.random.default_rng(seed)

    Xm_p, Xw_p, Y_p, ids_p = load_pmemo_dual_ssl(pmemo_mert, pmemo_w2v, pmemo_labels_csv)
    dom_p = torch.zeros(len(Y_p), dtype=torch.long)

    Xm_s, Xw_s, Y_s, ids_s, dom_s = load_iadse_dual_ssl(
        iadse_mert, iadse_w2v, iadse_labels_csv,
        iadse_valence_col, iadse_arousal_col, iadse_id_col,
    )

    def _sample(n, ratio):
        m = int(round(ratio * n))
        m = max(0, min(n, m))
        return np.sort(rng.choice(n, size=m, replace=False)) if m < n else np.arange(n)

    pi = _sample(len(Y_p), p)
    si = _sample(len(Y_s), k)

    X_mert = torch.cat([Xm_p[pi], Xm_s[si]], dim=0)
    X_w2v  = torch.cat([Xw_p[pi], Xw_s[si]], dim=0)
    Y      = torch.cat([Y_p[pi],  Y_s[si]],  dim=0)
    domain = torch.cat([dom_p[pi], dom_s[si]], dim=0)
    source_tags = (
        [f"pmemo:{ids_p[i]}" for i in pi] + [f"iadse:{ids_s[i]}" for i in si]
    )

    print("\n  📐 V-A Quadrant Coverage (Russell):")
    _print_coverage("PMEmo (kept)", Y_p[pi])
    _print_coverage("IADS-E (kept)", Y_s[si])
    _print_coverage("COMBINED", Y)
    print(f"  🌍 Joint set: {len(Y)} = p·|PMEmo|({len(pi)}) + k·|IADS-E|({len(si)})  "
          f"[k={k}, p={p}, seed={seed}]")

    return X_mert, X_w2v, Y, domain, source_tags


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