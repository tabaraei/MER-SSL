"""
data_utils_triple.py — Triple-branch loader (MERT + wav2vec2 + mel-spec)
=========================================================================
NEW file — existing data_utils.py is untouched. Reuses its robust
_match_dict_to_csv ID matcher and quadrant utilities.

load_pmemo_triple_ssl: intersects MERT ∩ wav2vec ∩ mel-spec ∩ CSV so the
mel-spectrogram branch is aligned to exactly the same songs/labels as the
dual-SSL model (apples-to-apples comparison).
"""

import pandas as pd
import torch

from data_utils import _match_dict_to_csv  # reuse existing robust matcher


def load_pmemo_triple_ssl(mert_path: str, w2v_path: str,
                          melspec_path: str, label_path: str):
    """
    Returns:
        X_mert : (N, 25, 1024)
        X_w2v  : (N, 13,  768)
        X_mel  : (N, 128, T)     pre-extracted log-mel spectrogram
        Y      : (N, 2)  [arousal, valence], min-max normalized to [0, 1]
        ids    : list[str] length N

    Only songs present in ALL FOUR sources are kept, in matched order.
    """
    print(f"  Loading MERT embeddings:     {mert_path}")
    mert_data = torch.load(mert_path, map_location="cpu")
    print(f"  Loading wav2vec2 embeddings: {w2v_path}")
    w2v_data = torch.load(w2v_path, map_location="cpu")
    print(f"  Loading mel-spectrograms:    {melspec_path}")
    mel_data = torch.load(melspec_path, map_location="cpu")

    print(f"  Loading labels: {label_path}")
    df = pd.read_csv(label_path)

    ar_col = [c for c in df.columns if "arousal" in c.lower()][0]
    va_col = [c for c in df.columns if "valence" in c.lower()][0]
    id_col = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]

    def _norm(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df[ar_col] = _norm(df[ar_col])
    df[va_col] = _norm(df[va_col])

    mert_matched = _match_dict_to_csv(mert_data, df, id_col)
    w2v_matched  = _match_dict_to_csv(w2v_data,  df, id_col)
    mel_matched  = _match_dict_to_csv(mel_data,  df, id_col)

    common_idx = [
        i for i in df.index
        if i in mert_matched and i in w2v_matched and i in mel_matched
    ]
    if not common_idx:
        raise ValueError(
            "No overlapping IDs across MERT, wav2vec, mel-spec, and CSV. "
            f"MERT∩CSV={len(mert_matched)}, w2v∩CSV={len(w2v_matched)}, "
            f"mel∩CSV={len(mel_matched)}. Did you run extract_pmemo_melspec.py?"
        )

    X_mert = torch.stack([mert_matched[i] for i in common_idx]).float()
    X_w2v  = torch.stack([w2v_matched[i]  for i in common_idx]).float()
    X_mel  = torch.stack([mel_matched[i]  for i in common_idx]).float()

    df_m = df.loc[common_idx]
    Y = torch.tensor(df_m[[ar_col, va_col]].values, dtype=torch.float32)
    ids = df_m[id_col].astype(str).tolist()

    print(f"  ✅ Triple matched {len(X_mert)} samples "
          f"(MERT ∩ wav2vec ∩ mel ∩ CSV) | mel shape {tuple(X_mel.shape[1:])}")
    return X_mert, X_w2v, X_mel, Y, ids
