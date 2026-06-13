"""
build_index_unified.py — Phase C index + query on the BEST model (symmetric)
=============================================================================
Replaces the single-MERT `mainC.py` / `retrieval.py` retrieval stack. Both the
index build and the query encoding go through the SAME
`UnifiedEnhancedEncoder.encode(...)` (Enhanced multi-encoder, cyclic key, 128-D
MLP bottleneck, L2-norm), so the database and the runtime query are encoded by
one model — no fall-back to a simple MERT baseline is possible.

Modes:
  build  — encode all 767 songs → prototypes_enhanced.npy
  query  — RE-ENCODE a query song through the same encoder, k-NN retrieve,
           + Audio ProtoPNet ante-hoc prototype profile
  verify — symmetry proof: re-encode every corpus song and confirm it matches
           the stored index vector (cosine ≈ 1, ‖Δ‖ ≈ 0)

Run from phaseC/:
  python build_index_unified.py --mode build
  python build_index_unified.py --mode verify
  python build_index_unified.py --mode query --query_id 760
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_PHASEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phaseB")
if _PHASEB not in sys.path:
    sys.path.insert(0, _PHASEB)

from data_utils import _match_dict_to_csv, get_emotion_quadrant   # noqa: E402
from models_enhanced import build_gap_vector, gap_dim_of          # noqa: E402
from encoder_unified import UnifiedEnhancedEncoder                # noqa: E402
from protopnet_readout import ProtoPNetReadout                    # noqa: E402

MERT_PATH = os.path.join(_PHASEB, "pmemo_mert_all_layers.pt")
W2V_PATH = os.path.join(_PHASEB, "pmemo_wav2vec_all_layers.pt")
THEORY_PATH = os.path.join(_PHASEB, "..", "phaseA", "data", "pmemo_music_theory.pt")
CSV_PATH = "/datasets/emotions/PMEmo2019/annotations/static_annotations.csv"
INDEX_PATH = "prototypes_enhanced.npy"
GAP_FEATURES = ["tempo", "key"]


def _match_theory(theory, df, id_col):
    out = {}
    for idx, raw in df[id_col].items():
        cid = str(int(raw)) if isinstance(raw, (int, float, np.number)) else str(raw)
        if cid in theory:
            out[idx] = theory[cid]
    return out


def load_aligned():
    """MERT ∩ w2v ∩ theory ∩ CSV, in one matched order, with ids + per-song theory."""
    mert = torch.load(MERT_PATH, map_location="cpu", weights_only=False)
    w2v = torch.load(W2V_PATH, map_location="cpu", weights_only=False)
    theory = torch.load(THEORY_PATH, map_location="cpu", weights_only=False)
    df = pd.read_csv(CSV_PATH)
    ar = [c for c in df.columns if "arousal" in c.lower()][0]
    va = [c for c in df.columns if "valence" in c.lower()][0]
    idc = [c for c in df.columns if any(x in c.lower() for x in ["music", "id"])][0]
    df[ar] = (df[ar] - df[ar].min()) / (df[ar].max() - df[ar].min() + 1e-8)
    df[va] = (df[va] - df[va].min()) / (df[va].max() - df[va].min() + 1e-8)
    mm, wm, tm = (_match_dict_to_csv(mert, df, idc), _match_dict_to_csv(w2v, df, idc),
                  _match_theory(theory, df, idc))
    common = [i for i in df.index if i in mm and i in wm and i in tm]
    Xm = torch.stack([mm[i] for i in common]).float()
    Xw = torch.stack([wm[i] for i in common]).float()
    theory_list = [tm[i] for i in common]
    dfm = df.loc[common]
    Y = torch.tensor(dfm[[ar, va]].values, dtype=torch.float32)
    ids = dfm[idc].astype(float).astype(int).tolist()
    return Xm, Xw, theory_list, Y, ids


def build(enc):
    Xm, Xw, theory_list, Y, ids = load_aligned()
    # assemble the cyclic-key gap matrix in the SAME way encode() does per song
    Xt = torch.stack([build_gap_vector(t, GAP_FEATURES, cyclic_key=True)
                      for t in theory_list]).float()
    print(f"  Aligned {len(ids)} songs | MERT {tuple(Xm.shape)} | w2v {tuple(Xw.shape)} "
          f"| theory {tuple(Xt.shape)} (cyclic key)")
    Z, pa, pv = enc.encode_batch(Xm, Xw, Xt)
    norms = np.linalg.norm(Z, axis=1)
    index = {
        "latents": Z.astype(np.float32),
        "music_ids": np.array(ids, dtype=np.int32),
        "arousal": Y[:, 0].numpy(), "valence": Y[:, 1].numpy(),
        "pred_arousal": pa, "pred_valence": pv,
        "encoder": "EnhancedDualSSLModel (cyclic key)",
        "checkpoint": "best_model_enhanced_final.pt",
    }
    np.save(INDEX_PATH, index)
    print(f"  ✅ index → {INDEX_PATH} | latents {Z.shape} | "
          f"L2 norms ∈ [{norms.min():.4f}, {norms.max():.4f}] (should be 1.0)")


def verify(enc):
    """SYMMETRY PROOF: re-encode every corpus song with the per-song encode()
    and confirm it equals the stored index vector."""
    Xm, Xw, theory_list, Y, ids = load_aligned()
    idx = np.load(INDEX_PATH, allow_pickle=True).item()
    stored = idx["latents"]
    sims, diffs = [], []
    for i in range(len(ids)):
        z, _, _ = enc.encode(Xm[i], Xw[i], theory_list[i])   # the QUERY path
        s = stored[i]
        sims.append(float(np.dot(z, s)))                      # cosine (both L2-normed)
        diffs.append(float(np.linalg.norm(z - s)))
    sims, diffs = np.array(sims), np.array(diffs)
    print(f"\n  SYMMETRY PROOF — re-encode (query path) vs stored index (build path):")
    print(f"    cosine(query, stored):  min {sims.min():.6f}  mean {sims.mean():.6f}")
    print(f"    ‖query − stored‖:       max {diffs.max():.2e}  mean {diffs.mean():.2e}")
    ok = sims.min() > 0.9999 and diffs.max() < 1e-3
    print(f"    → {'✅ SYMMETRIC' if ok else '❌ MISMATCH'}: build and query use the identical encoder.")
    sl = np.linalg.norm(stored, axis=1)
    print(f"    index L2 norms ∈ [{sl.min():.4f}, {sl.max():.4f}] | latent dim = {stored.shape[1]}")


def query(enc, readout, qid, k=5):
    Xm, Xw, theory_list, Y, ids = load_aligned()
    idx = np.load(INDEX_PATH, allow_pickle=True).item()
    if qid not in ids:
        print(f"❌ query id {qid} not in corpus. sample: {ids[:8]}"); return
    qi = ids.index(qid)
    # RE-ENCODE the query through the SAME encoder (not a lookup)
    z, pa, pv = enc.encode(Xm[qi], Xw[qi], theory_list[qi])
    db = idx["latents"]
    sims = db @ z                                            # cosine on L2-normed vectors
    sims[qi] = -1.0
    top = np.argsort(sims)[::-1][:k]
    print(f"\n🔍 Query {qid} | pred A={pa:.3f} V={pv:.3f} | encoder re-run (not lookup)")
    print(f"  Top-{k} similar songs (cosine):")
    for r in top:
        print(f"    id {int(idx['music_ids'][r]):>4} | sim {sims[r]:.4f} | "
              f"A={idx['arousal'][r]:.2f} V={idx['valence'][r]:.2f}")
    prof = readout.profile(Xm[qi])
    print(f"  ProtoPNet ante-hoc profile → predicted: {prof['predicted_quadrant']}")
    for q, a in prof["prototype_activation"].items():
        print(f"    {q:<14} activation {a:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "verify", "query"])
    ap.add_argument("--query_id", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    enc = UnifiedEnhancedEncoder()
    if args.mode == "build":
        build(enc)
    elif args.mode == "verify":
        verify(enc)
    elif args.mode == "query":
        if args.query_id is None:
            ap.error("--query_id required for query mode")
        query(enc, ProtoPNetReadout(), args.query_id, args.top_k)


if __name__ == "__main__":
    main()
