"""
eval_rag_faithfulness.py — Layer-2 grounding/faithfulness check for the Phase C RAG
====================================================================================
Quantifies how faithfully the LLM (Layer 2) prose stays to the deterministic Layer-1
template, on the exported 5-song set (`artifacts/explanations_5songs.txt`).

Two assertion types are scored (see 03_phaseC_explainability.md §3b):
  (i)  song IDs named in the prose      -> grounded iff ID in {query U retrieved U foils}
  (ii) directional valence/arousal claim -> grounded iff its sign agrees with the
       retrieved neighbours' mean V-A (the set the prose actually describes).

The Layer-1 reference set (retrieved top-5 + 3 foils) is recomputed from the index
exactly as `export_explanations.py` built it (top_k=5 exclude_self, n_foils=3), and
the recomputed top-5 is sanity-checked against the IDs printed in the export file.

This is an *illustrative* check (n=5), not a headline quantitative claim.
Run:  ../.venv/bin/python eval_rag_faithfulness.py
Log:  reports/run_logs/rag_faithfulness.log
"""

import os, re
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "prototypes_dual.npy")
EXPORT = os.path.join(HERE, "artifacts", "explanations_5songs.txt")

# ── keyword lexicon for the directional (V-A sign) claim ────────────────────────
AROUSAL_HIGH = ["high arousal", "energetic", "energize", "energiz", "intense", "intensity",
                "driving", "powerful", "charged", "charge", "exhilarat", "tension", "tense",
                "visceral", "lively", "vibrant", "moving", "alive", "escalat", "raw", "urgency"]
AROUSAL_LOW  = ["low arousal", "calm", "serene", "tranquil", "unwind", "soothing", "gentle",
                "quiet", "subdued", "mellow", "reflective", "introspective", "soft", "peace"]
VALENCE_HIGH = ["high valence", "positive", "positivity", "uplift", "bright", "happy", "joy",
                "warm", "euphoric", "vibrant", "elevate", "uplifted"]
VALENCE_LOW  = ["low valence", "negative", "melancholic", "melancholy", "dark", "sad", "somber",
                "heavy", "uncomfortable", "tense", "subdued"]

def _sign_from_text(text, hi_words, lo_words):
    t = text.lower()
    hi = sum(t.count(w) for w in hi_words)
    lo = sum(t.count(w) for w in lo_words)
    if hi == lo: return None              # no committed claim
    return "high" if hi > lo else "low"

def _quad_sign(x):                          # V-A value -> 'high'/'low' about 0.5
    return "high" if x >= 0.5 else "low"


def parse_export(path):
    txt = open(path, encoding="utf-8").read()
    blocks = re.split(r"#{3,}\n# SONG (\d+)\n#{3,}", txt)
    songs = {}
    # blocks = ['', id, body, id, body, ...]
    for i in range(1, len(blocks), 2):
        sid = int(blocks[i]); body = blocks[i + 1]
        printed = [int(m) for m in re.findall(r"Song ID (\d+) \(cosine", body)]
        prose = body.split("LAYER 2: LLM EXPLANATION")[-1].strip("- \n")
        songs[sid] = {"printed_top5": printed, "prose": prose}
    return songs


def main():
    d = np.load(INDEX, allow_pickle=True).item()
    lat = d["latents"].astype(np.float32)            # (767,128) L2-normalised
    ids = d["music_ids"].astype(int)
    aro = d["arousal"].astype(float); val = d["valence"].astype(float)
    id2row = {int(m): r for r, m in enumerate(ids)}
    valid_ids = set(int(x) for x in ids)

    songs = parse_export(EXPORT)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 78)
    out("Phase C RAG — Layer-2 grounding / faithfulness  (illustrative, n=5)")
    out("Index: prototypes_dual.npy | source: artifacts/explanations_5songs.txt")
    out("=" * 78)

    id_gp_scores = []        # ID-grounding precision (songs that name IDs)
    dir_axis_hits = dir_axis_total = 0
    dir_song_ok = 0; dir_song_total = 0

    for sid in sorted(songs):
        row = id2row[sid]; q = lat[row]
        sims = lat @ q                                # cosine (unit norm)
        order = np.argsort(-sims)
        top5 = [int(ids[r]) for r in order if r != row][:5]
        foils = [int(ids[r]) for r in np.argsort(sims)[:3]]
        ref = {sid} | set(top5) | set(foils)

        # sanity: recomputed top-5 must equal the printed top-5
        printed = songs[sid]["printed_top5"]
        match = (top5 == printed)

        # neighbours' mean V-A (what the prose describes)
        nrows = [id2row[m] for m in top5]
        na, nv = float(np.mean(aro[nrows])), float(np.mean(val[nrows]))

        prose = songs[sid]["prose"]
        # (i) ID grounding
        mentioned = [int(x) for x in re.findall(r"(?<![\d.])(\d{1,4})(?![\d.%])", prose)]
        mentioned = [m for m in mentioned if 1 <= m <= 1100]
        grounded = [m for m in mentioned if m in ref]
        id_gp = (len(grounded) / len(mentioned)) if mentioned else None
        if id_gp is not None: id_gp_scores.append(id_gp)

        # (ii) directional claim vs neighbour mean
        ca = _sign_from_text(prose, AROUSAL_HIGH, AROUSAL_LOW)
        cv = _sign_from_text(prose, VALENCE_HIGH, VALENCE_LOW)
        axis_ok = []
        for claim, ref_val, axis in [(ca, na, "A"), (cv, nv, "V")]:
            if claim is None: continue
            ok = (claim == _quad_sign(ref_val))
            axis_ok.append((axis, claim, _quad_sign(ref_val), ok))
            dir_axis_total += 1; dir_axis_hits += int(ok)
        song_dir_ok = all(o for *_, o in axis_ok) if axis_ok else None
        if song_dir_ok is not None:
            dir_song_total += 1; dir_song_ok += int(song_dir_ok)

        out("")
        out(f"SONG {sid}   query quad={_quad_sign(aro[row]).upper()[0]}A/{_quad_sign(val[row]).upper()[0]}V"
            f"  (A={aro[row]:.3f}, V={val[row]:.3f})   recomputed top-5 == printed: {match}")
        out(f"  Layer-1 reference set: query+top5+foils = {sorted(ref)}")
        out(f"    foils (lowest cosine) = {foils}")
        if mentioned:
            out(f"  (i) IDs named in prose = {mentioned} | grounded = {grounded}"
                f" | ID-grounding precision = {id_gp:.2f}")
        else:
            out(f"  (i) IDs named in prose = none (generic references) | ID-GP = N/A")
        out(f"  (ii) neighbour mean A={na:.3f} ({_quad_sign(na)}), V={nv:.3f} ({_quad_sign(nv)})")
        for axis, claim, refsign, ok in axis_ok:
            out(f"       prose claims {axis}={claim:<4} vs neighbours {axis}={refsign:<4} -> {'OK' if ok else 'MISMATCH'}")
        out(f"       directional faithfulness (this song): "
            f"{'PASS' if song_dir_ok else 'FAIL' if song_dir_ok is not None else 'N/A'}")

    out("")
    out("-" * 78)
    mean_idgp = np.mean(id_gp_scores) if id_gp_scores else float("nan")
    out(f"AGGREGATE (n={len(songs)} songs):")
    out(f"  ID-grounding precision : mean = {mean_idgp:.2f}  "
        f"(over {len(id_gp_scores)} song(s) that name explicit IDs)")
    out(f"  Directional faithfulness: {dir_song_ok}/{dir_song_total} songs fully consistent; "
        f"{dir_axis_hits}/{dir_axis_total} individual V/A axis-claims correct "
        f"({dir_axis_hits/dir_axis_total:.2f})")
    out("-" * 78)

    logdir = os.path.join(HERE, "..", "..", "..", "reports", "run_logs")
    logdir = os.path.abspath(logdir)
    if os.path.isdir(logdir):
        with open(os.path.join(logdir, "rag_faithfulness.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        out(f"[saved] {os.path.join(logdir, 'rag_faithfulness.log')}")


if __name__ == "__main__":
    main()
