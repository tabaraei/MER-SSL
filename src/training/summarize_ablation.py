"""
summarize_ablation.py — collate the k,p sweep into a markdown table
====================================================================
Parses logs/ablation_k*_p*.log produced by run_joint_ablation.sh,
extracts the final JOINT LEARNING averages, and writes a table sorted
by Valence R² (the metric this whole replication targets).

Usage:
    cd phaseB/
    python summarize_ablation.py
"""

import glob
import os
import re

LOG_GLOB = "logs/ablation_k*_p*.log"
OUT_MD = "logs/ablation_summary.md"

# Matches:  R²  Arousal : 0.6810 | Valence : 0.5601
RE_R2 = re.compile(r"R²\s*Arousal\s*:\s*([-\d.]+)\s*\|\s*Valence\s*:\s*([-\d.]+)")
# Matches:  CCC Arousal : 0.8142 | CCC Valence : 0.7182
RE_CCC = re.compile(r"CCC\s*Arousal\s*:\s*([-\d.]+)\s*\|\s*CCC\s*Valence\s*:\s*([-\d.]+)")
RE_KP = re.compile(r"ablation_k([-\d.]+)_p([-\d.]+)\.log$")


def _last(pattern, text):
    m = list(pattern.finditer(text))
    return (float(m[-1].group(1)), float(m[-1].group(2))) if m else (None, None)


def main():
    rows = []
    files = sorted(glob.glob(LOG_GLOB))
    if not files:
        print(f"⚠️  No logs matched {LOG_GLOB}. Run run_joint_ablation.sh first.")
        return

    for fp in files:
        kp = RE_KP.search(os.path.basename(fp))
        if not kp:
            continue
        k, p = kp.group(1), kp.group(2)
        txt = open(fp, encoding="utf-8", errors="replace").read()
        # Only parse the FINAL AVERAGE block to avoid per-fold noise
        anchor = txt.find("JOINT LEARNING FINAL AVERAGE")
        block = txt[anchor:] if anchor != -1 else txt
        r2_a, r2_v = _last(RE_R2, block)
        ccc_a, ccc_v = _last(RE_CCC, block)
        rows.append({
            "k": k, "p": p,
            "r2_a": r2_a, "r2_v": r2_v, "ccc_a": ccc_a, "ccc_v": ccc_v,
        })

    rows.sort(key=lambda r: (r["r2_v"] is None, -(r["r2_v"] or -1e9)))

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"

    lines = [
        "# Joint-Learning Ablation (IADS-E × PMEmo)",
        "",
        "Sorted by Valence R² (target metric). Eval = PMEmo test folds only.",
        "",
        "| k | p | R² Arousal | R² Valence | CCC A | CCC V |",
        "| :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['p']} | {fmt(r['r2_a'])} | {fmt(r['r2_v'])} "
            f"| {fmt(r['ccc_a'])} | {fmt(r['ccc_v'])} |"
        )
    md = "\n".join(lines) + "\n"

    os.makedirs("logs", exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"✅ Saved → {OUT_MD}")


if __name__ == "__main__":
    main()
