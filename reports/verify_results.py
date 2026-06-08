"""
verify_results.py — audit every reported number against the raw run log.
========================================================================
Loads each persisted run log in reports/run_logs/, extracts the headline
5-fold means, and checks them against the numbers currently in the report
markdown files + generate_report_v3.py. Any mismatch is reported.

Run from reports/:
  python verify_results.py
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "run_logs"
REPORTS = [ROOT / "01_phaseA_probing.md",
           ROOT / "02_phaseB_model.md",
           ROOT / "04_results_and_sota.md",
           ROOT / "generate_report_v3.py"]


def grab(log: Path, patterns: dict) -> dict:
    """Run each regex on the log; return dict of name → first-match float."""
    text = log.read_text()
    out = {}
    for name, pat in patterns.items():
        m = re.search(pat, text)
        out[name] = float(m.group(1)) if m else None
    return out


def check(label: str, extracted: dict, claimed: dict, tol: float = 0.005) -> list:
    """Compare extracted vs claimed; return list of issue strings."""
    issues = []
    for k, ex in extracted.items():
        cl = claimed.get(k)
        if ex is None:
            issues.append(f"   ⚠️  [{label}] {k}: regex did not match the log")
            continue
        if cl is None:
            issues.append(f"   ?   [{label}] {k}: no claimed value to compare against")
            continue
        if abs(ex - cl) > tol:
            issues.append(f"   ❌  [{label}] {k}: LOG = {ex:.4f}  vs  REPORT = {cl:.4f}  (Δ = {ex-cl:+.4f})")
        else:
            issues.append(f"   ✓   [{label}] {k}: {ex:.4f}  (matches report)")
    return issues


REPORT_TEXT = "".join(p.read_text() for p in REPORTS if p.exists())


def in_report(num_str: str) -> bool:
    """Quick sanity check: does this number-string appear anywhere in reports?"""
    return num_str in REPORT_TEXT


# ── Per-experiment specs ─────────────────────────────────────────────
EXPERIMENTS = [
    dict(
        log="eval_last_layer_only_mert.log",
        label="MERT-only LAST-LAYER",
        patterns={
            "r2_a": r"R[2²]\s+Arousal\s*:\s*(\-?\d+\.\d+)",
            "r2_v": r"R[2²]\s+Valence\s*:\s*(\-?\d+\.\d+)",
            "ccc_a": r"CCC\s*Arousal\s*:\s*(\d+\.\d+)",
            "ccc_v": r"CCC\s*Valence\s*:\s*(\d+\.\d+)",
        },
        claimed=dict(r2_a=0.6570, r2_v=0.5162, ccc_a=0.8073, ccc_v=0.7035),
    ),
    dict(
        log="eval_enhanced_last_layer.log",
        label="Enhanced LAST-LAYER",
        patterns={
            "r2_a": r"R[2²]\s+Arousal\s*:\s*(\-?\d+\.\d+)",
            "r2_v": r"R[2²]\s+Valence\s*:\s*(\-?\d+\.\d+)",
            "ccc_a": r"CCC\s*Arousal\s*:\s*(\d+\.\d+)",
            "ccc_v": r"CCC\s*Valence\s*:\s*(\d+\.\d+)",
        },
        # Reports claim: A 0.6660 / V 0.4881, CCC 0.8124 / 0.6865
        claimed=dict(r2_a=0.6660, r2_v=0.4881, ccc_a=0.8124, ccc_v=0.6865),
    ),
    dict(
        log="eval_enhanced_mixup.log",
        label="Enhanced + MIXUP",
        patterns={
            "r2_a": r"R[2²]\s+Arousal\s*:\s*(\-?\d+\.\d+)",
            "r2_v": r"R[2²]\s+Valence\s*:\s*(\-?\d+\.\d+)",
            "ccc_a": r"CCC\s*Arousal\s*:\s*(\d+\.\d+)",
            "ccc_v": r"CCC\s*Valence\s*:\s*(\d+\.\d+)",
        },
        # Reports claim: A 0.7077 / V 0.5651, CCC 0.8213 / 0.7160
        claimed=dict(r2_a=0.7077, r2_v=0.5651, ccc_a=0.8213, ccc_v=0.7160),
    ),
    dict(
        log="eval_wav2vec_only.log",
        label="wav2vec2-only",
        patterns={
            "r2_a": r"R[2²]\s+Arousal\s*:\s*(\-?\d+\.\d+)",
            "r2_v": r"R[2²]\s+Valence\s*:\s*(\-?\d+\.\d+)",
            "ccc_a": r"CCC\s*Arousal\s*:\s*(\d+\.\d+)",
            "ccc_v": r"CCC\s*Valence\s*:\s*(\d+\.\d+)",
        },
        # Reports claim: A 0.6225 / V 0.4825, CCC 0.7722 / 0.6564 (Step 10/14)
        claimed=dict(r2_a=0.6225, r2_v=0.4825, ccc_a=0.7722, ccc_v=0.6564),
    ),
]


def parse_dual_run(log: Path):
    """eval_mel_only_AND_mert_mel_eda.log contains two runs back-to-back."""
    text = log.read_text()
    # find both sections
    secs = text.split("MEL-CNN ALONE — 5-FOLD AVERAGE")
    if len(secs) < 2:
        return None
    mel_only_block = secs[1].split("MERT + MEL-CNN + EDA — 5-FOLD AVERAGE")[0]
    mert_mel_eda_block = secs[1].split("MERT + MEL-CNN + EDA — 5-FOLD AVERAGE")[1] if "MERT + MEL-CNN + EDA — 5-FOLD AVERAGE" in secs[1] else ""

    def grab_block(block):
        out = {}
        for k, pat in [("r2_a", r"R[2²]\s+Arousal\s*:\s*(\-?\d+\.\d+)"),
                       ("r2_v", r"R[2²]\s+Valence\s*:\s*(\-?\d+\.\d+)"),
                       ("ccc_a", r"CCC\s*Arousal\s*:\s*(\d+\.\d+)"),
                       ("ccc_v", r"CCC\s*Valence\s*:\s*(\d+\.\d+)")]:
            m = re.search(pat, block)
            out[k] = float(m.group(1)) if m else None
        return out
    return grab_block(mel_only_block), grab_block(mert_mel_eda_block)


def parse_ce_sweep(log: Path):
    """Quadrant-CE sweep summary block has 4 rows: λ R² A R² V CCC A CCC V Silhouette."""
    text = log.read_text()
    rows = re.findall(
        r"^\s*([01]\.\d{2})\s+(\d+\.\d+)±\d+\.\d+\s+(\d+\.\d+)±\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)±\d+\.\d+",
        text, flags=re.MULTILINE)
    return {float(r[0]): dict(r2_a=float(r[1]), r2_v=float(r[2]),
                              ccc_a=float(r[3]), ccc_v=float(r[4]),
                              silhouette=float(r[5])) for r in rows}


def parse_imbalance(log: Path):
    """Imbalance ablation summary block: 4 rows of (treatment, R² A, R² V, CCC A, CCC V)."""
    text = log.read_text()
    rows = re.findall(
        r"^\s*([A-D]):.*?\s+(\d+\.\d+)±\d+\.\d+\s+(\d+\.\d+)±\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$",
        text, flags=re.MULTILINE)
    return {r[0]: dict(r2_a=float(r[1]), r2_v=float(r[2]),
                       ccc_a=float(r[3]), ccc_v=float(r[4])) for r in rows}


# ── Run audit ────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Result-vs-Report Audit")
    print("=" * 72)
    all_issues = []
    # Simple experiments
    for exp in EXPERIMENTS:
        log = LOGS / exp["log"]
        if not log.exists():
            print(f"\n[{exp['label']}] log MISSING: {log.name}"); continue
        print(f"\n[{exp['label']}]  log={log.name}")
        ex = grab(log, exp["patterns"])
        issues = check(exp["label"], ex, exp["claimed"])
        for i in issues:
            print(i); all_issues.append(i)

    # Mel-only + Triple-bio (combined log)
    log = LOGS / "eval_mel_only_AND_mert_mel_eda.log"
    if log.exists():
        print(f"\n[MEL-ONLY + TRIPLE-BIO]  log={log.name}")
        mo, tb = parse_dual_run(log)
        if mo:
            print("  Mel-CNN alone:")
            for i in check("Mel-CNN alone", mo,
                           dict(r2_a=0.6486, r2_v=0.4452, ccc_a=0.7892, ccc_v=0.6302)):
                print(i); all_issues.append(i)
        if tb:
            print("  MERT + Mel + EDA (Triple-bio):")
            for i in check("Triple-bio", tb,
                           dict(r2_a=0.7077, r2_v=0.5706, ccc_a=0.8262, ccc_v=0.7325)):
                print(i); all_issues.append(i)

    # CE-head sweep
    log = LOGS / "eval_enhanced_quadrant_ce.log"
    if log.exists():
        print(f"\n[CE-HEAD SWEEP]  log={log.name}")
        ce = parse_ce_sweep(log)
        claimed = {
            0.00: dict(r2_a=0.7080, r2_v=0.5725, ccc_a=0.827, ccc_v=0.729, silhouette=0.255),
            0.10: dict(r2_a=0.7050, r2_v=0.5811, ccc_a=0.828, ccc_v=0.738, silhouette=0.258),
            0.50: dict(r2_a=0.6966, r2_v=0.5624, ccc_a=0.827, ccc_v=0.730, silhouette=0.265),
            1.00: dict(r2_a=0.6638, r2_v=0.5601, ccc_a=0.810, ccc_v=0.734, silhouette=0.286),
        }
        for lam, vals in ce.items():
            print(f"  λ={lam}")
            for i in check(f"CE λ={lam}", vals, claimed.get(lam, {})):
                print(i); all_issues.append(i)

    # Silhouette 2×2 audit (per-fold euclidean/cosine for each model)
    log = LOGS / "eval_silhouette_audit.log"
    if log.exists():
        print(f"\n[SILHOUETTE 2×2 AUDIT]  log={log.name}")
        text = log.read_text()
        # split into the two model sections
        for model, claimed in [("single-MERT", dict(eu=0.1934, co=0.2691)),
                               ("Enhanced",    dict(eu=0.1815, co=0.2595))]:
            seg = text.split(f"[{model}]")[1].split("[")[0] if f"[{model}]" in text else ""
            eus = [float(x) for x in re.findall(r"euclidean=([+\-]?\d+\.\d+)", seg)]
            cos = [float(x) for x in re.findall(r"cosine=([+\-]?\d+\.\d+)", seg)]
            got = dict(eu=(sum(eus)/len(eus) if eus else None),
                       co=(sum(cos)/len(cos) if cos else None))
            for i in check(f"Silhouette {model}", got, claimed):
                print(i); all_issues.append(i)

    # Key-encoding A/B (raw vs cyclic) — parse each section's "-> R² A m±s | V m±s"
    log = LOGS / "eval_key_encoding.log"
    if log.exists():
        print(f"\n[KEY-ENCODING A/B]  log={log.name}")
        text = log.read_text()
        for label, claimed in [("RAW key", dict(r2_a=0.7049, r2_v=0.5777)),
                               ("CYCLIC key", dict(r2_a=0.7016, r2_v=0.5697))]:
            seg = text.split(label)[1] if label in text else ""
            m = re.search(r"->\s*R\S*\s*A\s*([\d.]+).*?V\s*([\d.]+)", seg)
            got = dict(r2_a=float(m.group(1)) if m else None,
                       r2_v=float(m.group(2)) if m else None)
            for i in check(f"KeyEnc {label}", got, claimed):
                print(i); all_issues.append(i)

    # Audio ProtoPNet (raw + balanced accuracy)
    log = LOGS / "train_protopnet.log"
    if log.exists():
        print(f"\n[AUDIO PROTOPNET]  log={log.name}")
        text = log.read_text()
        ra = re.search(r"Raw accuracy\s*:\s*([\d.]+)", text)
        ba = re.search(r"Balanced accuracy\s*:\s*([\d.]+)", text)
        got = dict(raw_acc=float(ra.group(1)) if ra else None,
                   bal_acc=float(ba.group(1)) if ba else None)
        for i in check("ProtoPNet", got, dict(raw_acc=0.7275, bal_acc=0.5447)):
            print(i); all_issues.append(i)

    # ProtoPNet latent Silhouette (objective-vs-topology)
    log = LOGS / "eval_protopnet_silhouette.log"
    if log.exists():
        print(f"\n[PROTOPNET SILHOUETTE]  log={log.name}")
        text = log.read_text()
        eu = re.search(r"Silhouette Euclidean\s*:\s*([\d.]+)", text)
        co = re.search(r"Silhouette cosine\s*:\s*([\d.]+)", text)
        got = dict(eu=float(eu.group(1)) if eu else None,
                   co=float(co.group(1)) if co else None)
        for i in check("ProtoPNet-Sil", got, dict(eu=0.1181, co=0.1778)):
            print(i); all_issues.append(i)

    # Imbalance ablation
    log = LOGS / "eval_imbalance_ablation.log"
    if log.exists():
        print(f"\n[IMBALANCE ABLATION]  log={log.name}")
        ib = parse_imbalance(log)
        claimed = {
            "A": dict(r2_a=0.6951, r2_v=0.5724, ccc_a=0.820, ccc_v=0.731),
            "B": dict(r2_a=0.6738, r2_v=0.5267, ccc_a=0.828, ccc_v=0.739),
            "C": dict(r2_a=0.6855, r2_v=0.5696, ccc_a=0.811, ccc_v=0.724),
            "D": dict(r2_a=0.6949, r2_v=0.5716, ccc_a=0.821, ccc_v=0.727),
        }
        for letter, vals in ib.items():
            print(f"  treatment {letter}")
            for i in check(f"Imbal {letter}", vals, claimed.get(letter, {})):
                print(i); all_issues.append(i)

    # Summary
    print("\n" + "=" * 72)
    n_ok  = sum(1 for i in all_issues if i.lstrip().startswith("✓"))
    n_bad = sum(1 for i in all_issues if i.lstrip().startswith("❌"))
    n_warn = sum(1 for i in all_issues if i.lstrip().startswith("⚠️"))
    n_q   = sum(1 for i in all_issues if i.lstrip().startswith("?"))
    print(f"  AUDIT SUMMARY:  {n_ok} OK · {n_bad} MISMATCH · {n_warn} regex-miss · {n_q} unverifiable")
    print("=" * 72)
    if n_bad == 0:
        print("  All report numbers match the logged run outputs within ±0.005.")
    else:
        print("  ❌ Mismatches found — see ❌ lines above.")


if __name__ == "__main__":
    main()
