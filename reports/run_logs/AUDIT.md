# Audit log

`verify_results.py` extracts headline numbers from every persisted run log and
cross-checks them against the values currently quoted in:
- `01_phaseA_probing.md`
- `02_phaseB_model.md`
- `04_results_and_sota.md`
- `generate_report_v3.py`

## Latest audit (2026-05-28)

```
AUDIT SUMMARY:  60 OK · 0 MISMATCH · 0 regex-miss · 0 unverifiable
All report numbers match the logged run outputs within ±0.005.
```

Verified per experiment:

| Run log                                     | R² A | R² V | CCC A | CCC V | + extras       |
| :--                                         | :-:  | :-:  | :-:   | :-:   | :--            |
| eval_wav2vec_only.log                       | ✓    | ✓    | ✓     | ✓     |                |
| eval_last_layer_only_mert.log               | ✓    | ✓    | ✓     | ✓     |                |
| eval_enhanced_last_layer.log                | ✓    | ✓    | ✓     | ✓     |                |
| eval_imbalance_ablation.log (A · B · C · D) | ✓×4  | ✓×4  | ✓×4   | ✓×4   |                |
| eval_enhanced_mixup.log                     | ✓    | ✓    | ✓     | ✓     |                |
| eval_mel_only_AND_mert_mel_eda.log          | ✓×2  | ✓×2  | ✓×2   | ✓×2   |                |
| eval_enhanced_quadrant_ce.log (λ=0/0.1/0.5/1.0) | ✓×4 | ✓×4 | ✓×4  | ✓×4   | Silhouette ✓×4 |

Tolerance: ±0.005 absolute on R² / CCC / Silhouette (slack for the 3rd
decimal rounding used in the prose tables).

## How to re-audit after any future run

1. Save the run's stdout to `reports/run_logs/<descriptive_name>.log`.
2. If the new run has a different summary format, add a parser stanza to
   `verify_results.py` (or extend `EXPERIMENTS`).
3. `cd reports && python verify_results.py`.

## What this does NOT check

- Per-fold numbers — only the 5-fold means + std.
- Per-quadrant R² breakdowns (the minority-quadrant negatives).
- Numbers from earlier (pre-this-session) runs that don't have a persisted
  log here — those still trace back to the historic
  `MERT/ssl_scripts/phaseB/logs/` files and to `updater.md` steps 1–14.
- Numbers cited from prior published work (Zhang 2018, Music2Emo, Simonetta,
  EmoMucs) — those are cross-checked against the cited papers themselves,
  not against any local log.
