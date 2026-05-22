# Thesis Reports — Index

Clean, phase-organized report set for the thesis writing phase. Read in order; each
file documents one step of what was built and what was found. Numbers are current as
of the latest runs.

| File | Covers | Maps to thesis chapter |
| :-- | :-- | :-- |
| [00_overview.md](00_overview.md) | Motivation, research gap, 3-phase framework, headline results, contributions | Introduction |
| [01_phaseA_probing.md](01_phaseA_probing.md) | Does MERT encode music? Per-layer probing; gap = {tempo, key} | Methodology / Results (Phase A) |
| [02_phaseB_model.md](02_phaseB_model.md) | Emotion models, four-term loss, all encoder configs, key findings | Methodology / Results (Phase B) |
| [03_phaseC_explainability.md](03_phaseC_explainability.md) | Prototype retrieval, ante-hoc XAI, evaluation | Methodology / Results (Phase C) |
| [04_results_and_sota.md](04_results_and_sota.md) | Every results + SOTA table in one place | Results |
| [05_limitations_future_work.md](05_limitations_future_work.md) | Honest limitations + next steps | Discussion / Conclusion |
| [updater.md](updater.md) | Chronological lab log (Steps 1–6) — raw working notes | (not for the thesis body; reference only) |
| `archive/` | Superseded drafts (old 01–05 + presentations), kept for reference | — |

**Recurring caveat across all results:** global R²/CCC are inflated by the majority
HVHA quadrant (61% of PMEmo); minority-quadrant R² is negative across all configs.
This class-imbalance ceiling is the dominant limitation and is stated wherever
results appear.

**Project narrative in one line:** audio-only emotion prediction reaches the field
ceiling (Arousal R² 0.72 / Valence R² 0.58); the SupCR-organized latent space
supports emotionally coherent, self-explaining retrieval (Precision@5 ≈ 0.58); and
the gains, redundancies, and negative findings are reported with explicit honesty.

`JOURNEY.md` (repo root) is the plain-English personal narrative of the same story.
