# CLAUDE.md — MER Thesis Workspace

> Workspace memory for the thesis **"A Critical Audit of Self-Supervised Music
> Representations for Explainable Emotion Recognition"** (MERT on PMEmo 2019).
> These instructions are authoritative for all work in this repository.

---

## ACTIVE SKILLS (use these for writing / reviewing)

- **`academic-paper`** — 12-agent paper-writing pipeline (outline → draft → revision;
  LaTeX/DOCX/PDF; citation formatting; AI-disclosure). Use it to turn `reports/` into
  manuscript prose. **Constraint:** it must only be fed numbers that pass the audit
  gate below — it writes fluent prose around whatever numbers it is given, so feed it
  the Canonical Values and re-run `verify_results.py` on anything it produces.
- **`academic-pipeline`** — orchestrator: research → write → integrity-check → review →
  revise → re-review → finalize. Use it for the end-to-end manuscript workflow. The
  "integrity check" stage must be backed by `reports/verify_results.py`, not just prose
  self-review.

(Also installed, available when needed: `academic-paper-reviewer`, `deep-research`.)

## UTILITY SKILLS (not for writing — keep separate from the above)

- **`pipeline-map`** — read-only. Renders **Mermaid diagrams of the code/data pipelines**
  (Phase A/B/C scripts ↔ artifacts ↔ models) via the `claude-mermaid` MCP, saved to
  `reports/diagrams/`. It does **NOT** write, format, or review the manuscript and reports
  **no result numbers** — so it never touches the audit gate or Canonical Values. Use the
  `academic-*` skills above for any prose; use `pipeline-map` only to draw the architecture.
  (MCP `mermaid` is configured in repo `.mcp.json`; live preview needs SSH `-L 3737:localhost:3737`.)

---

# THESIS INSTRUCTION BLOCK (permanent)

## 1. Prime directive — honest, audit-proof reporting
- **Every reported number must trace to a real run.** No hallucinated, estimated, or
  undocumented experimental results may appear in any report or the PDF. If a number
  cannot be traced to a `reports/run_logs/*.log` (or a cited paper), it does not go in.
- **Negative and null findings are the contribution.** This thesis is framed as a
  *rigorous audit of SSL-for-affect*. Report what did not work, measured, and reframe —
  do not hide or oversell. An honest "we tested X and it didn't help" beats a confident
  unsupported claim every time.
- **Measure, don't assume.** Do not write "X because Y" until Y has been tested. Multiple
  convenient explanations in this project turned out false on measurement (the clustering
  claim, the key-encoding hypothesis, the ProtoPNet-clusters-more prediction). Pre-register
  a pass mark before each ablation; state it; hold to it.
- **Pre-registered pass mark (standard):** a treatment "wins" only if it beats baseline by
  >1 fold-std on both axes, OR >2 fold-std on either axis, OR lifts minority per-quadrant
  R² from <0 to ≥0. Differences inside fold-std are reported as *ties*, not wins.

## 2. The audit gate
- `reports/verify_results.py` is the single source of truth for numerical consistency.
  It extracts headline numbers from every `reports/run_logs/*.log` and checks them against
  the values quoted in the report `.md` files + `generate_report_v3.py`.
- **Run it before every PDF regeneration.** Current state: **72 numbers, 0 mismatches**.
- Every new experiment: save its stdout to `reports/run_logs/<descriptive_name>.log`,
  then extend `verify_results.py` with a parser stanza, then re-audit.

## 3. Reporting workflow (do this after EVERY new result, before the next task)
1. Update `JOURNEY.md` — plain-English diary entry (what I did / why / what happened /
   what I learned). Mention superseded steps briefly ("previously tried X, then changed to
   Y because Z") rather than deleting them.
2. Update the relevant `reports/0X_*.md` section(s).
3. Mirror any table/number into `reports/generate_report_v3.py` (the PDF is generated from
   this script, NOT from the `.md` files — keep them in sync).
4. Append a dated entry to `reports/updater.md` (the technical changelog — never delete
   history; mark superseded findings inline with `[SUPERSEDED → … ; see Step N]`).
5. Save the run log to `reports/run_logs/`, extend + run `verify_results.py`.
6. Regenerate the PDF: `cd reports && /usr/bin/python3 generate_report_v3.py`.

## 4. Canonical values (correct — never contradict these)
- **Phase A harmonic mode = 0.673** (Krumhansl–Schmuckler, best layer 11). The "~100%"
  figure is a DISCARDED artifact of a degenerate label (`probe_key.py:40`, mean-chroma
  threshold) and must never appear as a result.
- **Latent Silhouette ≈ 0.19 Euclidean / ≈ 0.26 cosine** (held-out, single-MERT ≈ Enhanced).
  "Silhouette ≈ 0 / near-zero" is SUPERSEDED (it came from in-sample indices).
- **single-MERT ≈ Enhanced on Silhouette** (model effect ≈ 0; 0.269 ≈ 0.260 cosine). Any
  claim that "multi-encoder is more structured/clustered than single-encoder" is FALSE.
- **Silhouette is intrinsic, not architectural:** across regression+SupCR (0.26), CE-head
  λ=1.0 (0.29), and ProtoPNet classification (0.18), it stays 0.18–0.29 — never near the
  ≳0.5 of clean clusters. Emotion is a continuous V-A gradient (Russell); the 4 quadrants
  are analytical bins.
- **Best valence = Triple (0.5758)** — nominal lead, *within* fold-noise of Spec-only
  (0.5709) / Triple-bio (0.5706). **Best arousal = Enhanced (0.7182).**
- **Music2Emo PMEmo-only = 0.536 V / 0.777 A** (arXiv:2502.03979 Table III). The
  0.458/0.639 values are wrong/superseded. Our Triple valence 0.576 > both Music2Emo variants.
- **Audio ProtoPNet = 0.728 raw / 0.545 balanced acc** (held-out), beats the post-hoc
  4-centroid (0.462–0.506) and the majority baseline (0.611); Sad recall 0.17 → 0.69.
- **Cyclic key encoding tested → NULL** (ΔV −0.008, inside fold-std). The encoding was not
  the valence bottleneck; key's link to valence is intrinsically weak at this data scale.
  Correct implementation: `models_enhanced.build_gap_vector(cyclic_key=True)`.
- **Precision@5 ≈ 0.58** (≈2× the 0.276 random baseline) is the protocol-robust evidence of
  emotional organization — lead with it, not Silhouette.
- **PMEmo minority quadrant n:** Calm/HVLA = 67, Angry/LVHA = 64, Sad/LVLA = 167 (HVHA = 469,
  ~61% majority). Per-quadrant R² is negative for all three minorities across all configs —
  a dataset-size floor, confirmed by imbalance ablation + mixup + CE-head (not a method bug).

## 5. Environment / commands
- Project venv python (CUDA, torch, librosa): `/home/arvin/thesis/mert/MERT/.venv/bin/python`.
- PDF generation needs reportlab → use system `/usr/bin/python3` for
  `generate_report_v3.py` only.
- 2 GPUs available; parallelise with `CUDA_VISIBLE_DEVICES`.
- Phase B scripts run from `MERT/ssl_scripts/phaseB/`; Phase C from `phaseC/`.
- Background runs: redirect stdout to `reports/run_logs/<name>.log`; do NOT rely on
  ephemeral `/tmp` task files for anything that must persist.

## 6. Style
- Terse, scientifically precise responses; academic rigor; no overstatement.
- Code matches surrounding conventions; honest comments.
- When a result is in, state it plainly with the number; when something is null/negative,
  say so and reframe; never hedge a verified result.

---

*Files: `reports/00–05_*.md` (sectioned report), `reports/generate_report_v3.py` (PDF),
`reports/updater.md` (changelog), `reports/run_logs/` (raw run outputs + AUDIT.md),
`reports/verify_results.py` (audit gate), `JOURNEY.md` (narrative diary),
`MERT/ssl_scripts/phaseA|B|C/` (code).*
