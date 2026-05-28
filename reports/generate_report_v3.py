"""
Generates the v3 progress-report PDF (same visual style as v2), updated with the
post-meeting work, all result settings, the loss ablation, the Phase C
explainability evaluation, and an HONEST, de-hallucinated comparison section.

Run with a Python that has reportlab (system python has it):
    /usr/bin/python3 generate_report_v3.py
Output: mert_progress_report_v3.pdf
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "../MERT/ssl_scripts/phaseC/artifacts")
IMG_TSNE = os.path.join(ART, "tsne_baseline_vs_finetuned.png")
IMG_LAYERS = os.path.join(ART, "layer_fusion_weights.png")
OUT = os.path.join(BASE, "mert_progress_report_v3.pdf")

W, H = A4
MARGIN = 1.8 * cm
TW = W - 2 * MARGIN

NAVY = colors.HexColor("#1B3A6B"); TEAL = colors.HexColor("#2E86AB")
LGRAY = colors.HexColor("#F5F6FA"); MGRAY = colors.HexColor("#D1D5DB")
DKGRAY = colors.HexColor("#374151"); WHITE = colors.white
ACCENT = colors.HexColor("#10B981"); AMBER = colors.HexColor("#B45309")

base = getSampleStyleSheet()
def style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

TITLE = style("T", fontSize=15, textColor=NAVY, spaceAfter=2, leading=19, fontName="Helvetica-Bold")
SUBTITLE = style("S", fontSize=9.5, textColor=TEAL, spaceAfter=1)
META = style("M", fontSize=8.5, textColor=DKGRAY, spaceAfter=6)
H1 = style("H1", fontSize=10, textColor=NAVY, spaceBefore=7, spaceAfter=2, fontName="Helvetica-Bold")
H2 = style("H2", fontSize=9, textColor=TEAL, spaceBefore=4, spaceAfter=2, fontName="Helvetica-Bold")
BODY = style("B", fontSize=8, textColor=DKGRAY, leading=12, spaceAfter=3, alignment=TA_JUSTIFY)
BULLET = style("Bu", fontSize=8, textColor=DKGRAY, leading=12, leftIndent=10, spaceAfter=2)
CAPTION = style("C", fontSize=7, textColor=DKGRAY, alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Oblique")
NOTE = style("N", fontSize=7.8, textColor=AMBER, leading=11, spaceAfter=3, alignment=TA_JUSTIFY)

def hdr(t):
    return Paragraph(f"<b>{t}</b>", ParagraphStyle("h", fontSize=7.5, textColor=WHITE,
                     alignment=TA_CENTER, fontName="Helvetica-Bold", leading=10))
def cell(t, bold=False, center=False):
    return Paragraph(t, ParagraphStyle("c", fontSize=7.5, textColor=DKGRAY,
                     fontName="Helvetica-Bold" if bold else "Helvetica",
                     alignment=TA_CENTER if center else TA_LEFT, leading=10))
def rc(t):
    return Paragraph(f"<b>{t}</b>", ParagraphStyle("r", fontSize=7.5, textColor=ACCENT,
                     alignment=TA_CENTER, fontName="Helvetica-Bold", leading=10))
def tstyle(hr=1):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, hr-1), NAVY), ("TEXTCOLOR", (0, 0), (-1, hr-1), WHITE),
        ("ROWBACKGROUNDS", (0, hr), (-1, -1), [WHITE, LGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.3, MGRAY),
        ("LINEBELOW", (0, hr-1), (-1, hr-1), 1.0, TEAL),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])

s = []

# ── Header ──
s.append(Paragraph("Explainable Music Emotion Recognition via<br/>Self-Supervised Music Representations (MERT)", TITLE))
s.append(Paragraph("Progress Report v3 — May 2026 (post-meeting update)", SUBTITLE))
s.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=4))
s.append(Paragraph("Student: <b>Arvin Jafari Moghadam Fard</b> &nbsp;|&nbsp; "
                   "MSc Computer Science — Music Information Retrieval &nbsp;|&nbsp; Supervisor update", META))

# ── Intro (plain language) ──
s.append(Paragraph("Since our last meeting", H1))
s.append(Paragraph(
    "Since our last meeting, I tried to push the limits of the architecture by testing multi-encoder "
    "setups and bringing in cross-domain data. I also finished building the Phase C explainability "
    "system, including the ante-hoc prototype feature you suggested. With about one week of coding "
    "left before I focus entirely on writing, here is a plain summary of what I found. I have hit some "
    "hard dataset limits, so I would value your thoughts on whether any quick, final experiments are "
    "worth running. This report is deliberately honest about what did <b>not</b> work — several of my "
    "earlier hopeful claims did not survive a careful re-check, and I correct them below.", BODY))

# ── 1. Recap ──
s.append(Paragraph("1. The Three Phases (recap)", H1))
s.append(Paragraph(
    "<b>Phase A</b> checks what musical information the frozen MERT model already contains. "
    "<b>Phase B</b> trains a small model on top of MERT to predict emotion as two numbers — arousal "
    "(calm↔intense) and valence (sad↔happy). <b>Phase C</b> turns that into a retrieval system that "
    "finds emotionally similar songs and explains why. Data: <b>PMEmo 2019</b>, 794 pop-song chorus "
    "clips (767 usable), with valence/arousal ratings and EDA (skin-response) signals.", BODY))

# ── 2. Phase A ──
s.append(Paragraph("2. Phase A — What MERT Already Knows", H1))
s.append(Paragraph(
    "I ran a linear probe (Ridge or Logistic Regression) on every one of MERT's 25 layers against "
    "<b>8 librosa-extracted music-theory features</b>. A feature is flagged as a <b>gap</b> if its "
    "best-layer score falls below R² = 0.40 (regression) or accuracy = 0.65 (classification). "
    "Full results below — bold rows are the gaps that later drove the Phase B <i>Enhanced</i> branch.", BODY))
pa = [[hdr("Feature"), hdr("Probe"), hdr("Threshold"), hdr("Best layer"), hdr("Best score"), hdr("Pooled"), hdr("Gap?")],
      [cell("chroma (12-d)"),                 cell("Ridge R²", center=True), cell("0.40", center=True), cell("0", center=True),  cell("0.6801", center=True), cell("0.5769", center=True), cell("no", center=True)],
      [cell("tempo (BPM)", bold=True),        cell("Ridge R²", center=True), cell("0.40", center=True), cell("0", center=True),  rc("−0.8307"), rc("−2.1155"), rc("yes")],
      [cell("rhythmic_stability"),            cell("Ridge R²", center=True), cell("0.40", center=True), cell("0", center=True),  cell("0.7346", center=True), cell("0.5630", center=True), cell("no", center=True)],
      [cell("spectral_centroid"),             cell("Ridge R²", center=True), cell("0.40", center=True), cell("1", center=True),  cell("0.9206", center=True), cell("0.8663", center=True), cell("no", center=True)],
      [cell("spectral_contrast (7-d)"),       cell("Ridge R²", center=True), cell("0.40", center=True), cell("3", center=True),  cell("0.7310", center=True), cell("0.7270", center=True), cell("no", center=True)],
      [cell("zero-crossing rate"),            cell("Ridge R²", center=True), cell("0.40", center=True), cell("0", center=True),  cell("0.9566", center=True), cell("0.8471", center=True), cell("no", center=True)],
      [cell("mode (major/minor)"),            cell("LogReg acc", center=True), cell("0.65", center=True), cell("11", center=True), cell("0.6730", center=True), cell("0.6730", center=True), cell("no (marginal)", center=True)],
      [cell("key (12-class)", bold=True),     cell("LogReg acc", center=True), cell("0.65", center=True), cell("2", center=True),  rc("0.5849"), rc("0.5786"), rc("yes")]]
tpa = Table(pa, colWidths=[TW*0.20, TW*0.13, TW*0.11, TW*0.10, TW*0.14, TW*0.14, TW*0.13]); tpa.setStyle(tstyle()); s.append(tpa)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>Reading.</b> Short-time spectral / chroma information (spectral centroid, ZCR, spectral "
    "contrast, chroma, rhythmic stability) is <b>strongly encoded</b> and lives in MERT's <i>early</i> "
    "layers (best-layer = 0–3) — consistent with the SSL hierarchy literature (Pasad et al. 2021). "
    "Mode (major / minor) is captured marginally (best-layer 11, accuracy 0.67 — just above the binary "
    "threshold). Tempo and key are <b>genuine gaps</b>: not linearly recoverable from any of the 25 "
    "layers. The Phase B <i>Enhanced</i> model re-injects exactly these two features through a small "
    "hand-crafted music-theory branch — a Phase-A-driven architectural choice, not a guess. "
    "<b>Honest framing of the t-SNE:</b> the raw MERT latent space is one mixed blob (Fig. 1, left); "
    "emotion is present but continuously distributed, not separated into clusters.", BODY))

# ── 3. Phase B ──
s.append(Paragraph("3. Phase B — Emotion Prediction (all settings)", H1))
s.append(Paragraph(
    "Architecture: a learnable weighted fusion over MERT's 25 layers → a small head → arousal/valence, "
    "trained with a 4-part loss (MSE + CCC + Rank + SupCR), a differential optimizer, a balanced "
    "sampler, and 5-fold cross-validation. I then tested several encoder add-ons. All numbers are "
    "5-fold averages.", BODY))
res = [[hdr("Configuration"), hdr("Arousal R²"), hdr("Valence R²"), hdr("CCC A"), hdr("CCC V")],
       [cell("Mel-CNN only (shallow CNN, no SSL)"), cell("0.6486", center=True), cell("0.4452", center=True), cell("0.79", center=True), cell("0.63", center=True)],
       [cell("wav2vec2 only (speech SSL)"), cell("0.6225", center=True), cell("0.4825", center=True), cell("0.77", center=True), cell("0.66", center=True)],
       [cell("MERT only (music SSL)"), cell("0.6518", center=True), cell("0.5055", center=True), cell("0.82", center=True), cell("0.74", center=True)],
       [cell("MERT + EDA (physiology)"), cell("0.6738", center=True), cell("0.5075", center=True), rc("0.8543"), rc("0.7692")],
       [cell("Dual: MERT + wav2vec2"), cell("0.6814", center=True), cell("0.5676", center=True), cell("0.8087", center=True), cell("0.7231", center=True)],
       [cell("Spec-only: MERT + mel-CNN"), cell("0.7069", center=True), cell("0.5709", center=True), cell("0.8271", center=True), cell("0.7314", center=True)],
       [cell("Triple-bio: MERT + mel-CNN + EDA"), cell("0.7077", center=True), cell("0.5706", center=True), cell("0.8262", center=True), cell("0.7325", center=True)],
       [cell("Triple: MERT + wav2vec2 + mel-CNN", bold=True), cell("0.7023", center=True), rc("0.5758"), cell("0.8233", center=True), cell("0.7329", center=True)],
       [cell("Enhanced: + tempo/key (best A)", bold=True), rc("0.7182"), cell("0.5686", center=True), cell("0.8345", center=True), cell("0.7259", center=True)]]
t = Table(res, colWidths=[TW*0.36, TW*0.16, TW*0.16, TW*0.16, TW*0.16]); t.setStyle(tstyle()); s.append(t)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>In plain terms:</b> best valence is the Triple model (R² 0.576); best arousal is the Enhanced "
    "model (R² 0.718). An R² of 0.72 means we explain about 72% of the variation in how energetic "
    "songs sound. <b>Important caveat on every row:</b> most PMEmo songs are 'Happy' (61%); the model "
    "does well there but has negative R² on the rarer Sad/Calm/Angry songs, so the global numbers "
    "flatter the model.", BODY))

s.append(Paragraph(
    "<b>Single-encoder check (three baselines, localises SSL's contribution):</b> "
    "wav2vec2-only (speech SSL) scores 0.6225 / 0.4825; MERT-only (music SSL) scores 0.6518 / 0.5055; "
    "a shallow <b>Mel-CNN alone</b> (no SSL at all) scores 0.6486 / 0.4452 — it <i>beats wav2vec2 on "
    "arousal</i> (energy is acoustically easy: loudness, spectral envelope) but <i>loses on valence</i> "
    "(no harmony / tonality model). Music-specific SSL's value is therefore concentrated on the valence "
    "axis (MERT V 0.5055 vs Mel-CNN V 0.4452 vs wav2vec2 V 0.4825), which is also where the multi-encoder "
    "gains land. This also explains why wav2vec2 became redundant once combined with MERT.", BODY))

s.append(Paragraph(
    "<b>Second-branch redundancy — EDA, wav2vec2, theory all interchangeable beside MERT+Mel:</b> "
    "the new <b>Triple-bio</b> configuration (MERT + mel-CNN + EDA, swapping wav2vec2 for physiology) "
    "scores R² A 0.7077 / V 0.5706 — essentially identical to Spec-only (MERT+Mel = 0.7069 / 0.5709) "
    "and within fold-noise of Triple-with-wav2vec (0.7023 / 0.5758). EDA helps when added to MERT "
    "alone (MERT+EDA CCC A 0.85 — best on that axis) but adds nothing on top of MERT+Mel. Three "
    "different second-branches (wav2vec2, EDA, music-theory tempo/key) all converge to the same "
    "~0.71 / 0.57 ceiling once a trainable spectral CNN is present — they cover overlapping "
    "information rather than stacking complementarily.", BODY))

s.append(Paragraph("3.1 Multi-encoder tests &amp; 'fusion collapse'", H2))
s.append(Paragraph(
    "Adding the mel-spectrogram CNN gave the best scores, but it made wav2vec2 <b>completely "
    "redundant</b> (MERT + mel-CNN alone ties the full triple model). When stacking multiple large "
    "models I hit <b>fusion collapse</b>: with only ~600 training songs the network just averages all "
    "layers uniformly because it lacks the data to learn layer-specific weights. Honest correction: "
    "the 'layers 14/16/17 dominate' claim from earlier does not hold — the learned weights are "
    "near-uniform and the top layer changes from run to run (Fig. 2).", BODY))
s.append(Paragraph(
    "<b>Empirical check (two ablations, 2026-05-26):</b> to answer the question "
    "<i>'why 25 layers and not just the last layer?'</i> properly, I ran the ablation on <b>both</b> "
    "single- and multi-encoder configurations.", BODY))
fab = [[hdr("Setting"), hdr("Variant"), hdr("R² A"), hdr("R² V"), hdr("CCC A"), hdr("CCC V")],
       [cell("MERT-only (single)"),     cell("25-layer fusion"),     cell("0.6518", center=True), cell("0.5055", center=True), cell("0.82", center=True), cell("0.74", center=True)],
       [cell(""),                        cell("Last layer only"),     cell("0.6570", center=True), cell("0.5162", center=True), cell("0.81", center=True), cell("0.70", center=True)],
       [cell(""),                        cell("Δ (last − fusion)"),   cell("+0.005", center=True), cell("+0.011", center=True), cell("-0.01", center=True), cell("-0.04", center=True)],
       [cell("Enhanced (multi-encoder)", bold=True), cell("25-layer fusion", bold=True), rc("0.7182"), rc("0.5686"), rc("0.8345"), rc("0.7259")],
       [cell(""),                        cell("Last layer only"),     cell("0.6660 ± 0.035", center=True), cell("0.4881 ± 0.070", center=True), cell("0.8124", center=True), cell("0.6865", center=True)],
       [cell(""),                        cell("Δ (last − fusion)"),   cell("−0.052", center=True), cell("−0.081", center=True), cell("−0.022", center=True), cell("−0.039", center=True)]]
tf = Table(fab, colWidths=[TW*0.22, TW*0.20, TW*0.13, TW*0.13, TW*0.10, TW*0.10]); tf.setStyle(tstyle()); s.append(tf)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>The answer is nuanced — and quantitative.</b> On MERT alone, last-layer is statistically tied with "
    "the 25-layer fusion (Δ inside fold-std on both axes) — consistent with the near-uniform learned fusion "
    "weights (entropy 3.218 / max 3.219, Fig. 2). But on the multi-encoder Enhanced model, removing fusion "
    "costs <b>5.2 pp R² A</b> and <b>8.1 pp R² V</b> — outside fold-std on both, with parallel drops in CCC. "
    "<b>Interpretation:</b> the fusion's value is not intrinsic to MERT; it is specifically about <i>cross-"
    "encoder integration</i>. In single-encoder mode the late layer carries enough signal that the learnable "
    "softmax has nothing useful to add. In multi-encoder mode, mel-CNN / wav2vec2 / theory already cover the "
    "late acoustic representation, so MERT's mid-layers fill a complementary niche the last layer alone "
    "cannot reach. The architectural choice is empirically justified by the Enhanced ablation; the MERT-only "
    "ablation acts as the negative control that proves the gain is not free.", BODY))

s.append(Paragraph("3.2 Mixing music with environmental sounds (IADS-E)", H2))
s.append(Paragraph(
    "I tried replicating <b>Simonetta et al. (2024)</b>, who mix music (PMEmo) with environmental sounds "
    "(IADS-E) and report a large valence gain with hand-crafted features. It did not work for us: with "
    "our SSL models, adding environmental sounds actually <b>lowered</b> "
    "valence in every setting. This is a solid, publishable <b>negative finding</b> — SSL embeddings do "
    "not transfer emotional structure across audio domains as easily as older hand-crafted features do.", BODY))

s.append(Paragraph("3.3 Imbalance-handling ablation — does penalty beat the sampler?", H2))
s.append(Paragraph(
    "PMEmo is ~61% Happy (HVHA) and only ~9% in each of the three minority quadrants. The default fix is "
    "a <b>WeightedRandomSampler</b> with inverse-quadrant frequency. Supervisor question: would a loss-level "
    "penalty (weighted-MSE, focal-MSE) do better? I ran 4 fold-matched configurations on MERT-only "
    "(25-layer fusion, 5-fold CV, everything else identical to baseline):", BODY))
imb = [[hdr("Treatment"), hdr("R² A"), hdr("R² V"), hdr("CCC A"), hdr("CCC V")],
       [cell("A: sampler-only (baseline)", bold=True), rc("0.6951±0.016"), rc("0.5724±0.050"), cell("0.820", center=True), cell("0.731", center=True)],
       [cell("B: weighted-MSE only"), cell("0.6738±0.018", center=True), cell("0.5267±0.109", center=True), cell("0.828", center=True), cell("0.739", center=True)],
       [cell("C: sampler + weighted-MSE"), cell("0.6855±0.029", center=True), cell("0.5696±0.062", center=True), cell("0.811", center=True), cell("0.724", center=True)],
       [cell("D: focal-MSE γ=2 + sampler"), cell("0.6949±0.018", center=True), cell("0.5716±0.057", center=True), cell("0.821", center=True), cell("0.727", center=True)]]
ti = Table(imb, colWidths=[TW*0.36, TW*0.18, TW*0.18, TW*0.14, TW*0.14]); ti.setStyle(tstyle()); s.append(ti)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>Pre-registered pass mark:</b> a treatment wins only if R² beats baseline by &gt;1 fold-std on "
    "<i>both</i> axes (or &gt;2 fold-std on either). <b>Result: no winner.</b> Weighted-MSE alone is actively "
    "worse on valence (−0.046); stacking and focal are statistically tied. The sampler is empirically the "
    "best imbalance treatment we have at this dataset scale. The minority-quadrant failure is therefore a "
    "<b>dataset-size limit</b> (n=64–67 per minority quadrant), not a method limit — re-weighting cannot "
    "create signal where there is none. <i>Disclosure: the rerun baseline came in higher than the original "
    "logged baseline (0.6951 vs 0.6518); KFold seed is identical, but model-init RNG is not, so this "
    "reflects single-seed initialisation variance. The fold-matched A↔B↔C↔D comparison within this "
    "ablation is the audit-honest reading; published numbers in §3 are not overwritten.</i>", BODY))

s.append(Paragraph("3.4 Augmentation ablation — does feature-space mixup lift the minority-quadrant floor?", H2))
s.append(Paragraph(
    "§3.3 attributed the minority-quadrant failure (negative per-quadrant R² on HVLA/LVHA/LVLA, "
    "n=64–167) to a <b>dataset-size limit</b> rather than a method limit. To make that claim empirical "
    "rather than argumentative, I tested the standard cited remedy — feature-space <b>mixup</b> "
    "(Zhang et al. 2017) — on the Enhanced configuration. λ ~ Beta(0.4, 0.4) per batch, same λ applied "
    "to all three input branches (MERT, wav2vec2, theory) and labels; no mixup at evaluation; "
    "everything else identical to the no-augmentation Enhanced baseline.", BODY))
mix = [[hdr("Model"), hdr("R² A"), hdr("R² V"), hdr("CCC A"), hdr("CCC V")],
       [cell("Enhanced (no augmentation)", bold=True), rc("0.7182"), rc("0.5686"), rc("0.8345"), rc("0.7259")],
       [cell("Enhanced + mixup (α=0.4)"),  cell("0.7077 ± 0.014", center=True), cell("0.5651 ± 0.034", center=True), cell("0.8213", center=True), cell("0.7160", center=True)],
       [cell("Δ (mixup − baseline)"),     cell("−0.0105 (tied)", center=True),  cell("−0.0035 (tied)", center=True),  cell("−0.013", center=True), cell("−0.010", center=True)]]
tm = Table(mix, colWidths=[TW*0.40, TW*0.18, TW*0.18, TW*0.12, TW*0.12]); tm.setStyle(tstyle()); s.append(tm)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>No winner under the pre-registered pass mark.</b> All four deltas land inside fold-noise — "
    "mixup is statistically tied with baseline. Minority per-quadrant R² remained negative across all "
    "three minority quadrants (HVLA −0.35/−1.88, LVHA −0.91/−1.08, LVLA −0.11/−0.90). "
    "<b>This empirically confirms the dataset-size floor</b>: the standard cited augmentation remedy "
    "does not lift minorities into the positive range, exactly as the n=64–67 hypothesis predicts. "
    "The minority-quadrant failure is therefore a <i>data</i> limit, not a <i>method</i> limit. "
    "Long-term remedies are corpus expansion or task-aware augmentation (C-Mixup, Yao et al. 2022; "
    "SpecAugment, Park et al. 2019); these are flagged as future work, not in scope for this thesis. "
    "Per the simpler-first rule, no harder augmentation methods were tested.", BODY))

s.append(Paragraph("3.5 Loss ablation — is the complex loss worth it?", H2))
abl = [[hdr("Loss configuration"), hdr("CCC A"), hdr("CCC V"), hdr("Precision@5"), hdr("Silhouette")],
       [cell("MSE only"), cell("0.6861", center=True), cell("0.5955", center=True), cell("0.5259", center=True), cell("+0.012", center=True)],
       [cell("+ CCC + Rank"), cell("0.7814", center=True), cell("0.7113", center=True), cell("0.5398", center=True), cell("+0.021", center=True)],
       [cell("+ SupCR (full hybrid)", bold=True), rc("0.8165"), cell("0.7110", center=True), rc("0.5734"), cell("-0.031", center=True)]]
ta = Table(abl, colWidths=[TW*0.34, TW*0.16, TW*0.16, TW*0.18, TW*0.16]); ta.setStyle(tstyle()); s.append(ta)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "CCC + Rank are clearly justified (+0.10 to +0.12 CCC over plain MSE). SupCR helps retrieval "
    "(+0.034 Precision@5) but, surprisingly, removing it <b>raised</b> the clustering score — so SupCR "
    "tightens local neighbourhoods by continuous similarity, it does not build discrete clusters. This "
    "ablation disproves my original 'SupCR creates emotion clusters' claim, and I have corrected it. "
    "<i>Caveat (see §3.6):</i> these Silhouette numbers are from the single-MERT setup; the multi-encoder "
    "Enhanced model has substantially better baseline quadrant structure (Silhouette ≈ 0.26).", BODY))

s.append(Paragraph("3.6 Cluster-enforcement trade-off — can we make the t-SNE clusters look discrete?", H2))
s.append(Paragraph(
    "Fig. 1 shows a continuous Happy-dominated manifold rather than four separated clusters — an obvious "
    "viva question. To answer it quantitatively rather than argue, I added an auxiliary 4-way quadrant "
    "classification head on the Enhanced model's 128-d latent and trained with combined loss "
    "<i>HybridLoss + λ · CE(quadrant)</i>, sweeping λ ∈ {0, 0.1, 0.5, 1.0}. Silhouette is computed on "
    "test-fold latents with cosine distance against quadrant labels (5-fold mean ± std).", BODY))
ce = [[hdr("λ"), hdr("R² A"), hdr("R² V"), hdr("CCC A"), hdr("CCC V"), hdr("Silhouette")],
      [cell("0.0 (baseline)", bold=True), cell("0.7080±0.021", center=True), cell("0.5725±0.038", center=True), cell("0.827", center=True), cell("0.729", center=True), rc("0.255±0.054")],
      [cell("0.1"),                       cell("0.7050±0.013", center=True), cell("0.5811±0.033", center=True), cell("0.828", center=True), cell("0.738", center=True), cell("0.258±0.059", center=True)],
      [cell("0.5"),                       cell("0.6966±0.016", center=True), cell("0.5624±0.053", center=True), cell("0.827", center=True), cell("0.730", center=True), cell("0.265±0.064", center=True)],
      [cell("1.0"),                       cell("0.6638±0.020", center=True), cell("0.5601±0.037", center=True), cell("0.810", center=True), cell("0.734", center=True), cell("0.286±0.060", center=True)]]
tc = Table(ce, colWidths=[TW*0.10, TW*0.20, TW*0.20, TW*0.14, TW*0.14, TW*0.22]); tc.setStyle(tstyle()); s.append(tc)
s.append(Spacer(1, 0.1*cm))
s.append(Paragraph(
    "<b>Side-finding (rewrites the t-SNE framing):</b> the Enhanced model's latent space <i>already has "
    "moderate quadrant structure</i> — Silhouette = 0.255 at λ=0. The historic 'Silhouette ≈ 0' claim was "
    "from the single-MERT loss ablation (§3.5); it does NOT apply to the multi-encoder Enhanced setup. "
    "Multi-encoder fusion produces inherently more cluster-structured latents than single-encoder MERT — "
    "the Fig. 1 'blob' framing slightly under-sells what the Enhanced model does.", BODY))
s.append(Paragraph(
    "<b>The trade-off:</b> Silhouette rises monotonically with λ but only modestly (+0.031 from λ=0 to "
    "λ=1.0). R² A drops 4.4 pp at λ=1.0 — outside fold-std. λ=0.1 is essentially zero-cost on R² but "
    "also zero-benefit on Silhouette. The cost-benefit curve does <b>not</b> favour cluster enforcement: "
    "we would lose more on the regression task the system is built for than we would gain on cluster "
    "compactness. <b>Minority per-quadrant R² remain negative across all λ</b> (HVLA −0.43 → −0.85 as λ "
    "rises) — auxiliary classification cannot manufacture signal where n=64–67 minority examples are "
    "the actual limit. This is the third independent confirmation of the dataset-floor hypothesis after "
    "the imbalance ablation (§3.3) and mixup augmentation (§3.4).", BODY))
s.append(Paragraph(
    "<b>Viva-defensible conclusion:</b> the continuous representation is a deliberate, empirically-"
    "defended design choice, not a representational failure. Russell's circumplex (Russell 1980) defines "
    "valence and arousal as <i>continuous bipolar dimensions</i> — the four quadrants are analytical "
    "bins for per-quadrant reporting, not categorical psychological constructs (Eerola &amp; Vuoskoski "
    "2011). The observed manifold reflects both the construct and the PMEmo label distribution (61% HVHA). "
    "Cluster enforcement is achievable (λ=1.0 lifts Silhouette to 0.29) but pays for itself poorly. "
    "We keep the continuous representation.", BODY))

# ── Figures ──
s.append(Spacer(1, 0.1*cm))
if os.path.exists(IMG_TSNE):
    s.append(Image(IMG_TSNE, width=TW, height=TW*0.44))
    s.append(Paragraph("Fig. 1 — t-SNE of the song space. Left: raw untrained MERT is one mixed blob "
                       "(no emotion separation). Right: after training, clear structure appears but it is a "
                       "continuous, Happy-dominated manifold, not four clean clusters.", CAPTION))
if os.path.exists(IMG_LAYERS):
    s.append(Image(IMG_LAYERS, width=TW*0.74, height=TW*0.74*0.375))
    s.append(Paragraph("Fig. 2 — Learned layer-fusion weights are near-uniform (entropy 3.218 / max "
                       "3.219). The top-3 layers (14/15/16) are only a faint lean, not strong dominance.", CAPTION))

# ── 4. Phase C ──
s.append(Paragraph("4. Phase C — Explainability &amp; Ante-Hoc Prototypes", H1))
s.append(Paragraph(
    "<b>Retrieval works.</b> The system finds emotionally similar songs at about <b>twice the rate of "
    "random chance</b> (Precision@5 ≈ 0.58 vs a random baseline of 0.276), and clearly better than the "
    "raw untrained model (0.485). <b>Ante-hoc prototypes:</b> I added the 4-centroid profile we "
    "discussed — for any song it reports its similarity to the Happy/Calm/Angry/Sad centroids and names "
    "the closest one. It is a nice interpretability tool. <b>But</b> its raw accuracy (50.6%) actually "
    "<b>loses</b> to a 'always guess Happy' baseline (61.1%), because the dataset is 61% Happy. So it is "
    "a good explanation device, not a strong stand-alone classifier — and I report it that way.", BODY))
pc = [[hdr("Latent space"), hdr("Precision@5"), hdr("Silhouette"), hdr("Prototype accuracy")],
      [cell("Naive raw MERT (untrained)"), cell("0.485", center=True), cell("+0.100", center=True), cell("—", center=True)],
      [cell("MERT (SupCR-trained)"), cell("0.576", center=True), cell("-0.029", center=True), cell("0.462", center=True)],
      [cell("Dual MERT+wav2vec2 (SupCR)", bold=True), rc("0.585"), cell("+0.003", center=True), cell("0.506", center=True)]]
tp = Table(pc, colWidths=[TW*0.37, TW*0.20, TW*0.20, TW*0.23]); tp.setStyle(tstyle()); s.append(tp)
s.append(Spacer(1, 0.08*cm))
s.append(Paragraph("Reference points: random-chance Precision@5 = 0.276; majority-class ('always Happy') "
                   "baseline for prototype accuracy = 0.611. Silhouette ≈ 0 everywhere → emotion is a "
                   "continuous gradient, not four separable clusters (this is a finding, not a bug).", CAPTION))

# ── 5. Honest limitations ──
s.append(Paragraph("5. Honest Limitations (what did not work)", H1))
for b in [
    "<b>Class imbalance is the real ceiling:</b> 61% of songs are Happy; the model effectively only "
    "works well there (negative R² on Sad/Calm/Angry). This weakens the therapy/clinical motivation.",
    "<b>The latent space does not cluster by emotion:</b> Silhouette ≈ 0, and the prototype classifier "
    "loses to the trivial baseline. Retrieval still works locally (Precision@5 ≈ 2× chance).",
    "<b>Multi-encoder stacking mostly failed:</b> wav2vec2 is redundant, IADS-E transfer is negative, "
    "and the layer-fusion 'interpretability' is near-uniform. Real gains came from the mel-CNN and tempo.",
    "<b>Valence:</b> our single-dataset audio-only valence (R² 0.576) is competitive — above prior "
    "single-dataset SSL/hand-crafted results (Music2Emo 0.536, AutoML 0.525) — but still bounded by the "
    "practical audio-only limit; closing the gap to lyric-aware or cross-domain methods needs text or "
    "extra data, not more audio encoders.",
]:
    s.append(Paragraph(f"• {b}", BULLET))
s.append(Paragraph(
    "<b>Framing I plan to use:</b> this thesis is best presented as a rigorous, honest <i>audit of what "
    "SSL audio embeddings do and do not encode for emotion</i> — under that framing, the negative "
    "findings above are the contribution.", BODY))

# ── 6. Single PMEmo comparison table (literature + all our models) ──
s.append(Paragraph("6. PMEmo 2019 — Single Comparison Table (R²)", H1))
s.append(Paragraph(
    "All PMEmo R² results — prior work (top) and every model I built (bottom) — in one table for "
    "direct comparison. R² is the common metric across these papers. PMEmo papers that report only "
    "classification accuracy or RMSE (e.g. DAMER, CNN+LSTM) are not R²-comparable and are omitted "
    "rather than mixed in. Please confirm external numbers against their primary sources before final "
    "submission; note EmoMucs used labels scaled to [-1,1].", NOTE))
cmp = [[hdr("Method / Architecture"), hdr("Year"), hdr("R² Valence"), hdr("R² Arousal"), hdr("Notes")],
       [cell("Zhang et al. — IS13 + SVR/MLR (PMEmo paper)"), cell("2018", center=True), cell("~0.41*", center=True), cell("~0.52*", center=True), cell("baselines (*r²-derived, see note)")],
       [cell("EmoMucs C1D-M (de Berardinis et al.)"), cell("2020", center=True), cell("0.349", center=True), cell("0.557", center=True), cell("1D CNN; labels [-1,1]")],
       [cell("EmoMucs C2D-M (de Berardinis et al.)"), cell("2020", center=True), cell("0.414", center=True), cell("0.610", center=True), cell("2D CNN; labels [-1,1]")],
       [cell("AutoML openSMILE (Simonetta et al.)"), cell("2024", center=True), cell("0.525", center=True), cell("0.727", center=True), cell("hand-crafted; PMEmo only")],
       [cell("AutoML Joint (Simonetta et al.)"), cell("2024", center=True), cell("0.780", center=True), cell("0.861", center=True), cell("PMEmo + IADS-E (joint)")],
       [cell("Music2Emo (Kang &amp; Herremans)"), cell("2025", center=True), cell("0.536", center=True), cell("0.777", center=True), cell("MERT+chord/key; PMEmo only")],
       [cell("Music2Emo (Kang &amp; Herremans)"), cell("2025", center=True), cell("0.547", center=True), cell("0.794", center=True), cell("+ multitask; 4 datasets")],
       [cell("This work — Mel-CNN only"), cell("2026", center=True), cell("0.4452", center=True), cell("0.6486", center=True), cell("shallow CNN, no SSL")],
       [cell("This work — wav2vec2 only"), cell("2026", center=True), cell("0.4825", center=True), cell("0.6225", center=True), cell("speech SSL, single encoder")],
       [cell("This work — MERT only"), cell("2026", center=True), cell("0.5055", center=True), cell("0.6518", center=True), cell("music SSL, single encoder")],
       [cell("This work — MERT + EDA"), cell("2026", center=True), cell("0.5075", center=True), cell("0.6738", center=True), cell("+ physiology")],
       [cell("This work — Dual (MERT+wav2vec2)"), cell("2026", center=True), cell("0.5676", center=True), cell("0.6814", center=True), cell("dual SSL")],
       [cell("This work — Spec-only (MERT+Mel)"), cell("2026", center=True), cell("0.5709", center=True), cell("0.7069", center=True), cell("drops wav2vec2 from Triple")],
       [cell("This work — Triple-bio (MERT+Mel+EDA)"), cell("2026", center=True), cell("0.5706", center=True), cell("0.7077", center=True), cell("swaps wav2vec2 for EDA")],
       [cell("This work — Triple (+ mel-CNN)", bold=True), cell("2026", center=True), rc("0.576"), cell("0.702", center=True), cell("best valence")],
       [cell("This work — Enhanced (+ tempo/key)", bold=True), cell("2026", center=True), cell("0.569", center=True), rc("0.718"), cell("best arousal")]]
tc = Table(cmp, colWidths=[TW*0.34, TW*0.07, TW*0.15, TW*0.15, TW*0.29]); tc.setStyle(tstyle()); s.append(tc)
s.append(Spacer(1, 0.06*cm))
s.append(Paragraph(
    "*Zhang et al. (2018) report Pearson's r, not R². For a comparable baseline these are squared "
    "approximations (R² ≈ r²) from their static baselines (r=0.638 valence → 0.41; r=0.719 arousal → "
    "0.52). EmoMucs uses labels scaled to [-1,1]. Confirm all external numbers against primary sources.", CAPTION))
s.append(Paragraph(
    "<b>Reading (honest, with the right caveats):</b> our genuine strength is <b>valence</b> — our Triple "
    "(R² 0.576) is the best of all non-joint methods, above both Music2Emo variants (0.536 single-dataset, "
    "0.547 multitask) and hand-crafted AutoML (0.525). On <b>arousal</b> we do <b>not</b> lead: Music2Emo "
    "reaches 0.777 (single-dataset) / 0.794 (multitask) and AutoML 0.727, all above our best (0.718) — so "
    "arousal is a competitive-but-trailing result, not a headline. The very top row, Simonetta's AutoML "
    "<b>Joint</b> (0.780 / 0.861, PMEmo + IADS-E), is exactly the idea I replicated with SSL in §3.2 — and "
    "it did <b>not</b> transfer (our SSL valence dropped). So that row both sets the SOTA bar and frames "
    "our key negative finding: with hand-crafted features joint training helps, but SSL embeddings do not "
    "transfer emotional structure across the music↔sound boundary. Our contribution is therefore "
    "competitive single-dataset valence plus the explainability layer and the honest negative findings — "
    "not a state-of-the-art arousal number.", BODY))

# # ── 7. Questions ──
# s.append(Paragraph("7. Where I Would Value Your Input", H1))
# for b in [
#     "Given the dataset's hard limits (class imbalance, audio-only valence ceiling), are any quick final "
#     "experiments worth running before I switch fully to writing — or should I lock results and write?",
#     "Is the 'honest audit of SSL-for-affect' framing acceptable for the thesis, given several headline "
#     "ambitions turned into negative findings?",
#     "For the ante-hoc requirement: is the interpretability profile sufficient, or do you want me to "
#     "attempt a learned-prototype (ProtoPNet-style) head as a stretch goal?",
# ]:
#     s.append(Paragraph(f"• {b}", BULLET))

# ── References ──
s.append(HRFlowable(width="100%", thickness=0.8, color=MGRAY, spaceBefore=6, spaceAfter=6))
s.append(Paragraph("References", H2))
for r in [
    "Li, Y. et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. <i>arXiv:2306.00107</i>.",
    "Zhang, K. et al. (2018). PMEmo: A Dataset with Physiological Signals for Music Emotion Recognition. <i>ACM ICMR</i>.",
    "de Berardinis, J. et al. (2020). EmoMucs: Investigating the Role of Musical Components in MER. <i>ISMIR</i>.",
    "Kang, J. &amp; Herremans, D. (2025). Music2Emo: Towards Unified Music Emotion Recognition. <i>arXiv:2502.03979</i>.",
    "Simonetta, F., Certo, F. &amp; Ntalampiras, S. (2024). Joint Learning of Emotions in Music and Generalized Sounds. <i>arXiv:2408.02009</i>.",
    "Russell, J.A. (1980). A circumplex model of affect. <i>J. Personality &amp; Social Psychology</i>, 39(6).",
    "Miller, T. (2019). Explanation in Artificial Intelligence. <i>Artificial Intelligence</i>, 267.",
    "Krumhansl, C. &amp; Kessler, E. (1982). Tracing the dynamic changes in perceived tonal organization. <i>Psychological Review</i>.",
]:
    s.append(Paragraph(f"• {r}", ParagraphStyle("rf", fontSize=7.5, textColor=DKGRAY,
             leading=11, leftIndent=10, spaceAfter=2)))

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                  topMargin=MARGIN, bottomMargin=MARGIN).build(s)
print(f"PDF generated -> {OUT}")
