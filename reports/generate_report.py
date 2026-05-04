"""
Generates a 2-3 page professional PDF progress report for professor meeting.
Run: python3 generate_report.py
Output: mert_progress_report_v2.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import FrameBreak, PageBreak
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SSL  = os.path.join(BASE, "../MERT/ssl_scripts")
IMG_TSNE    = os.path.join(SSL, "phaseA/mert_emotion_clusters.png")
IMG_LAYERS  = os.path.join(SSL, "phaseB/layer_weights_kfold.png")
IMG_LAYPROB = os.path.join(SSL, "phaseB/archive/layer_performance.png")
OUT = os.path.join(BASE, "mert_progress_report_v2.pdf")

W, H = A4
MARGIN = 1.8 * cm

# ── Color Palette ──────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B3A6B")
TEAL   = colors.HexColor("#2E86AB")
LGRAY  = colors.HexColor("#F5F6FA")
MGRAY  = colors.HexColor("#D1D5DB")
DKGRAY = colors.HexColor("#374151")
WHITE  = colors.white
ACCENT = colors.HexColor("#10B981")

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

TITLE   = style("ReportTitle",  fontSize=15, textColor=NAVY, spaceAfter=2,
                leading=19, fontName="Helvetica-Bold")
SUBTITLE= style("Subtitle",    fontSize=9.5, textColor=TEAL, spaceAfter=1,
                fontName="Helvetica")
META    = style("Meta",        fontSize=8.5, textColor=DKGRAY, spaceAfter=6,
                fontName="Helvetica")
H1      = style("H1",          fontSize=10, textColor=NAVY, spaceBefore=6,
                spaceAfter=2, fontName="Helvetica-Bold")
H2      = style("H2",          fontSize=9, textColor=TEAL, spaceBefore=4,
                spaceAfter=2, fontName="Helvetica-Bold")
BODY    = style("Body",        fontSize=8, textColor=DKGRAY, leading=12,
                spaceAfter=3, alignment=TA_JUSTIFY, fontName="Helvetica")
BULLET  = style("Bullet",      fontSize=8, textColor=DKGRAY, leading=12,
                leftIndent=10, spaceAfter=2, fontName="Helvetica",
                bulletIndent=4)
CAPTION = style("Caption",     fontSize=7, textColor=DKGRAY, alignment=TA_CENTER,
                spaceAfter=4, fontName="Helvetica-Oblique")
BADGE   = style("Badge",       fontSize=8,   textColor=WHITE, alignment=TA_CENTER,
                fontName="Helvetica-Bold")

# ── Table helpers ──────────────────────────────────────────────────────────────
def hdr_cell(txt):
    return Paragraph(f"<b>{txt}</b>", ParagraphStyle("hdr", fontSize=7.5,
        textColor=WHITE, alignment=TA_CENTER, fontName="Helvetica-Bold", leading=10))

def cell(txt, bold=False, center=False):
    fn = "Helvetica-Bold" if bold else "Helvetica"
    al = TA_CENTER if center else TA_LEFT
    return Paragraph(txt, ParagraphStyle("c", fontSize=7.5, textColor=DKGRAY,
        fontName=fn, alignment=al, leading=10))

def result_cell(txt):
    return Paragraph(f"<b>{txt}</b>", ParagraphStyle("rc", fontSize=8,
        textColor=ACCENT, fontName="Helvetica-Bold", alignment=TA_CENTER, leading=10))

def table_style(header_rows=1):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1, header_rows-1), NAVY),
        ("TEXTCOLOR",   (0,0), (-1, header_rows-1), WHITE),
        ("ROWBACKGROUNDS",(0,header_rows), (-1,-1), [WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.3, MGRAY),
        ("LINEBELOW",   (0, header_rows-1), (-1, header_rows-1), 1.0, TEAL),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0), (-1,-1), 5),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ])

def img(path, width, caption_text=None):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width*0.62))
        if caption_text:
            items.append(Paragraph(caption_text, CAPTION))
    return items

# ── Document ───────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN)

story = []
TW = W - 2 * MARGIN  # usable text width

# ══ HEADER ════════════════════════════════════════════════════════════════════
story.append(Paragraph("Explainable Music Emotion Recognition via<br/>Self-Supervised Music Representations (MERT)", TITLE))
story.append(Paragraph("Progress Report — May 2026", SUBTITLE))
story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=4))
story.append(Paragraph(
    "Student: <b>Arvin Jafari Moghadam Fard</b> &nbsp;|&nbsp; "
    "Programme: <b>MSc Computer Science — Music Information Retrieval</b> &nbsp;|&nbsp; "
    "Supervisor Meeting Report", META))

# ══ 1. INTRODUCTION ═══════════════════════════════════════════════════════════
story.append(Paragraph("1. Introduction", H1))
story.append(Paragraph(
    "This report presents the completed experimental phases of a thesis investigating self-supervised "
    "music representations for Music Emotion Recognition (MER). The MERT model — pre-trained on "
    "160,000 hours of music — provides rich multi-layer embeddings that serve as the backbone for "
    "emotion prediction and, ultimately, explainable music retrieval. The thesis addresses two open "
    "problems in MER simultaneously: the <b>valence prediction ceiling</b> present in all audio-only "
    "systems, and the <b>black-box problem</b> that limits deployment in trust-sensitive settings such "
    "as therapeutic music or mood-regulation applications.", BODY))

# ══ 2. DATASET ════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Dataset", H1))
story.append(Paragraph(
    "All experiments use the <b>PMEmo 2019</b> dataset — 794 pop songs annotated by 457 participants "
    "with continuous valence–arousal scores per chorus segment, plus synchronised EDA "
    "(electrodermal activity) physiological signals. After ID alignment, <b>767 songs</b> are used.", BODY))

# ══ 3. PIPELINE ═══════════════════════════════════════════════════════════════
story.append(Paragraph("3. Proposed Pipeline", H1))

pipeline_data = [
    [hdr_cell("Phase"), hdr_cell("Input"), hdr_cell("Method"), hdr_cell("Output")],
    [cell("A — Probing"),
     cell("Frozen MERT (25 × 1024)"),
     cell("Linear probe (classifier / regressor)"),
     cell("Validates foundation")],
    [cell("B — Affective Model"),
     cell("All-layer embeddings + EDA"),
     cell("WeightedFusion + HybridLoss + 5-fold CV"),
     cell("V-A predictions + latent space")],
    [cell("C — Explainability", bold=True),
     cell("Phase B latent space", bold=True),
     cell("Prototype retrieval + XAI explanation", bold=True),
     cell("Future work ▶", bold=True)],
]
pipeline_tbl = Table(pipeline_data,
    colWidths=[TW*0.15, TW*0.25, TW*0.37, TW*0.23])
pipeline_tbl.setStyle(table_style())
pipeline_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#E0F7FA")),
    ("TEXTCOLOR",  (0, 3), (-1, 3), NAVY),
]))
story.append(pipeline_tbl)
story.append(Spacer(1, 0.2*cm))

# ══ 4. PHASE A ════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Phase A — Representation Probing", H1))
story.append(Paragraph(
    "Before any emotion fine-tuning, we validated that MERT's frozen embeddings preserve "
    "music-theoretically meaningful information. A linear probe confirms that if a linear model "
    "succeeds, the information is <i>explicitly</i> encoded in the representation — not a "
    "learned artefact of fine-tuning.", BODY))

probeA_data = [
    [hdr_cell("Probe Task"), hdr_cell("Model"), hdr_cell("Result"), hdr_cell("Interpretation")],
    [cell("Major / Minor mode"),
     cell("Linear classifier (1024 → 1)"),
     result_cell("~100% accuracy"),
     cell("Harmonic structure is explicitly encoded")],
    [cell("Tempo (BPM)"),
     cell("Linear regressor (1024 → 1)"),
     result_cell("R² = 0.12"),
     cell("Rhythm partially encoded; non-linear probe needed")],
    [cell("Emotion geometry"),
     cell("t-SNE visualisation"),
     result_cell("Quadrant separation"),
     cell("Emotion topology exists before training")],
]
tA = Table(probeA_data, colWidths=[TW*0.22, TW*0.26, TW*0.18, TW*0.34])
tA.setStyle(table_style())
story.append(tA)
story.append(Spacer(1, 0.1*cm))
story.append(Paragraph(
    "<b>Conclusion:</b> MERT is a valid foundation. Harmonic awareness and emotional geometry "
    "exist in the raw representation — any improvements in Phase B are refinements, not "
    "compensating for a weak backbone.", BODY))

# ══ VISUALISATIONS — row 1 ════════════════════════════════════════════════════
story.append(Spacer(1, 0.1*cm))
half = (TW - 0.4*cm) / 2

tsne_items = img(IMG_TSNE, half,
    "Fig. 1 — t-SNE of MERT embeddings coloured by emotion quadrant\n"
    "(HVHA / HVLA / LVHA / LVLA). Quadrant separation is visible\nbefore any emotion fine-tuning.")
lw_items   = img(IMG_LAYERS, half,
    "Fig. 2 — Learnable MERT layer fusion weights after Phase B\ntraining. "
    "Layers 14, 16, 17 receive highest attribution.")

if tsne_items and lw_items:
    fig_row = [[tsne_items[0], lw_items[0]]]
    fig_tbl = Table(fig_row, colWidths=[half, half], hAlign="CENTER")
    fig_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(fig_tbl)
    cap_row = [[Paragraph(tsne_items[1].text if hasattr(tsne_items[1],'text') else
                          "Fig. 1 — t-SNE of MERT embeddings coloured by emotion quadrant.", CAPTION),
                Paragraph(lw_items[1].text if hasattr(lw_items[1],'text') else
                          "Fig. 2 — MERT layer fusion weights: layers 14,16,17 dominate.", CAPTION)]]
    cap_tbl = Table(cap_row, colWidths=[half, half])
    cap_tbl.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(cap_tbl)

# ══ 5. PHASE B ════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Phase B — Hybrid Affective Model", H1))
story.append(Paragraph("<b>Architecture:</b>", H2))
story.append(Paragraph(
    "A learnable <b>WeightedLayerFusion</b> applies a softmax over all 25 MERT layers, producing "
    "a fused 1024-dim representation. This is projected through a 1024→256→128 bottleneck head "
    "(LayerNorm, ReLU, Dropout) to a 128-dim emotion-aware latent space, then regressed to "
    "arousal and valence. EDA signals (7-dim statistical features) are fused via a late-fusion "
    "head (160→64→2), providing physiological grounding.", BODY))

story.append(Paragraph("<b>Training innovations:</b>", H2))
bullets = [
    "<b>Hybrid Loss (4 components):</b> MSE (1.0) + CCC (0.5) + Rank (0.3) + SupCR (0.1) — "
    "each targeting a distinct failure mode: absolute error, correlation bias, ordinal structure, "
    "and latent space organisation.",
    "<b>Differential Optimizer:</b> Fusion weights trained at lr = 10⁻² (overcomes frozen-weight "
    "initialisation); head/regressor at lr = 10⁻⁴. Resolves the 'frozen fusion weight' problem.",
    "<b>Balanced Sampler:</b> Sample weights inversely proportional to Russell quadrant frequency, "
    "resolving Simpson's Paradox in the quadrant-imbalanced PMEmo dataset.",
    "<b>5-Fold Cross-Validation</b> with CosineAnnealingLR over 100 epochs per fold.",
]
for b in bullets:
    story.append(Paragraph(f"• {b}", BULLET))
story.append(Spacer(1, 0.2*cm))

# Results table
story.append(Paragraph("<b>Validated Results (5-Fold CV):</b>", H2))
res_data = [
    [hdr_cell("Configuration"), hdr_cell("Arousal R²"), hdr_cell("Valence R²"),
     hdr_cell("CCC Arousal"), hdr_cell("CCC Valence")],
    [cell("Audio-Only (Hybrid)"),
     cell("0.6518", center=True), cell("0.5055", center=True),
     cell("0.82", center=True),   cell("0.74", center=True)],
    [cell("+ EDA Fusion (Full System)", bold=True),
     result_cell("0.6738"), result_cell("0.5075"),
     result_cell("0.8543"), result_cell("0.7692")],
]
rt = Table(res_data, colWidths=[TW*0.32, TW*0.17, TW*0.17, TW*0.17, TW*0.17])
rt.setStyle(table_style())
story.append(rt)
story.append(Spacer(1, 0.25*cm))

# SOTA table
story.append(Paragraph("<b>State-of-the-Art Comparison (PMEmo 2019):</b>", H2))
sota_data = [
    [hdr_cell("Method"), hdr_cell("Year"), hdr_cell("Approach"),
     hdr_cell("R² Valence"), hdr_cell("R² Arousal"), hdr_cell("CCC V/A")],
    [cell("PMEmo (Zhang et al.)"),
     cell("2019", center=True), cell("Hand-crafted IS13 + ML"),
     cell("~0.42", center=True), cell("~0.51", center=True), cell("—", center=True)],
    [cell("Deep MER (Dutta & Chanda)"),
     cell("2021", center=True), cell("Mel-Spec + CRNN"),
     cell("~0.45", center=True), cell("~0.55", center=True), cell("—", center=True)],
    [cell("Hybrid SSL Multi-task"),
     cell("2023", center=True), cell("Wav2Vec2 + Attention"),
     cell("~0.48", center=True), cell("~0.61", center=True), cell("—", center=True)],
    [cell("IAENG Hybrid"),
     cell("2025", center=True), cell("Multimodal Frequency-Domain"),
     cell("~0.60", center=True), cell("~0.62", center=True), cell("—", center=True)],
    [cell("This Work (MERT Hybrid+EDA)", bold=True),
     cell("2026", center=True), cell("Music-SSL + WeightedFusion + SupCR", bold=True),
     result_cell("0.5075"), result_cell("0.6738"), result_cell("0.77 / 0.85")],
]
st = Table(sota_data,
    colWidths=[TW*0.26, TW*0.08, TW*0.30, TW*0.12, TW*0.12, TW*0.12])
st.setStyle(table_style())
story.append(st)

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "<b>Note on CCC vs R²:</b> CCC (Concordance Correlation Coefficient) — the AVEC affective "
    "computing standard — simultaneously penalises correlation, mean shift, and variance mismatch. "
    "Our <b>CCC of 0.8543 (arousal)</b> reflects high-fidelity emotional tracking even where "
    "absolute R² is constrained by the audio-only valence ceiling.", BODY))

# Layer-wise probing image
lp_items = img(IMG_LAYPROB, TW * 0.62,
    "Fig. 3 — Layer-wise linear probing R² across all 25 MERT transformer layers. "
    "Arousal (blue) peaks at the final layer; valence (orange) shows progressive "
    "improvement in higher semantic layers, motivating the WeightedLayerFusion design.")
if lp_items:
    story.append(Spacer(1, 0.15*cm))
    story.append(KeepTogether([
        lp_items[0],
        Paragraph("Fig. 3 — Layer-wise linear probing R² across all 25 MERT transformer layers. "
                  "Arousal (blue) peaks at the final layer; valence (orange) motivates WeightedLayerFusion.", CAPTION),
    ]))

# ══ 6. RESEARCH GAP ═══════════════════════════════════════════════════════════
story.append(Paragraph("6. Research Gap & Honest Limitations", H1))
story.append(Paragraph(
    "Phase B achieves competitive performance but identifies two unresolved gaps:", BODY))
lim_data = [
    [hdr_cell("Limitation"), hdr_cell("Scientific Framing")],
    [cell("Valence R² ceiling at 0.51"),
     cell("Known MER ceiling — valence requires lyrics and cultural context absent from audio alone. "
          "CCC of 0.77 confirms agreement quality is high.")],
    [cell("MERT frozen (no backbone fine-tuning)"),
     cell("Downstream task trained on ~600 samples — fine-tuning 330M params risks overfitting "
          "without a larger annotated corpus.")],
    [cell("No explainability layer"),
     cell("The system produces V-A numbers but cannot explain *why* a song occupies a given "
          "emotional region. This is the primary thesis gap to close.")],
]
lt = Table(lim_data, colWidths=[TW*0.30, TW*0.70])
lt.setStyle(table_style())
story.append(lt)

# ══ 7. FUTURE WORK ════════════════════════════════════════════════════════════
story.append(Paragraph("7. Planned Next Phase — Explainable Emotion Retrieval", H1))
story.append(Paragraph(
    "The Phase B latent space is emotionally structured by design (via SupCR loss), making it a "
    "natural substrate for retrieval. The planned next phase builds an <b>XAI layer</b> on top "
    "of this latent space with the following components:", BODY))

fw_bullets = [
    "<b>Prototype-based retrieval:</b> k-NN search in the 128-dim latent space returns the k most "
    "emotionally similar songs. Retrieval via concrete exemplars is an ante-hoc explanation strategy "
    "(Case-Based Reasoning, Aamodt & Plaza 1994).",
    "<b>Contrastive foils:</b> Surface the most dissimilar songs alongside retrieved results, enabling "
    "the explanation 'why this song and not that one' — the most cognitively natural form of "
    "explanation for humans (Miller, 2019).",
    "<b>Physiological grounding:</b> EDA features provide biological convergent validity — explaining "
    "not just that songs are emotionally similar, but that listeners' bodies responded similarly "
    "(Thayer, 1989 biopsychological model).",
    "<b>Interpretable layer attribution:</b> The WeightedLayerFusion weights already reveal which "
    "acoustic feature levels (low/mid/high MERT layers) drove each retrieval decision.",
]
for b in fw_bullets:
    story.append(Paragraph(f"• {b}", BULLET))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "This positions the thesis contribution as unique: not just a model that predicts emotion, "
    "but a system that explains its reasoning in terms accessible to clinicians, music therapists, "
    "and end users — directly addressing the trust and transparency gap in affective AI.", BODY))

# ══ 8. CONCLUSION ══════════════════════════════════════════════════════════════
story.append(HRFlowable(width="100%", thickness=0.8, color=MGRAY, spaceBefore=6, spaceAfter=6))
story.append(Paragraph("8. Conclusion", H1))
story.append(Paragraph(
    "Phases A and B demonstrate that music-specific self-supervised representations (MERT) provide "
    "a strong, interpretable foundation for emotion prediction. The hybrid model achieves competitive "
    "performance on PMEmo 2019 (Arousal CCC = 0.8543), with the highest arousal R² of any "
    "audio-only method in the literature. The clear next step — and primary remaining thesis "
    "contribution — is the explainability layer that translates the learned emotional latent space "
    "into human-interpretable, psychologically grounded recommendations.", BODY))

# ══ REFERENCES ════════════════════════════════════════════════════════════════
story.append(Paragraph("References", H2))
refs = [
    "Li, Y. et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. <i>arXiv:2306.00107</i>.",
    "Zhang, K. et al. (2019). PMEmo: A Dataset with Physiological Signals for Music Emotion Recognition. <i>ACM ICMR</i>.",
    "Russell, J.A. (1980). A circumplex model of affect. <i>J. Personality & Social Psychology</i>, 39(6).",
    "Miller, T. (2019). Explanation in Artificial Intelligence: Insights from the social sciences. <i>Artificial Intelligence</i>, 267.",
    "Thayer, R.E. (1989). <i>The Biopsychology of Mood and Arousal</i>. Oxford University Press.",
    "Aamodt, A. & Plaza, E. (1994). Case-Based Reasoning: Foundational Issues, Methodological Variations. <i>AI Communications</i>, 7(1).",
]
for r in refs:
    story.append(Paragraph(f"• {r}", ParagraphStyle("ref", fontSize=7.5,
        textColor=DKGRAY, leading=11, leftIndent=10, spaceAfter=2, fontName="Helvetica")))

# ── Build ──────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"✅  PDF generated → {OUT}")
