"""
music_theory_annotator.py — Phase C standalone music-theory annotator
======================================================================
Standalone, ML-free: librosa + a hardcoded lookup table. No dependency
on Phase B outcome, no import of any other phase. Existing Phase C files
are NOT modified (per the project constraint); instead this module
exposes a ready-to-use text block and an integration snippet below.

`librosa.estimate_key()` does not exist — key/mode use the canonical
Krumhansl–Schmuckler profile correlation (Krumhansl & Kessler 1982).

Public API
----------
    annotate(song_path) -> dict
        {key, tempo, brightness, rhythmic_stability, dominant_pitches}
    format_music_theory_block(annotation) -> str
        The multi-line "Music theory grounding:" block for explanations.

Standalone CLI:
    python music_theory_annotator.py /datasets/emotions/PMEmo2019/chorus/100.mp3

────────────────────────────────────────────────────────────────────────
INTEGRATION SNIPPET (no existing file is modified by this module).
To surface the block in Phase C output, add these 2 lines where the
query song path is known in phaseC/mainC.py (after the query is loaded):

    from explainability.music_theory_annotator import annotate, format_music_theory_block
    print(format_music_theory_block(annotate(query_audio_path)))

Or, to fold it into ExplainableRAG.generate_template() output, append
its return value with the same two lines at the call site in mainC.py —
explainer.py itself stays untouched.
────────────────────────────────────────────────────────────────────────
"""

import sys

import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_key_mode(chroma_mean: np.ndarray):
    """Krumhansl–Schmuckler key finding → (key 0..11, mode 1=major/0=minor)."""
    c = chroma_mean - chroma_mean.mean()
    if np.allclose(c, 0):
        return 0, 1
    best = (-2.0, 0, 1)
    for tonic in range(12):
        for profile, mode in ((_KK_MAJOR, 1), (_KK_MINOR, 0)):
            p = np.roll(profile, tonic).astype(float)
            p -= p.mean()
            denom = np.linalg.norm(c) * np.linalg.norm(p)
            corr = float(np.dot(c, p) / denom) if denom > 0 else -2.0
            if corr > best[0]:
                best = (corr, tonic, mode)
    return best[1], best[2]


def _brightness(norm_centroid: float) -> str:
    """Fixed-threshold approximation of the dataset spectral-centroid
    tertiles (standalone single-song → no dataset distribution available)."""
    if norm_centroid >= 0.18:
        return "bright"
    if norm_centroid <= 0.08:
        return "warm"
    return "moderate"


def annotate(song_path: str) -> dict:
    """Run librosa on one song and return a simple music-theory dict."""
    y, sr = librosa.load(song_path, sr=None)

    chroma = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)  # (12,)
    key, mode = _estimate_key_mode(chroma)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = int(round(float(np.atleast_1d(tempo).ravel()[0])))

    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    brightness = _brightness(cent / (sr / 2.0))

    tg = librosa.feature.tempogram(y=y, sr=sr)
    tg_mean = float(np.mean(tg))
    rhythmic_stability = round(1.0 - (float(np.std(tg)) / tg_mean), 3) if tg_mean else 0.0

    top3 = np.argsort(chroma)[::-1][:3]
    dominant_pitches = [NOTE_NAMES[i] for i in top3]

    return {
        "key": f"{NOTE_NAMES[key]} {'major' if mode else 'minor'}",
        "tempo": tempo,
        "brightness": brightness,
        "rhythmic_stability": rhythmic_stability,
        "dominant_pitches": dominant_pitches,
        "_mode": mode,  # internal: drives character lookup
    }


def _character(mode: int, brightness: str, tempo: int) -> str:
    """Hardcoded mode→character mapping (no ML)."""
    if mode == 1:  # major
        return "bright, energetic" if brightness == "bright" else "warm, gentle"
    # minor
    return "tense, urgent" if tempo >= 100 else "dark, introspective"


def _tempo_phrase(tempo: int) -> str:
    if tempo < 76:
        return "slow — spacious, relaxed"
    if tempo <= 120:
        return "moderate — neither restless nor static"
    return "fast — driving, restless"


def _timbre_phrase(brightness: str) -> str:
    return {
        "bright": "high spectral centroid — present, articulate instruments",
        "warm": "low spectral centroid — mellow, non-bright instruments",
        "moderate": "mid spectral centroid — balanced timbre",
    }[brightness]


def format_music_theory_block(a: dict) -> str:
    """Render the 'Music theory grounding' section for an explanation."""
    character = _character(a["_mode"], a["brightness"], a["tempo"])
    pitches = ", ".join(a["dominant_pitches"])
    return (
        "Music theory grounding:\n"
        f" - Key: {a['key']} (modal character: {character})\n"
        f" - Tempo: {a['tempo']} BPM ({_tempo_phrase(a['tempo'])})\n"
        f" - Timbre: {a['brightness']} ({_timbre_phrase(a['brightness'])})\n"
        f" - Dominant pitch classes: {pitches} "
        f"(top chroma energy — supports the {a['key']} analysis)\n"
        f" - Rhythmic stability: {a['rhythmic_stability']} "
        f"({'steady pulse' if a['rhythmic_stability'] >= 0.5 else 'loose / rubato feel'})"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python music_theory_annotator.py <audio_path>")
        raise SystemExit(1)
    ann = annotate(sys.argv[1])
    print("\nannotate() →", {k: v for k, v in ann.items() if not k.startswith("_")})
    print()
    print(format_music_theory_block(ann))
