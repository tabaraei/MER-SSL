"""
explainer.py — Two-Layer Explanation Engine for Phase C
=========================================================
Produces two complementary explanation types:

  1. generate_template()     — deterministic, technical, citable in the thesis.
                               Covers V-A coordinates, CCC-based similarity,
                               EDA summary, and per-neighbor delta analysis.

  2. generate_rag_context()  — enriched prompt for an LLM backend.
                               Incorporates four XAI-grounded enhancements:
                                 (a) Static emotion knowledge base (music theory)
                                 (b) Contrastive foils — songs NOT retrieved
                                 (c) EDA physiological narrative
                                 (d) MERT layer attribution (model internals)
                                 (e) Mood trajectory suggestion (listening arc)

Academic references:
  - Miller (2019). Explanation in Artificial Intelligence: Insights from the social sciences.
    Artificial Intelligence, 267, 1–38.  [motivates contrastive foil approach]
  - Russell (1980). A circumplex model of affect. Journal of Personality and Social Psychology.
    [motivates quadrant-based descriptions]
  - Thayer (1989). The biopsychology of mood and arousal. Oxford University Press.
    [motivates EDA-arousal interpretation]
"""

import numpy as np

# Human-readable labels for the 4 Russell-quadrant prototypes (ante-hoc profile).
_QUAD_LABELS = {
    "HVHA": "Happy/Energetic",
    "HVLA": "Calm",
    "LVHA": "Tense/Angry",
    "LVLA": "Sad",
}


class ExplainableRAG:

    QUADRANT = {
        (True,  True):  ("High Arousal / High Valence", "energetic and positive",  "HVHA"),
        (True,  False): ("High Arousal / Low Valence",  "intense and negative",    "LVHA"),
        (False, True):  ("Low Arousal / High Valence",  "calm and positive",       "HVLA"),
        (False, False): ("Low Arousal / Low Valence",   "melancholic and subdued", "LVLA"),
    }

    # Static music knowledge base injected into the LLM prompt.
    # Grounded in music psychology literature — avoids artist-name hallucination.
    # Each quadrant entry provides music-theory context the LLM can use
    # to produce genre-aware, human-sounding recommendations without fabricating
    # specific artists or song titles.
    EMOTION_KNOWLEDGE = {
        "HVHA": {
            "tempo":        "fast (typically 130–180 BPM)",
            "harmony":      "major keys, bright timbres, or power chords with strong rhythmic drive",
            "genre_family": "energetic genres: uptempo pop, EDM, hip-hop, hard rock, metal",
            "listener_fit": "listeners seeking excitement, motivation, or physical activity (workouts, dancing, sports)",
            "mood_words":   "powerful, euphoric, exhilarating, driving",
            "eda_expected": "elevated skin conductance, frequent SCR peaks — strong physiological arousal",
            "rec_framing":  "If you're drawn to music that gets you moving and feeling alive, these tracks carry that same charge.",
        },
        "LVHA": {
            "tempo":        "fast to moderate (100–160 BPM) with tense or dark emotional character",
            "harmony":      "minor keys, dissonant intervals, aggressive timbres or dense electronic textures",
            "genre_family": "thrash metal, industrial, aggressive electronic, dark hip-hop, post-punk",
            "listener_fit": "listeners using music for catharsis, pressure focus, or emotional processing",
            "mood_words":   "intense, confrontational, dark, urgent, relentless",
            "eda_expected": "high physiological activation with stress-like patterns; sharp, frequent SCR peaks",
            "rec_framing":  "These tracks share the same raw, high-tension energy — music that hits hard without softening the edges.",
        },
        "HVLA": {
            "tempo":        "slow to moderate (60–100 BPM)",
            "harmony":      "major keys or modal scales, consonant harmonies, warm acoustic timbres",
            "genre_family": "ambient, classical, acoustic folk, soft jazz, lo-fi, chillout",
            "listener_fit": "listeners seeking relaxation, positive focus, or a calming background atmosphere",
            "mood_words":   "peaceful, warm, content, gentle, uplifting",
            "eda_expected": "low physiological arousal, stable skin conductance, very few SCR peaks",
            "rec_framing":  "Like your track, these songs inhabit a calm, bright emotional space — perfect for unwinding or focused work.",
        },
        "LVLA": {
            "tempo":        "slow (40–80 BPM)",
            "harmony":      "minor keys, descending melodic lines, sparse arrangements, intimate production",
            "genre_family": "sad indie, blues, dark folk, melancholic classical, slow singer-songwriter",
            "listener_fit": "listeners in a reflective or introspective mood, processing sadness or nostalgia",
            "mood_words":   "melancholic, nostalgic, tender, brooding, somber",
            "eda_expected": "low overall EDA with slow-rising responses — emotional depth over physical excitement",
            "rec_framing":  "These tracks occupy the same quiet, heavy emotional space — music that sits with you, not against you.",
        },
    }

    # ── Russell quadrant helpers ───────────────────────────────────────────────

    def _quadrant(self, arousal, valence, mid=0.5):
        return self.QUADRANT[(arousal >= mid, valence >= mid)]

    def _arousal_description(self, arousal):
        if arousal >= 0.75: return "very high"
        if arousal >= 0.55: return "high"
        if arousal >= 0.45: return "moderate"
        if arousal >= 0.25: return "low"
        return "very low"

    def _valence_description(self, valence):
        if valence >= 0.75: return "very positive"
        if valence >= 0.55: return "positive"
        if valence >= 0.45: return "neutral"
        if valence >= 0.25: return "negative"
        return "very negative"

    # ── EDA helpers ───────────────────────────────────────────────────────────

    def _eda_summary(self, eda_feats):
        """One-line EDA summary for the template output."""
        if eda_feats is None or np.allclose(eda_feats, 0):
            return "no physiological data available"
        mean_eda, std_eda, slope, n_peaks, mean_amp, max_eda, high_ratio = eda_feats
        activity  = "elevated" if mean_eda > 0.5 else "moderate" if mean_eda > 0.25 else "low"
        trend     = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "stable"
        peak_desc = "several SCR peaks" if n_peaks > 0.5 else "few SCR peaks" if n_peaks > 0.2 else "minimal SCR activity"
        return f"{activity} electrodermal activity ({peak_desc}, {trend} trend, high-arousal ratio={high_ratio:.0%})"

    def _eda_narrative(self, eda_feats):
        """Multi-sentence EDA interpretation for the RAG prompt (physiological narrative)."""
        if eda_feats is None or np.allclose(eda_feats, 0):
            return None
        mean_eda, std_eda, slope, n_peaks, mean_amp, max_eda, high_ratio = eda_feats

        arousal_level = ("high physiological arousal" if mean_eda > 0.6
                         else "moderate physiological engagement" if mean_eda > 0.3
                         else "a calm physiological state")
        variability   = ("highly variable" if std_eda > 0.4
                         else "moderately variable" if std_eda > 0.2
                         else "stable")
        trend_desc    = ("gradually escalating arousal" if slope > 0.01
                         else "decreasing arousal over time" if slope < -0.01
                         else "sustained arousal throughout")
        peak_desc     = ("frequent emotional activation peaks" if n_peaks > 0.6
                         else "moderate emotional peaks" if n_peaks > 0.3
                         else "few distinct peaks — smooth continuous engagement")

        return (f"Listeners of this track showed {arousal_level} with {variability} skin conductance "
                f"and {trend_desc}. The physiological profile features {peak_desc}. "
                f"High-arousal proportion: {high_ratio:.0%} of listening duration "
                f"(normalized across dataset).")

    # ── MERT layer attribution ─────────────────────────────────────────────────

    def _layer_narrative(self, layer_weights):
        """
        Interprets the model's learned MERT layer fusion weights as plain text.
        MERT layers: 0–7 = low-level acoustic, 8–15 = rhythmic/tonal, 16–24 = semantic/melodic.
        These weights are global to the model (not per-song), explaining WHAT the model
        learned to rely on across the dataset.
        """
        if layer_weights is None:
            return None
        w   = np.array(layer_weights, dtype=np.float32)
        w   = w / (w.sum() + 1e-8)
        n   = len(w)

        low_end  = max(1, n // 3)
        mid_end  = max(low_end + 1, 2 * n // 3)

        low  = w[:low_end].sum()
        mid  = w[low_end:mid_end].sum()
        high = w[mid_end:].sum()

        dominant = max(
            [(low,  "low-level acoustic features (timbre, onset, spectral texture)"),
             (mid,  "mid-level rhythmic and tonal patterns"),
             (high, "high-level semantic and melodic structure")],
            key=lambda x: x[0]
        )
        top_layer = int(np.argmax(w))

        return (f"The model's learned MERT layer fusion (across {n} transformer layers) "
                f"relies most on {dominant[1]}. "
                f"Layer {top_layer} carries the highest individual weight ({w[top_layer]:.1%}). "
                f"Layer-group contributions — low-level: {low:.0%} | "
                f"mid-level: {mid:.0%} | high-level: {high:.0%}. "
                f"This means retrieval is primarily driven by "
                f"{'sound texture and timbre' if low > max(mid, high) else 'melodic/harmonic content' if high > mid else 'rhythmic and tonal patterns'}.")

    # ── Mood trajectory ───────────────────────────────────────────────────────

    def _mood_trajectory(self, neighbors):
        """
        Suggests two listening orderings based on arousal arc.
        Energizing arc: ascending arousal. Wind-down arc: descending.
        """
        if len(neighbors) < 2:
            return None
        sorted_asc  = sorted(neighbors, key=lambda n: n["arousal"])
        sorted_desc = list(reversed(sorted_asc))
        spread = sorted_asc[-1]["arousal"] - sorted_asc[0]["arousal"]

        if spread < 0.05:
            return ("The retrieved songs form a tight emotional cluster "
                    f"(arousal spread = {spread:.3f}) — any playback order yields a consistent mood.")

        asc_ids  = [n["music_id"] for n in sorted_asc]
        desc_ids = [n["music_id"] for n in sorted_desc]
        return (f"Mood arc options (Δarousal = {spread:.2f}): "
                f"Energizing order → {asc_ids} (low to high arousal, builds energy). "
                f"Wind-down order → {desc_ids} (high to low arousal, eases into calm).")

    # ── Template explanation (technical, deterministic) ───────────────────────

    def _prototype_profile_lines(self, prototype_profile):
        """Render the ante-hoc 4-quadrant prototype activation profile."""
        if not prototype_profile or not prototype_profile.get("similarities"):
            return []
        sims = prototype_profile["similarities"]
        best = prototype_profile.get("best")
        lines = [
            "",
            "🎯  PROTOTYPE ACTIVATION PROFILE (ante-hoc)",
            "-" * 65,
            "  Cosine similarity of the query to each emotion-quadrant centroid",
            "  (the 4 prototypes are the mean latent of each Russell quadrant):",
        ]
        for code, sim in sorted(sims.items(), key=lambda kv: kv[1], reverse=True):
            marker = "  ◄ best match" if code == best else ""
            lines.append(f"    {code} ({_QUAD_LABELS.get(code, code)}) : {sim:+.4f}{marker}")
        if best:
            lines.append(
                f"  → Query most strongly matches the {best} "
                f"({_QUAD_LABELS.get(best, best)}) prototype.")
        return lines

    def generate_template(self, query_id, query_arousal, query_valence,
                          query_eda, neighbors, prototype_profile=None):
        q_label, q_desc, q_quad = self._quadrant(query_arousal, query_valence)

        lines = [
            "=" * 65,
            f"🎵  QUERY: Song ID {query_id}",
            "=" * 65,
            f"  Emotion Region  : {q_label} ({q_quad})",
            f"  Arousal         : {self._arousal_description(query_arousal)} ({query_arousal:.3f})",
            f"  Valence         : {self._valence_description(query_valence)} ({query_valence:.3f})",
            f"  Physiology      : {self._eda_summary(query_eda)}",
        ]
        lines += self._prototype_profile_lines(prototype_profile)
        lines += [
            "",
            "📋  RETRIEVAL EXPLANATION",
            "-" * 65,
            (f"  Songs retrieved via cosine similarity in the emotion-aware latent space "
             f"of the MERT-based Hybrid model (Weighted Layer Fusion + SupCR loss). "
             f"Query occupies '{q_quad}' — {q_desc} affective region."),
            "",
            f"🔍  TOP-{len(neighbors)} RETRIEVED SONGS",
            "-" * 65,
        ]

        for n in neighbors:
            _, n_desc, n_quad = self._quadrant(n["arousal"], n["valence"])
            da = n["arousal"] - query_arousal
            dv = n["valence"] - query_valence
            delta = []
            if abs(da) > 0.05:
                delta.append(f"arousal {'↑' if da > 0 else '↓'}{abs(da):.2f}")
            if abs(dv) > 0.05:
                delta.append(f"valence {'↑' if dv > 0 else '↓'}{abs(dv):.2f}")
            delta_text = (", ".join(delta) + " vs query") if delta else "identical emotion coordinates"

            lines += [
                f"  ┌─ Rank {n['rank']} | Song ID {n['music_id']} "
                f"(cosine sim = {n['similarity']:.4f})",
                f"  │  Quadrant     : {n_quad} ({n_desc})",
                f"  │  Arousal      : {self._arousal_description(n['arousal'])} ({n['arousal']:.3f})",
                f"  │  Valence      : {self._valence_description(n['valence'])} ({n['valence']:.3f})",
                f"  │  Physiology   : {self._eda_summary(n['eda_feats'])}",
                f"  │  Δ from query : {delta_text}",
                f"  └─────────────────────────────────────────────────────────",
            ]

        avg_a   = np.mean([n["arousal"] for n in neighbors])
        avg_v   = np.mean([n["valence"] for n in neighbors])
        avg_sim = np.mean([n["similarity"] for n in neighbors])
        traj    = self._mood_trajectory(neighbors)

        lines += [
            "",
            "🧠  MUSIC-THEORETIC SUMMARY",
            "-" * 65,
            (f"  avg neighbor arousal={avg_a:.3f} | valence={avg_v:.3f} | "
             f"mean cosine sim={avg_sim:.4f}"),
        ]
        if traj:
            lines += [f"  {traj}"]
        lines += ["=" * 65]
        return "\n".join(lines)

    # ── RAG prompt (enriched, for LLM) ────────────────────────────────────────

    def generate_rag_context(self, query_id, query_arousal, query_valence,
                             query_eda, neighbors, foils=None, layer_weights=None):
        """
        Builds the enriched LLM prompt with five XAI-grounded sections:
          (a) Emotion knowledge base — music theory context per quadrant
          (b) EDA physiological narrative — body response interpretation
          (c) Contrastive foils — songs NOT retrieved, grounding the explanation
          (d) MERT layer attribution — what the model actually learned
          (e) Mood trajectory — suggested listening order
        """
        _, q_desc, q_quad = self._quadrant(query_arousal, query_valence)
        kb = self.EMOTION_KNOWLEDGE[q_quad]

        eda_text   = self._eda_narrative(query_eda)
        layer_text = self._layer_narrative(layer_weights)
        traj_text  = self._mood_trajectory(neighbors)

        # Translate neighbors to plain English
        neighbor_lines = []
        for n in neighbors:
            _, n_desc, n_quad = self._quadrant(n["arousal"], n["valence"])
            da = n["arousal"] - query_arousal
            dv = n["valence"] - query_valence
            delta = []
            if abs(da) > 0.05:
                delta.append(f"arousal {'slightly higher' if da > 0 else 'slightly lower'}")
            if abs(dv) > 0.05:
                delta.append(f"valence {'slightly higher' if dv > 0 else 'slightly lower'}")
            delta_str = " and ".join(delta) if delta else "nearly identical emotional profile"
            line = (f"  Rank {n['rank']} — Song {n['music_id']}: {n_desc} ({n_quad}), "
                    f"A={n['arousal']:.3f} V={n['valence']:.3f}, "
                    f"sim={n['similarity']:.4f} [{delta_str}]")
            n_eda = self._eda_narrative(n["eda_feats"])
            if n_eda:
                line += f"\n    └ Physiology: {n_eda}"
            neighbor_lines.append(line)

        # Translate foils to plain English
        foil_lines = []
        if foils:
            for f in foils:
                _, f_desc, f_quad = self._quadrant(f["arousal"], f["valence"])
                foil_lines.append(
                    f"  Song {f['music_id']}: {f_desc} ({f_quad}), "
                    f"A={f['arousal']:.3f} V={f['valence']:.3f}, "
                    f"sim={f['similarity']:.4f} — rejected (emotionally distant from query)"
                )

        # Build prompt
        prompt = (
            "You are a music recommendation assistant with expertise in music psychology "
            "and emotion-aware AI retrieval systems. Explain why a set of songs was recommended "
            "in a warm, human way — like a knowledgeable friend who understands both the science "
            "and the feeling of music. Avoid technical jargon unless briefly defining it.\n\n"

            f"━━━ QUERY SONG (ID {query_id}) ━━━\n"
            f"  Emotional character : {q_desc} — {kb['mood_words']}\n"
            f"  Emotional region    : {self.QUADRANT[(query_arousal >= 0.5, query_valence >= 0.5)][0]} ({q_quad})\n"
            f"  Arousal             : {self._arousal_description(query_arousal)} ({query_arousal:.3f})\n"
            f"  Valence             : {self._valence_description(query_valence)} ({query_valence:.3f})\n"
            f"  Music theory        : {kb['harmony']}. Tempo: {kb['tempo']}.\n"
            f"  Genre family        : {kb['genre_family']}\n"
            f"  Listener profile    : {kb['listener_fit']}\n"
        )

        if eda_text:
            prompt += f"  Physiology          : {eda_text}\n"

        prompt += f"\n━━━ RETRIEVED SONGS (most similar in latent space) ━━━\n"
        prompt += "\n".join(neighbor_lines) + "\n"

        if foil_lines:
            prompt += (
                "\n━━━ CONTRASTIVE FOILS (most dissimilar — NOT retrieved) ━━━\n"
                + "\n".join(foil_lines) + "\n"
                + "  → Use these to explain WHAT MAKES the retrieved songs special: "
                "what emotional qualities they share that clearly distinguish them from the foils.\n"
            )

        if layer_text:
            prompt += f"\n━━━ MODEL EXPLANATION (MERT layer attribution) ━━━\n  {layer_text}\n"

        if traj_text:
            prompt += f"\n━━━ MOOD TRAJECTORY ━━━\n  {traj_text}\n"

        prompt += (
            f"\n━━━ YOUR TASK ━━━\n"
            f"Write a response with exactly these four parts:\n"
            f"1. RECOMMENDATION (2–3 sentences): Talk directly to the listener. "
            f"Start with: '{kb['rec_framing']}' then say why these specific songs fit.\n"
            f"2. EMOTIONAL CONNECTION (1–2 sentences): What binds all retrieved songs "
            f"to the query emotionally? Reference the V-A plane naturally if helpful.\n"
            f"3. WHY NOT THE OTHERS (1 sentence, only if foils provided): "
            f"Briefly contrast with the rejected songs — what emotional gap separates them.\n"
            f"4. LISTENING EXPERIENCE (1 sentence): Describe the shared physiological or "
            f"mood experience these songs create for the listener.\n"
            f"Total length: 150–200 words. Warm tone, no bullet points.\n\n"
            f"Response:"
        )

        ctx = {
            "query": {
                "music_id": query_id,
                "arousal":  round(query_arousal, 4),
                "valence":  round(query_valence, 4),
                "quadrant": q_quad,
            },
            "neighbors": [
                {"rank": n["rank"], "music_id": n["music_id"],
                 "similarity": round(n["similarity"], 4),
                 "arousal": round(n["arousal"], 4), "valence": round(n["valence"], 4)}
                for n in neighbors
            ],
            "foils": [
                {"music_id": f["music_id"], "similarity": round(f["similarity"], 4),
                 "arousal": round(f["arousal"], 4), "valence": round(f["valence"], 4)}
                for f in (foils or [])
            ],
        }
        return prompt, ctx
