# My Thesis Journey — Music Emotion Recognition

## What I was trying to do (and why)

I wanted to build a system that listens to a piece of music and figures out the emotion in it — and, just as importantly, can explain *why* it thinks so. This field is called Music Emotion Recognition, or MER: teaching a computer to map a song onto feelings like "energetic and happy" or "calm and sad."

The reason this matters is that most music apps already recommend songs, but they treat emotion as a black box. They might tell you two songs are similar, but they can't say *why* in a way a human understands. For things like therapy playlists, mood regulation, or just trust in a recommendation, that "why" is the whole point. A system that says "I picked this because it shares the same calm, gentle character as your song" is far more useful than one that just hands you a track and stays silent.

So the question I set out to answer was twofold: first, do modern AI music models actually capture *real musical feeling* (not just genre or sound texture), and second, can I turn that into a system that both predicts emotion accurately and explains itself in plain, music-aware language. To keep this measurable, I described every song using two numbers — arousal (how energetic it is, calm vs intense) and valence (how positive it feels, sad vs happy) — which together form a standard "emotion map" researchers use.

**The takeaway from the start: my goal was never just accuracy — it was accuracy plus a believable explanation, and that double goal shaped every decision afterward.**

## Where I started — MERT embeddings (Phase A)

I chose to build on top of a model called MERT. MERT is a large AI model that was trained on a huge amount of music without anyone labelling it — it just learned the patterns of music on its own. That kind of model is called self-supervised, meaning it teaches itself from raw data instead of from human labels. The advantage is that it already "understands" a lot about music before I add anything.

I used MERT as a frozen model. Frozen means I did not change MERT itself at all — I left its internal knowledge exactly as it came and only built small new pieces on top of it. I did this because MERT was trained on far more music than my small dataset contains, so retraining it would likely make things worse, not better, and it would cost a lot of computing power. It was smarter to treat MERT as a fixed expert and just learn how to read its opinions.

But before trusting it, I needed to check that MERT actually contains musically meaningful information. I did this with probing. Probing means attaching a very simple model to MERT's output and seeing whether that simple model can recover a known musical fact — if a simple model succeeds, the information must already be sitting there in MERT's representation. I probed for two things: harmonic mode (whether a song is in a major or minor key, roughly "bright" vs "dark") and tempo (how fast it is in beats per minute).

The results were telling. For major versus minor, the probe hit 100% accuracy — meaning MERT perfectly separates bright-sounding from dark-sounding music, so harmony is clearly baked into it. Tempo was much weaker, with a score (R²) of about 0.12, which means the simple probe could only explain about 12% of the variation in song speed — so MERT knows speed only vaguely. That gap didn't worry me; it told me where MERT is strong and where it might need help later.

**The takeaway: before building anything fancy, I confirmed MERT genuinely carries musical meaning — strongly for harmony, weakly for tempo — which gave me a solid, evidence-based foundation to build on.**

## Building the emotion prediction model (Phase B)

With a trustworthy foundation, I built the part that actually predicts emotion. My first instinct was simple regression — just fit a line from MERT's numbers to the arousal and valence scores. But that wasn't enough, because emotion prediction has several different ways of being "wrong," and a single basic error measure can't catch them all.

So I used a four-part loss. A loss is just the score the model tries to make as small as possible during training; combining four of them means the model has to satisfy four different definitions of "good" at once. The first part simply checks how far off the predicted numbers are. The second part checks that the predictions rise and fall in step with the real emotions without sitting at a constant offset (this one, called CCC, is the strict standard in emotion research). The third part makes sure songs end up in the right *order* from low to high energy, even if the exact numbers are a little off. And the fourth part — the most important for later — pulls songs that feel similar close together inside the model's internal map, so emotionally similar songs end up as neighbours. That last one quietly set up the explanation system I'd build in Phase C.

Two real problems showed up during training. The first was that the part of my model that decides how much to listen to each of MERT's internal layers simply wasn't learning — it was treating every layer equally and refusing to develop preferences. I fixed this by letting that specific part learn much faster than the rest of the model (giving it a bigger learning rate), which finally let it form opinions about which layers matter. The second problem was that my dataset, PMEmo (about 767 pop-song clips rated by listeners), is lopsided — most songs are upbeat and happy. A lazy model could score well just by guessing "happy" most of the time. I fixed this by showing the rarer emotions (sad, calm, angry songs) more often during training, so the model couldn't coast on the majority.

The results, in plain terms: the model reached an arousal score (R²) of about 0.65, meaning it explains roughly 65% of the variation in how energetic songs are — quite good. Valence landed near 0.51, explaining about half the variation in how positive a song feels — noticeably harder. That valence gap turned out to be a theme of the whole project: how positive a song feels often depends on lyrics and culture, not just sound, so audio-only systems hit a ceiling there.

I also ran an extra experiment with EDA fusion. EDA, electrodermal activity, is a tiny physical signal — small changes in the skin's sweat/conductance that happen automatically when a person feels aroused — and PMEmo recorded it from listeners. I added it because it's a body-based clue about emotion that is completely independent of the audio. It pushed the strict arousal score (CCC) up to about 0.85, my best result on the energetic dimension, which makes sense: arousal is physical, and the body literally reacts to it.

**The takeaway: a single error measure wasn't enough — emotion needed several goals at once, careful fixes for a stubborn model and a lopsided dataset, and even a peek at listeners' bodies, with valence remaining the genuinely hard nut to crack.**

## Making the model explain itself (Phase C)

Predicting two numbers is useful, but for my thesis it wasn't enough. A number like "valence 0.4" tells a listener nothing they can feel or trust. The heart of my project was explanation, so Phase C turned the predictions into something a person could actually understand.

I used prototype-based retrieval. In plain words: instead of inventing an explanation out of thin air, the system answers a query by finding real example songs from the collection that sit closest to it in the emotion map, and those real examples *are* the explanation. It's reasoning by analogy — "this song belongs here because it's almost identical to these other songs you can listen to." Because Phase B had already pulled similar songs into tight neighbourhoods, this step worked naturally.

To make the explanations concrete, I leaned on the four emotion quadrants — the four corners of the emotion map: happy/energetic, calm/positive, sad/subdued, and tense/angry. Every song falls into one of these, so the system can say which corner a song lives in and which neighbours share it. It even shows the opposite — songs it deliberately did *not* pick — because explaining "why this and not that" is how people naturally explain things.

This is also where my supervisor pushed me on a key distinction: ante-hoc versus post-hoc explanation. Post-hoc means explaining a black box after the fact, guessing at its reasoning. Ante-hoc means the system's reasoning is transparent *by design* — the explanation is the actual decision process, not a story told afterward. My supervisor strongly preferred ante-hoc, because a guessed explanation can be wrong in ways you can't detect. My retrieval system is ante-hoc at its core: the decision genuinely *is* "these real songs are the nearest neighbours," so the explanation is faithful rather than invented.

**The takeaway: I learned that a trustworthy explanation isn't decoration added at the end — it has to be the model's real reasoning, and designing for that honesty changed how I judged the whole system.**

## Trying to push the results further — adding wav2vec2 (Dual-SSL)

Valence was still stuck around 0.51, so I tried adding a second AI model alongside MERT, an approach I called Dual-SSL (two self-supervised models working together).

The second model was wav2vec2. The interesting thing is that wav2vec2 was trained on *speech*, not music. So it "hears" different things — voice-like qualities, articulation, the rise and fall of energy in a sound — rather than musical harmony and melody. My hope was that these speech-style cues might catch emotional signals that MERT's purely musical training misses, especially for valence, which is the slippery one.

It helped, but modestly. Valence rose from about 0.51 to 0.57 — meaning the combined model now explained about 57% of the variation in how positive songs feel, a real but small step up. Arousal sat around 0.68.

The more interesting result was something I didn't expect, which I came to call fusion collapse. The little component that decides how much to trust each internal layer of each model just spread its attention evenly across everything instead of picking favourites — and no trick I tried (squeezing the inputs smaller, or adding a penalty that should have forced it to specialize) changed that. After digging in, the reason became clear: with only around 600 training songs, there simply isn't enough data for the model to learn fine-grained preferences once it has lots of inputs to lean on. It just averages everything and moves on.

**The takeaway: adding a second, differently-trained model gave a small honest gain, but the bigger lesson was about limits — with a small dataset, the model can't learn to be picky, and recognizing that constraint was more valuable than the score itself.**

## Testing with a new dataset — IADS-E (the negative finding)

Next I tried borrowing data from a different domain. There's a published paper (Simonetta and colleagues) showing that mixing music with recordings of everyday environmental sounds — labelled with the same emotion scales — can improve emotion prediction, because both share a common emotional space and the environmental sounds fill in emotional regions music rarely visits. I tried to replicate that idea using a dataset of environmental sounds called IADS-E.

It didn't work. Every way I mixed the environmental sounds in, my music valence score got *worse*, not better. The reason, as best I can tell, is that the AI models I'm using are specialists — MERT knows music, wav2vec2 knows speech — and their internal "emotional language" doesn't transfer cleanly across the gap between music and random environmental noise. The earlier paper used older hand-crafted features that happened to transfer better; my modern self-supervised features did not.

Crucially, this is still a valid and useful thesis result. A negative finding that's honestly tested and clearly explained is real knowledge: it tells the field that self-supervised audio features, for all their power, don't automatically carry emotional meaning across very different kinds of sound. That's worth reporting, and it strengthens the story rather than weakening it.

**The takeaway: I learned that "it didn't work" can be a genuine contribution when you understand and can explain *why* — and that chasing a borrowed idea taught me something real about the limits of these models.**

## Where I am now and what comes next

The most recent stretch of work pushed all three phases forward at once, and it's the part I'm actively in the middle of writing up.

First, I added a mel-spectrogram CNN branch. A spectrogram is basically a picture of a sound — frequencies on one axis, time on the other — and a CNN is a small network that learns directly from images. So this branch learns emotion straight from the "picture" of each song, trained from scratch on my own data rather than borrowed from elsewhere. This gave me my best arousal result yet, finally crossing 0.70 (explaining over 70% of the variation in energy). The surprise was that adding this tiny home-grown network made wav2vec2 redundant — a model using MERT plus the little spectrogram network did just as well as one that also included the big speech model. A small thing I trained myself quietly replaced a large pre-trained one.

Second, I went back to Phase A and ran a much more thorough probing study — testing all 25 of MERT's internal layers against eight different musical properties — to find exactly what MERT is blind to. It came down to two things: tempo and key. MERT simply doesn't expose absolute speed or musical key very well.

Third, I used those gaps to inform Phase B. I fed tempo and key back into the model directly, as explicit extra inputs, to see if handing the model the very things it was missing would help. It did — but only for arousal, which rose to my best-ever 0.7182. Feeding in tempo helped because fast music genuinely tends to feel more energetic. Key didn't help valence, and I'm fairly sure that's because I fed key in as a plain number from 0 to 11, which throws away the fact that musical keys are circular and related in subtle ways. The honest lesson: giving the model exactly what it lacks works — but only when that thing truly relates to the target *and* is fed in sensibly.

Fourth, I used librosa (a standard audio-analysis toolkit) to enrich the Phase C explanations with real music theory — naming each song's key, tempo, and brightness in the explanation text. There's an important honesty point here that I'm careful to state: these descriptions come from librosa analyzing the audio directly, *not* from the AI model's own reasoning. So they truthfully describe the song, but they are corroboration alongside the model, not a window into what the model internally computed.

What's genuinely still ahead: I want to measure overfitting properly on the little spectrogram network (it learns from very few songs, so I need to confirm it isn't just memorizing), try a music-pretrained CNN instead of a from-scratch one, and tackle the deepest remaining problem — that all my headline scores are propped up by the majority of happy songs, while the model still struggles on the rarer sad, calm, and angry ones. That class imbalance, more than any single model choice, is the real ceiling now.

**The takeaway: the project's frontier is no longer about stacking on bigger models — it's about feeding the model the specific things it lacks, being honest about where the gains really come from, and facing the dataset's built-in bias head-on.**

## Adding new steps

New entries go below this line, in the same format: what I did, why, what happened, what I learned.

### Did the explanation system actually work? — Phase C evaluation

**What I did.** I finally measured whether my retrieval system actually fetches emotionally similar songs, instead of just assuming it does. I wrote an evaluation that, for every song, looks at the songs sitting closest to it in the model's internal "emotion map" and checks two things: are those near neighbours genuinely close in feeling (a score called Precision@k), and do the four emotion groups form clean, separate clusters (a score called Silhouette). I did this the strict way — each song was judged by a version of the model that had never seen it during training, so the model couldn't just be remembering. I ran it on both my single-MERT model and my best dual model.

**Why I did it.** My whole thesis rests on the claim that one of my training tricks (a loss that pulls similar-feeling songs together) actually organizes the model's internal space by emotion. Until now that was just a hope. I needed real numbers to either back it up or be honest about it.

**What happened.** Precision came out around 58% for the top neighbours — meaning well over half of each song's nearest neighbours really are emotionally close, which is clearly better than random. So the retrieval genuinely works. The clustering score, though, came out at essentially zero for both models (the dual model was a hair above zero, the single model a hair below). My two models were basically tied on both measures.

**What I learned.** The two scores tell a coherent story once you stop expecting them to agree. Songs do land next to emotionally similar songs (good Precision), but emotion isn't four neat boxes — it's a smooth gradient from calm to intense and from sad to happy. Asking "are there four cleanly separated clumps?" gives a near-zero answer even when the space is well organized, because a song right on the border between two moods is honestly close to both. So I learned not to oversell that near-zero clustering number as either success or failure — it's the wrong question for something continuous. The real evidence is Precision@k, and that holds up. I also learned, again, that my fancier dual model didn't actually organize the emotional space any better than plain MERT — it just predicts the numbers slightly better.

### Did the training actually help? — the naive baseline + thesis pictures

**What I did.** I built a deliberately dumb baseline: take the raw, untouched MERT model, just average its last layer for each song, and run the exact same retrieval test on that — no training, none of my tricks. Then I compared it against my trained space. I also exported the figures I'll put in the thesis: a side-by-side "before vs after training" map of the song space, a bar chart of which MERT layers the model leans on, and the full written explanations for five example songs (one from each emotional corner plus one ambiguous one).

**Why I did it.** It's easy to claim "my training organized the space," but a skeptical examiner will ask: how do you know the raw model wasn't already this good? The only honest answer is to measure the raw model and show the difference. The figures serve the same purpose — they make the claims visible instead of just stated.

**What happened.** The dumb baseline got about 48% on the retrieval score; my trained space got about 58%. So training added roughly ten points — a clear, real improvement, not noise. The before/after map made it obvious too: raw MERT is one shapeless cloud with all the emotions mixed together, while the trained space breaks into clear strands. There was one honest surprise, though: by the "four separate clumps" score, the *untrained* model actually looked slightly better. 

**What I learned.** Training genuinely helped at the thing that matters — finding emotionally similar songs — and I can now prove it with a baseline instead of just asserting it. The surprise about the clumpiness score taught me something real: my training doesn't try to build four tidy islands, it pulls songs together by smooth emotional closeness, which is exactly why the retrieval got better while the "islands" score got flatter. Raw MERT probably has some coarse clumpiness from genre or sound texture, but that's not the same as being good at emotion. It made me more confident that I'm measuring the right thing, and more careful about which number to trust. I also learned, looking at the layer bar chart, that the model spreads its attention almost evenly across all layers — the slight favourites (layers 14–16) are real but tiny, so I should describe them as a faint lean, not a strong preference.

**Footnote — getting the written explanations working on a locked-down server.** The friendly, human-readable half of my explanation system uses a language model to turn the raw numbers into prose. My first attempt assumed I could run a local model server (Ollama), but the university machine doesn't allow the admin rights that needs. Rather than give up on that half, I switched the system to load a small open language model (Qwen 1.5B) directly inside my existing Python environment — no admin rights, no separate server. It now writes a clean four-part explanation for each song. The lesson: I kept the rigorous, deterministic half of the explanation as the thing I actually cite, and treated the language-model prose as a swappable presentation layer — so a deployment hiccup never threatened the real contribution.

### Was my complicated loss worth it? — the ablation (and an honest hit to one of my claims)

**What I did.** My model is trained with a four-part scoring rule, not just the simple "how far off is the number" one. A fair examiner will ask: did the three extra parts actually earn their place, or did I just add complexity for its own sake? So I retrained the model three ways — the plain simple version, a version with two of the extra parts added, and the full version — keeping everything else identical, and compared them.

**Why I did it.** Two reasons. First, to justify the design honestly instead of asserting it. Second, one of those extra parts (the one meant to pull similar-feeling songs together) is the exact thing my thesis claimed "organizes the emotional space into clusters" — and my earlier figures suggested that claim was shaky. The ablation was the clean way to find out for sure.

**What happened.** Two of the extra parts (the ones that reward tracking the emotional ups-and-downs correctly) clearly earned their keep — adding them lifted the main agreement score by about ten points, a big, unambiguous gain over the plain version. But the contrastive part — the one I claimed builds emotional clusters — gave a more uncomfortable answer. It did help retrieval a little, but when I removed it the "clean clusters" score actually went *up*, not down. In other words, that part is not building clusters at all; if anything it slightly works against tidy clusters.

**What I learned.** This was the most clarifying experiment of the project. The good news: my complex loss is genuinely justified — I can now prove the plain version is meaningfully worse, so the extra machinery isn't decoration. The humbling news: I had been telling a story about that one component ("it organizes the space into emotional clusters") that my own experiment disproves. What it actually does is pull songs together by smooth emotional closeness, which helps find similar songs but deliberately blurs hard category lines. I'm rewriting that claim to match the evidence — it earns its place for retrieval, not for clustering. Being forced to correct my own narrative with my own numbers is, I think, the part of this thesis I'm most confident is honest.
