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

The results were telling, but only once I measured them honestly. My first quick probes seemed to show ~100% accuracy on major-versus-minor and a tempo R² of about 0.12 — and both turned out to be misleading. The mode "label" was a degenerate proxy: it just thresholded overall chroma loudness, so almost every song fell on one side and a trivial constant guess scored ~100%. And the 0.12 tempo number came from an early one-off probe I never logged. When I redid this properly — estimating real major/minor with the standard Krumhansl–Schmuckler method, and sweeping a clean Ridge probe across all 25 of MERT's layers — the honest picture was different. MERT knows major-versus-minor only modestly (about 67% accuracy, best at a middle layer), and it is essentially *blind* to absolute tempo: the tempo R² is actually **negative** (about −0.83 at the best single layer, and −2.1 when all layers are pooled together). A negative R² means the probe does *worse than just guessing the average speed for every song* — there is no linear direction inside MERT that tracks tempo, so trying to read one off only fits noise. Far from worrying me, that sharp gap was the most useful thing Phase A gave me: it told me precisely where MERT is strong (harmony, timbre) and where it would need an explicit helping hand later (tempo and key).

**The takeaway: before building anything fancy, I confirmed MERT genuinely carries musical meaning — clearly for harmony and timbre, but not for absolute tempo and only weakly for key — which gave me a solid, evidence-based map of where to build and where to compensate.**

## Building the emotion prediction model (Phase B)


### Model Architecture Explanations
### Model Architecture Pipeline and Tensor Flow

Here is exactly how our pipeline works, step-by-step, from the raw audio to the final 128-D vector:

---

#### Step 1: The MERT Feature Extraction
We start by feeding the raw $24\text{ kHz}$ audio waveform into the frozen MERT model. Inside MERT, the audio first passes through a CNN feature extractor, and then through a stack of transformer layers. In total, MERT outputs 25 distinct layers of representations for the song. Each of these 25 layers contains a sequence of 1024-dimensional vectors capturing different levels of musical semantics.

---

#### Step 2: The "Weighted Layer Fusion" (The Mixer)
Instead of just throwing away the first 24 layers and only using the last one, we use a module called `WeightedLayerFusion`.

* **How it works:** The model assigns a learnable parameter (a weight) to each of the 25 layers. We pass these 25 weights through a mathematical function called a softmax, which forces all the weights to sum up to exactly $1.0$ (or $100\%$).
* **The calculation:** The model multiplies Layer 1 by Weight 1, Layer 2 by Weight 2, all the way to Layer 25, and then adds them all together into a single, fused 1024-D vector.
* **Our Scientific Finding:** Ideally, we wanted the model to strongly prefer specific layers (e.g., giving $80\%$ weight to layer 15). However, our empirical audit showed a "fusion collapse": because our PMEmo dataset is so small ($\sim 600$ training songs), the network lacked the gradient pressure to learn specialized weights. As a result, the weights stayed nearly uniform (an entropy of $3.218$ out of a maximum $3.219$), meaning the model essentially just calculated an equal average of all 25 layers.

---

#### Step 3: The Multi-Encoder Assembly (Adding the other features)
Once we have our fused 1024-D vector from MERT, we bring in the other features. In our "Enhanced" or "Triple" architectures, we extract features from other frozen models separately:

* We get a 768-D vector from the speech-pretrained wav2vec2 model.
* We use a tiny linear branch to process our extracted music-theory gap features (tempo and cyclic key), producing a 32-D vector.

We then concatenate (glue side-by-side) these vectors together. For example, in the Enhanced Dual-SSL model, fusing MERT (1024) + wav2vec2 (768) + theory (32) creates a massive 1824-D vector.

---

#### Step 4: The Bottleneck and Latent Space
A 1824-dimensional vector is far too large for a dataset of only $\sim 600$ songs; it would instantly overfit. To solve this, we force this massive vector through a Multi-Layer Perceptron (MLP) "bottleneck".

* The MLP shrinks the data step-by-step through dense layers: $1024 \text{ (or } 1824\text{)} \rightarrow 256 \rightarrow 128$ dimensions.
* We then apply $L_2$-normalization to this final 128-D vector. Geometrically, this projects every song onto the surface of a 128-dimensional sphere. This specific spherical space is our "contrastive latent space", which is heavily shaped by our SupCR loss to pull emotionally similar songs close together.

---

#### Step 5: The Output
Finally, a small regression head reads that 128-D coordinate and outputs just two continuous numbers: the exact Arousal and Valence predictions for the song.

> By explaining it this way, you show your examiners exactly how the tensor dimensions flow through the network, while honestly acknowledging that the `WeightedLayerFusion` acts more as a transparent interpretability hook rather than a magical feature selector.

---

#### 1. Why MERT uses a CNN first, then Transformers
MERT is built on the same architectural foundation as speech models like wav2vec 2.0 and HuBERT. Feeding raw audio waveforms directly into a Transformer is computationally impossible because raw audio contains tens of thousands of samples per second (e.g., $24,000\text{ Hz}$), and Transformers scale quadratically with sequence length. 

To solve this, the audio first goes through a multi-layer 1-Dimensional Convolutional Neural Network (1D-CNN) which acts as an acoustic feature extractor. The CNN downsamples the raw, high-resolution audio waveform into a much lower framerate (e.g., 50 or 75 frames per second) by extracting local, short-range acoustic textures like edges of notes or drum hits. Once the audio is "tokenized" into this manageable sequence of vectors, the 12-layer Transformer block takes over to model the long-range temporal dependencies and global musical semantics across the whole song.

---

#### 2. Is concatenating a 1024-D vector with a 32-D theory vector too naive?
It seems unbalanced at first glance, but from a machine learning perspective, it is a mathematically standard and robust method for multi-modal feature fusion. While the 1024 dimensions from MERT are massive, our Phase A linear probing proved that they are "blind" to absolute tempo and musical key. The tiny 32-D vector (generated from our $\text{Linear}(2, 32)$ layer) injects this explicitly missing structural gap. 

The concatenation itself is just a data-gathering step to create a single 1056-D tensor. The real magic happens immediately after, when this concatenated vector is pushed into the Multi-Layer Perceptron (MLP) bottleneck. The MLP’s dense weight matrices look at all 1056 inputs simultaneously and learn to dynamically scale and mix them. If the 32-D key/tempo features are highly predictive of Arousal (which they are), the network's gradient descent will simply assign larger weights to those 32 connections, completely overcoming the dimensional imbalance.

---

#### 3. What is $L_2$ Normalization and why do we use it?
$L_2$ normalization is a mathematical operation where you divide a vector by its own magnitude (its length), forcing the vector's length to become exactly $1.0$:

$$\mathbf{\hat{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

Geometrically, if you $L_2$-normalize all the 128-D vectors coming out of our bottleneck, you project every single song onto the surface of a 128-dimensional hypersphere. This is strictly necessary for our Phase C explainability system for two reasons:

* **Distance calculation:** Our SupCR contrastive loss and our $k$-NN retrieval system rely on finding the nearest neighbors using cosine similarity or Euclidean distance. By forcing all vectors to have a radius of $1.0$, we eliminate magnitude as a variable. This means two songs will be considered similar only if they point in the exact same semantic direction in the latent space, ignoring irrelevant scaling factors.
* **Stability:** In deep metric learning, unbounded vectors can cause the network's loss to explode or collapse. Normalizing the embeddings stabilizes the gradients during training and creates a well-behaved continuous gradient of emotions across the latent space.

With a trustworthy foundation, I built the part that actually predicts emotion. My first instinct was simple regression — just fit a line from MERT's numbers to the arousal and valence scores. But that wasn't enough, because emotion prediction has several different ways of being "wrong," and a single basic error measure can't catch them all.

So I used a four-part loss. A loss is just the score the model tries to make as small as possible during training; combining four of them means the model has to satisfy four different definitions of "good" at once. The first part simply checks how far off the predicted numbers are. The second part checks that the predictions rise and fall in step with the real emotions without sitting at a constant offset (this one, called CCC, is the strict standard in emotion research). The third part makes sure songs end up in the right *order* from low to high energy, even if the exact numbers are a little off. And the fourth part — the most important for later — pulls songs that feel similar close together inside the model's internal map, so emotionally similar songs end up as neighbours. That last one quietly set up the explanation system I'd build in Phase C.



In my Phase B emotion prediction model (such as the "Enhanced" architecture), I designed a four-part objective called the Hybrid Loss. Because a single error metric cannot capture all the ways emotion prediction fails, I used four distinct loss terms with the following specific weights
:
MSE (Mean Squared Error): Weight = 1.0
CCC (Concordance Correlation Coefficient): Weight = 0.5
Rank Loss (Soft Spearman): Weight = 0.3
SupCR (Supervised Contrastive Regression): Weight = 0.1
To answer your question directly: no, they are not all applied to the same part of the model. The loss function is specifically split into two different architectural locations to achieve two completely different mathematical goals.

Here is exactly where and how they are used:
1. Supervising the Final Output (The Regression Head) The MSE (1.0), CCC (0.5), and Rank (0.3) losses are all applied at the very end of the network, directly to the final 2D regression output (the predicted Valence and Arousal coordinates)
.
Why here? These three terms evaluate the final prediction. MSE forces the model to get the exact numerical coordinates right. CCC ensures the predictions accurately track the variance and mean of the human annotations. The Rank loss ensures the ordinal sorting of the songs from low-to-high energy is preserved
.

2. Shaping the Intermediate Latent Space (The Bottleneck) The SupCR (0.1) loss is applied one layer earlier in the model—specifically to the 128-dimensional L 
2
​
 -normalized latent vector (the bottleneck output) just before it goes into the final regression head
.
Why here? The goal of SupCR is not to calculate coordinate errors, but to act as a geometric anchor. It shapes the intermediate topological space during gradient descent, pulling the 128-D representations of emotionally similar songs closer together
. I deliberately applied it here because Phase C (our Explainable RAG system) cuts off the regression head and uses this exact 128-D latent space to perform its k-Nearest Neighbors (k-NN) retrieval. Without applying SupCR directly to the latent bottleneck, the retrieval system would just pull random songs instead of emotionally coherent ones
.
By splitting the loss this way, my model learns to project raw audio into a geometrically organized continuous manifold (guided by SupCR at the bottleneck) while still outputting highly accurate circumplex coordinates (guided by MSE+CCC+Rank at the final head)


Two real problems showed up during training. The first was that the part of my model that decides how much to listen to each of MERT's internal layers simply wasn't learning — it was treating every layer equally and refusing to develop preferences. I fixed this by letting that specific part learn much faster than the rest of the model (giving it a bigger learning rate), which finally let it form opinions about which layers matter. The second problem was that my dataset, PMEmo (about 767 pop-song clips rated by listeners), is lopsided — most songs are upbeat and happy. A lazy model could score well just by guessing "happy" most of the time. I fixed this by showing the rarer emotions (sad, calm, angry songs) more often during training, so the model couldn't coast on the majority.

The results, in plain terms: the model reached an arousal score (R²) of about 0.65, meaning it explains roughly 65% of the variation in how energetic songs are — quite good. Valence landed near 0.51, explaining about half the variation in how positive a song feels — noticeably harder. That valence gap turned out to be a theme of the whole project: how positive a song feels often depends on lyrics and culture, not just sound, so audio-only systems hit a ceiling there.

I also ran an extra experiment with EDA fusion. EDA, electrodermal activity, is a tiny physical signal — small changes in the skin's sweat/conductance that happen automatically when a person feels aroused — and PMEmo recorded it from listeners. I added it because it's a body-based clue about emotion that is completely independent of the audio. It pushed the strict arousal score (CCC) up to about 0.85, my best result on the energetic dimension, which makes sense: arousal is physical, and the body literally reacts to it.

**The takeaway: a single error measure wasn't enough — emotion needed several goals at once, careful fixes for a stubborn model and a lopsided dataset, and even a peek at listeners' bodies, with valence remaining the genuinely hard nut to crack.**

## Making the model explain itself (Phase C)


### RAG Knowledge Base and Phase B/C Transition

#### The Knowledge Base for the RAG System
The **Knowledge Base (or Vector Database)** of your RAG system is not a collection of raw audio files, spectrograms, or text descriptions. It is a collection of **frozen, $L_2$-normalized 128-dimensional continuous vector embeddings** derived from your entire dataset of songs (e.g., the ~600 songs from the PMEmo dataset). 

Every song in your database is represented as a single coordinate point on the surface of a 128-dimensional hypersphere. This geometric space is highly structured because it was optimized using Supervised Contrastive Loss ($\text{SupCR}$) during Phase B. Consequently, songs with identical or highly similar emotional profiles (Valence and Arousal) are clustered tightly together in the same continuous neighborhood.

When a user provides a new query song, the RAG pipeline operates as follows:
1. **Embedding Generation:** The query song is passed through the frozen Phase B encoder, transforming it into its own 128-D vector coordinate.
2. **Vector Retrieval:** The system calculates the distance (typically Cosine or Euclidean distance) between the query coordinate and all the stored song coordinates in the knowledge base.
3. **Context Injection ($k$-NN):** The system extracts the metadata, emotional labels, or proto-features of the $k$-Nearest Neighbors (the closest points on the sphere). This retrieved acoustic context is then fed into your downstream component to generate the final explained output.

#### Latent Space vs. Model Output: The Technical Distinction
The 128-dimensional latent space is the intermediate output produced by the network's bottleneck, sitting exactly one layer before the final regression head. The absolute final output of the Phase B model *during training* consists of the two continuous numbers representing Valence and Arousal. 

#### Phase Handover and Pipeline Mechanics
The architectural transition and data flow between the two phases operate as follows:

* **Creation in Phase B:** During Phase B, the audio passes through the `WeightedLayerFusion` and the auxiliary branches, which are concatenated and forced through a Multi-Layer Perceptron (MLP) bottleneck that shrinks the data down to 128 dimensions. This vector is then $L_2$-normalized to project it onto a hypersphere. This specific 128-D spherical representation is the latent space. A final linear regression head then reads this 128-D coordinate to output the final Arousal and Valence numbers.
* **Transition to Phase C:** Once Phase B finishes training, you essentially "chop off" that final linear regression head and freeze the rest of the weights. The model no longer outputs two numbers; instead, its final output becomes that 128-D latent vector.
* **Inheritance by Phase C:** Phase C inherits this frozen encoder and its resulting 128-D latent space. Because Phase B was trained with the Hybrid Loss (specifically the SupCR contrastive term), this space is already highly organized so that emotionally similar songs are clustered in the same local continuous neighborhoods. Phase C simply runs its $k$-Nearest Neighbors ($k$-NN) retrieval and Audio ProtoPNet classifications directly inside this structured 128-D output space.

> **Defense Presentation Guide Summary:** "The latent space used for Phase C retrieval is the $L_2$-normalized 128-D bottleneck representation extracted directly from the frozen Phase B encoder."
Predicting two numbers is useful, but for my thesis it wasn't enough. A number like "valence 0.4" tells a listener nothing they can feel or trust. The heart of my project was explanation, so Phase C turned the predictions into something a person could actually understand.

I used prototype-based retrieval. In plain words: instead of inventing an explanation out of thin air, the system answers a query by finding real example songs from the collection that sit closest to it in the emotion map, and those real examples *are* the explanation. It's reasoning by analogy — "this song belongs here because it's almost identical to these other songs you can listen to." Because Phase B had already pulled similar songs into tight neighbourhoods, this step worked naturally.

To make the explanations concrete, I leaned on the four emotion quadrants — the four corners of the emotion map: happy/energetic, calm/positive, sad/subdued, and tense/angry. Every song falls into one of these, so the system can say which corner a song lives in and which neighbours share it. It even shows the opposite — songs it deliberately did *not* pick — because explaining "why this and not that" is how people naturally explain things.

This is also where my supervisor pushed me on a key distinction: ante-hoc versus post-hoc explanation. Post-hoc means explaining a black box after the fact, guessing at its reasoning. Ante-hoc means the system's reasoning is transparent *by design* — the explanation is the actual decision process, not a story told afterward. My supervisor strongly preferred ante-hoc, because a guessed explanation can be wrong in ways you can't detect. My retrieval system is ante-hoc at its core: the decision genuinely *is* "these real songs are the nearest neighbours," so the explanation is faithful rather than invented.

**The takeaway: I learned that a trustworthy explanation isn't decoration added at the end — it has to be the model's real reasoning, and designing for that honesty changed how I judged the whole system.**


The Old Method: The Fixed 4-Centroid Readout (The Failure) Initially, I used a very simple method. After the Phase B encoder finished training and organized the songs into the 128-dimensional latent space, I ran a simple script to find the mathematical center (the "centroid") of the four emotion quadrants
.
How it worked: When a new song came in, the system measured its distance to these four fixed points and said, "It is closest to the Happy center."
Why it failed: Because these centers were fixed after training, it was technically a post-hoc guess, not a true ante-hoc classifier
. Worse, it performed terribly. It achieved an accuracy of only 50.6%, which actually lost to a "dumb" baseline of just guessing the majority class "Happy" every single time (61.1%)
. It was especially bad at finding "Sad" songs, successfully recognizing them only 17% of the time
.



The Upgrade: The Audio ProtoPNet (The Success) To fix this, I rebuilt the classifier using an Audio ProtoPNet (Prototypical Part Network)
. Instead of calculating fixed centers after the fact, the ProtoPNet is a learnable prototype network
.
How it works: Inside the 128-D latent space, I initialized 20 prototype vectors (5 for each of the 4 emotion quadrants)
. Instead of keeping them frozen, these prototypes are updated during gradient descent
. The network is trained with separation losses to actively push these prototypes to the exact spots in the space where the classes are actually separable
.
How it makes decisions: When a song enters the network, the model calculates its L2 distance to all 20 prototypes
. It uses an "identity prior," meaning a prototype assigned to the "Sad" category is mathematically only allowed to cast a vote for the "Sad" class
. The song is classified purely based on which prototypes it is geometrically closest to
.

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

### Putting a hard number on the "which-emotion-prototype" feature (and an honest audit)

**What I did.** My supervisor wanted the system to say which of the four emotion prototypes a song matches best — and I'd been showing that as a nice example, but never as a measured number. So I computed it properly: for every song, does its best-matching prototype actually equal its true emotion quadrant? I also went back through my written reports looking for any claim that didn't have a real measurement behind it, and corrected them.

**Why I did it.** A single hand-picked example proves nothing; an examiner will ask "how often is it right?" And I'd started to worry that some of my write-up made confident claims I hadn't actually measured.

**What happened.** The honest number was humbling: the prototype feature is right about 51% of the time, but a dumb rule that just always guesses the most common emotion ("Happy") is right 61% of the time. So my feature loses to the trivial baseline — and it's especially bad on sad songs (right only 17% of the time). The audit also caught two write-up claims I couldn't back: I'd said the raw model's song-map shows separated emotion clusters (it doesn't — it's one blob), and that specific layers "dominate" the model (they don't — the model spreads attention almost evenly). I fixed both. On the positive side, one shaky-sounding claim turned out fine once measured: my retrieval is genuinely about twice as good as random guessing.

**What I learned.** This was the most important honesty pass of the project. The prototype feature is still worth keeping as a *readable explanation* — it shows the four similarity scores transparently — but I can no longer call it an accurate classifier, and I won't. The bigger lesson is a habit: every claim in the thesis should point to a number I actually ran, and where the number is unflattering (the prototype accuracy, the mixed clusters, the near-uniform layers), the honest move is to report it and reframe, not to hide it. The thesis is more credible for saying "here is what did not work, measured" than for overselling.

### Is music-specific pre-training actually better? — MERT vs wav2vec2 alone

**What I did.** I trained the exact same emotion model twice, changing only the backbone: once on MERT (trained on music) and once on wav2vec2 (trained on speech), each on its own. Same loss, same setup, same cross-validation — so any difference is purely the encoder.

**Why I did it.** I'd been assuming MERT is the right backbone because it was trained on music, but I'd never shown it head-to-head against the speech model on its own. An examiner would reasonably ask "how do you know the music model is actually better?"

**What happened.** MERT won on every measure. The clearest gap was on valence (positive-vs-negative feeling), where the music model's agreement score was 0.74 versus the speech model's 0.66. On energy the two were closer but MERT still led.

**What I learned.** This cleanly justifies my choice of MERT as the foundation — music-specific pre-training genuinely helps, especially for the harder, harmony-dependent valence axis, which a speech model has no reason to understand. It also ties a bow on an earlier puzzle: when I combined the two encoders, the speech one looked redundant — now I can see why, because on its own it is simply the weaker of the two.

### Does the fancy "25-layer fusion" actually do anything? — an empirical check

**What I did.** My supervisor asked a sharp question: why combine all 25 layers of the music model with a learnable mixer, instead of just using its final layer like most papers do? I re-trained the exact same emotion model on just the last layer alone, with everything else identical, and compared.

**Why I did it.** It is one thing to have a nice diagram with a learnable layer-mixer; it is another to show it actually buys you accuracy. And I already had a warning sign: the mixer had been learning weights that came out almost perfectly uniform across all 25 layers, which suggests it wasn't really finding anything useful to mix.

**What happened.** The last-layer-alone version scored R² 0.66 (energy) / 0.52 (positivity), versus 0.65 / 0.51 for the full 25-layer mixer — essentially identical, and if anything the simpler version is a hair better on the headline metric. This is a clean negative result that backs up the "uniform-weights" warning sign: on this dataset, the mixer has nothing useful to learn.

**What I learned.** Two honest take-aways. First, on PMEmo the last layer of the music model already carries the signal — adding a learnable mixture over all 25 layers does not improve the regression. Second, I am keeping the mixer in the architecture anyway, but I have changed how I describe it: it is an **interpretability hook** (it lets me *show* which layers the model is weighting), not an accuracy-booster. The thesis is more credible for catching this and stating it plainly than for keeping a sophisticated-sounding component and claiming gains it does not deliver.

### Can we fix the class-imbalance problem with the loss instead of the sampler? — checked, no

**What I did.** Our dataset is lopsided (61% "happy" songs, ~9% each for the other corners), and our current fix re-balances which songs the model sees in each batch. My supervisor asked whether a more sophisticated approach — penalising the loss for getting minority songs wrong, rather than just sampling them more — would do better. I tested four versions side-by-side: the current sampler, a pure loss-penalty, the two stacked, and a "focal" loss that automatically focuses on hard examples. I committed in advance to a strict rule for declaring a winner (the new method must beat the baseline by more than the run-to-run noise on both energy and positivity, or it doesn't count).

**Why I did it.** This is a fair question — there's a whole literature of fancier imbalance fixes and an examiner could legitimately ask why we settled for the simplest one. The honest way to answer it is to run them and see.

**What happened.** None of the three alternatives beat the sampler. The pure loss-penalty was actually a bit *worse* on positivity. Stacking the two was tied. The focal loss was tied. So the sampler we've been using is, empirically, as good as anything else I could try on top of it. I also surfaced a separate honest finding: when I re-ran the baseline as part of this experiment, the numbers came out a bit higher than the original logged baseline — most likely because we never fixed the random seed for initialisation, so each rerun is one draw from a distribution. I'm flagging this in the report rather than replacing the older number with the newer one (that would be cherry-picking).

**What I learned.** Two things. First, the minority-quadrant failure is not "we picked the wrong reweighting trick" — it's "there are 64 sad songs and the model can't learn an emotion structure from 64 examples no matter how I weight them". The honest cure is more data or audio augmentation, not a cleverer loss. Second, the dissertation is stronger for *having* this ablation: I can now say "we tested 4 imbalance treatments under a pre-registered pass mark; only the sampler survived" instead of presenting the sampler as an arbitrary choice. That's the kind of methodological discipline an examiner will reward.

### Does the 25-layer fusion ever earn its keep? — yes, but only in the multi-encoder model

**What I did.** After the single-encoder test showed the fancy layer-mixer wasn't helping (the simpler "just use the last layer" version performed identically), the obvious follow-up was: does the mixer help in our *best* model — the one that combines three encoders (music model + speech model + music-theory features)? My supervisor's question "why all 25 layers and not just the last one?" only really matters if the mixer pays for itself somewhere. So I re-ran the best model with both encoders restricted to their last layer only, and compared.

**Why I did it.** If the mixer doesn't help anywhere, it's dead weight in the thesis — I should remove it and simplify. If it helps somewhere but not everywhere, that's a more interesting and more defensible story. I needed to know which.

**What happened.** This time the mixer clearly won. The "last layer only" version scored R² 0.67 (energy) / 0.49 (positivity) versus 0.72 / 0.57 with the mixer — losses of 5 to 8 percentage points, *outside* the run-to-run noise on both axes. So the mixer is not dead weight; it earns its keep specifically in the multi-encoder setting.

**What I learned.** This gives me a much sharper, more honest answer to the supervisor's question. The mixer's value is not "MERT needs all 25 layers" — it doesn't, on its own. The mixer's value is *cross-encoder coordination*: when the model already has access to the late acoustic representation from three sources (spectrogram CNN, speech model, music-theory features), it benefits from also having access to MERT's *middle* layers, which carry complementary information that the late layer alone can't reach. The single-encoder ablation is the negative control that proves the gain isn't free, and the multi-encoder ablation is the positive result that justifies the architecture. This is the kind of paired empirical answer that I think will hold up in a viva.

### Can we patch the minority-quadrant problem with the standard augmentation trick? — no, and that confirms the data floor

**What I did.** Earlier I argued (but had not measured) that the catastrophic minority-quadrant scores — the model predicts sad/angry songs *worse than the mean* — were a *dataset-size* problem rather than a *method* problem (only 64–67 examples in each of the rare emotion corners). My supervisor would reasonably ask whether I'd actually tried the standard literature fix: **mixup** (Zhang et al. 2017), which artificially expands the training set by linearly mixing pairs of training samples. I ran it on our best model under the same strict pass-mark rule I used before, and compared.

**Why I did it.** I wanted the "data floor" claim to be empirical, not just plausible. If the standard cited augmentation method doesn't fix the minority quadrants, then the data-floor diagnosis holds; if it does fix them, my framing was wrong and I'd need to update the thesis.

**What happened.** Mixup came in statistically tied with the baseline — the headline scores moved by less than the run-to-run noise on all four metrics. More importantly, every minority-quadrant R² remained negative (still worse than predicting the dataset mean). So the simplest cited remedy does not lift the floor.

**What I learned.** Two clean conclusions. First, the data-floor diagnosis is now empirically grounded, not just argued: when you have ~64 examples of "angry" songs, no amount of clever interpolation between them creates the kind of signal a regressor needs to predict them correctly. Second, per the rule I'd pre-agreed with myself ("if the simpler method doesn't help, don't escalate to harder ones"), I'm not going to chase C-Mixup or audio re-extraction. The honest thesis story is: the floor exists, it's a property of the dataset, the standard remedy was tested under a pre-registered pass mark and tied baseline, and the long-term fix is a larger affect-annotated music corpus — which is future work, not in scope for the dissertation.

### Two more baselines: a CNN without any SSL, and a triple with biosignals instead of speech features

**What I did.** I added two more rows to the comparison table — one to plug a hole, one to answer a question.
The hole: I had never tested the simplest non-SSL baseline, a shallow CNN trained on spectrograms with no music or speech foundation model at all. The question: would the physiological (EDA) signal work *as the second branch* alongside MERT and the spectrogram CNN, replacing the speech model that turned out to be redundant?

**Why I did it.** A good SOTA table needs the "what does the simplest thing do?" baseline so that every gain over it can be attributed to a specific architectural choice — otherwise critics can plausibly say "you only beat older methods because you have more parameters". And the triple-with-EDA-instead-of-wav2vec2 question is the most natural follow-up to my earlier finding that the speech model contributed nothing once the music model was in the mix: is *any* second branch interchangeable, or is EDA different because it's a different modality?

**What happened.**
Two clean results. The plain CNN on spectrograms — no SSL — scored 0.65 (energy) and 0.45 (positivity). On energy it *beats* the speech SSL baseline; on positivity it *loses* to both SSL baselines. That makes intuitive sense: energy is essentially loudness and spectral shape (which a CNN gets for free), whereas positivity requires harmonic and tonal understanding that comes from pre-training on music. This nicely localises the SSL contribution: music SSL pays off mainly for *valence*, not arousal.
The new triple — MERT + spectrogram CNN + EDA biosignals — scored 0.71 / 0.57. That is statistically identical to MERT + spectrogram CNN *without* EDA, and to MERT + spectrogram CNN + speech features. In other words, EDA is also redundant once a trainable spectrogram branch is present. The pattern is now consistent across three different second-branches.

**What I learned.** Two cleanly defensible takeaways. First, the SSL story has a sharper boundary than I'd appreciated: SSL earns its keep on *valence*, and a plain CNN can match it on arousal. Second, the redundancy result is no longer about wav2vec2 specifically — it's structural: once MERT + a trainable spectral branch is in the model, additional second branches (speech features, biosignals, music-theory features) all collapse to the same ceiling, presumably because they carry overlapping information that MERT and the spectrogram CNN have already covered. This reinforces the data-bottleneck story over the architecture-bottleneck story: with only ~600 training songs, there's not enough signal for the model to exploit a third independent information source.

### Can we force the latent space into four clean clusters? — measured trade-off, and the answer is "yes but not worth it"

**What I did.** My supervisor (and almost certainly a viva examiner) will see the t-SNE figure showing the model's latent space as a single continuous Happy-dominated blob and ask the obvious question: can you make it look like four separated clusters per the four emotion quadrants? Rather than just argue about it, I ran the ablation. I added a small auxiliary head that classifies songs into the four quadrants alongside the regular regression head, and trained with the combined loss at four different strengths of the auxiliary signal. I measured both regression performance (the metric the system is actually for) and a clustering score (Silhouette).

**Why I did it.** I want the thesis to be able to say "we tested this and here are the trade-offs", not "we argued this was a bad idea but didn't try it". The argument-only answer is fragile under questioning; the empirical answer survives.

**What happened.** Two interesting things, one of which I didn't expect.
First, the unexpected one: the best model already has *moderately good* quadrant structure in its latents (Silhouette around 0.26 on the test set), which is much higher than the near-zero we'd previously reported. The near-zero number came from the single-encoder ablation; with the full multi-encoder model, the latent space is more structured than I'd thought. So the t-SNE figure's "blob" framing actually slightly under-sells what the model does. *[CORRECTION (Step 23 / §2a-sexies): the model effect is ≈0 — single-MERT 0.269 ≈ Enhanced 0.260 cosine, statistically tied. The "multi-encoder is more structured than single-encoder" claim is FALSE/superseded; both are weak-to-moderate ≈0.26. See the later diary entry that overturns this.]*
Second, the trade-off: pushing the classification signal harder did make clusters tighter, but not by very much (Silhouette went from 0.26 to 0.29, a small gain), and the strongest setting cost about 4 percentage points on the energy-prediction score — a real loss outside the noise. The weakest setting was free but did nothing. The honest reading is: the cost-benefit curve does not favour forcing clusters in this setup. And once again, the minority quadrants (sad/calm/angry songs) refused to get better with any of these settings — actually got slightly worse — confirming for the third time that the minority-quadrant failure is about how few sad songs are in the dataset, not about model choice.

**What I learned.** I now have an empirically grounded viva answer instead of a hand-waved one. The continuous latent space isn't a representational failure — Russell's psychological model says emotion *is* continuous along valence and arousal — and when I tested the most obvious "make it look discrete" remedy, it produced only a small clustering improvement at a clear regression cost. So I keep the continuous representation, I have an ablation table that shows I considered the alternative, and I have a sharper story about *why* the multi-encoder model is better than the single-encoder one (the multi-encoder's latents are inherently more quadrant-structured, not less) *[CORRECTION (Step 23 / §2a-sexies): this last parenthetical is FALSE — the matched audit found single-MERT (0.269) ≈ Enhanced (0.260) cosine, statistically tied. Multi-encoder is NOT more structured; the multi-encoder's advantage is on regression accuracy, not latent clustering.]*. This is one of the cleanest examples in the project of an ablation strengthening the defence rather than just confirming the diagnosis.

### Making sure every number is traceable — the logging and audit pass

**What I did.** I stopped to make the project's bookkeeping honest. Every time I'd run an experiment, the results printed to a temporary file that the system would eventually throw away — so the numbers in my report were correct, but I couldn't easily *prove* they matched the actual runs anymore. So I copied every experiment's raw output into a permanent folder inside my reports (one clearly-named log per experiment), and then I wrote a small checking script that reads each log, pulls out the headline scores, and compares them against the exact numbers written in my report files. Anything that didn't match would get flagged.

**Why I did it.** This is the difference between "trust me, the numbers are right" and "here is the proof." An examiner — or future-me, six months from now — should be able to open a results table, find the run that produced it, and confirm the two agree without having to re-run anything. And honestly, after catching myself earlier making claims I hadn't measured, I wanted a mechanical safety net so that kind of drift couldn't creep back in unnoticed.

**What happened.** The checker compared 60 separate numbers (every energy/positivity/agreement/clustering score across all my recent experiments) and found zero mismatches — everything in the report traces back exactly to a real run, within rounding. It also caught a couple of harmless formatting differences in two older logs, which I fixed so the checker reads them cleanly too.

**What I learned.** Reproducibility isn't a thing you do at the end — it's a habit that protects you the whole way. Now the rule is simple: every run's output gets saved with a clear name, and one command re-verifies the whole report against those saved runs before I regenerate the PDF. It's the least glamorous part of the project and possibly the one that will save me the most pain, because "I can prove every number" is exactly the kind of thing a viva defence stands or falls on.

### Hunting down my own contradictions — five conflicts, and a number I had to overturn

**What I did.** I went back through my own reports looking for places where two documents said things that, read side by side, would look like I was contradicting myself. I found five. The biggest was the "clustering score" (Silhouette): some pages said it was ≈0 (no clustering), one said 0.255 (moderate clustering), and they were quietly talking about different models *and* different distance settings without saying so. Rather than just pick a number, I ran one clean experiment that measured the score for both models under both distance settings on held-out songs, so there'd be a single honest answer.

**Why I did it.** A jury reads the whole thing in sequence. If page 12 says "≈0" and page 30 says "0.255" for what looks like the same system, that single inconsistency can cost me more credibility than any weak result, because it makes them doubt *everything*. Better that I find these myself and fix them than have them found for me in the viva.

**What happened.** The clean experiment overturned a claim I'd made. I had written that my best multi-encoder model had *more* emotional structure in its space than the plain single model. Not true: measured properly, head-to-head, they're basically identical (≈0.19 on one distance setting, ≈0.26 on the other, for *both* models). And neither is ≈0 — the old "≈0" numbers turned out to come from an older, in-sample measurement on a saved file, not from a clean held-out test. So the honest story is: the space has *weak-to-moderate* structure (around 0.26), the same for both models, which is well below what cleanly separated clusters would score (~0.5+) — a structured continuum, not four boxes, and not a featureless blob either. I corrected this everywhere it appeared.
The other four: (2) a "~100% accuracy" mode result was based on a broken label that wasn't even measuring major/minor — I removed it and kept the honest 0.673; (3) some stale competitor numbers were only in old correction-notes, never in the live tables, but I flagged them anyway; (4) my "best valence" model only wins by less than the noise, so I now say so plainly; and (5) one of my two added features (musical key) was fed to the model as a plain 0–11 integer, which throws away the fact that keys wrap around in a circle — so it couldn't help, and I now explain that clearly as an encoding mistake with a known fix, not a failure of the underlying idea.

**What I learned.** Two things, both about trustworthiness. First, the most dangerous errors in a thesis aren't the weak results — those are fine if reported honestly — they're the *inconsistencies*, because they make a reader stop trusting your numbers. Hunting them down myself, with a fresh measurement when needed, is some of the highest-value work I can do before submitting. Second, I'm now genuinely glad I built the result-checking script earlier, because every one of these fixes got re-verified against the raw logs automatically (64 numbers, zero mismatches) — so I know the corrected reports actually match the experiments, not just my memory of them.

### Two fixes my examiners would ask for: a smarter key feature, and a prototype model that actually works

**What I did.** Acting on feedback aimed at examination standards, I tackled two specific things. First, the musical-key feature: I'd been feeding it to the model as a plain number 0–11, which is wrong because keys wrap around in a circle (the key "B" is right next to "C", but 11 and 0 look maximally far apart to a model). I re-coded it the proper way, placing each key on a circle using sine and cosine, and then ran a clean head-to-head test (old way vs new way, everything else identical). Second, I rebuilt the part of my system that matches a song to one of the four emotion "prototypes". The old version computed four fixed average points after training and just measured distance to them — and it embarrassingly lost to a dumb "always guess Happy" baseline. I replaced it with a proper *learnable* prototype network (an Audio ProtoPNet), where the prototype points are trained by the model itself during learning, and the song is classified by how close it sits to each one.

**Why I did it.** Both were honest weak spots an examiner would press on. The key feature was a known encoding mistake I'd flagged but not fixed. The prototype method was the one place my "explainable" system underperformed a trivial baseline — which undercuts the whole explainability claim if left unaddressed.

**What happened — one falsified guess, one clear win.**
The key fix was, scientifically, a *negative* result: encoding the key correctly made essentially no difference to how well the model predicts positivity (the change was smaller than the run-to-run noise). **This is important because it overturns something I'd previously written.** I had confidently said key "didn't help because I encoded it badly." That turns out to be wrong — I fixed the encoding and it still didn't help. So the real reason is different and more honest: the link between musical key and how positive a song feels is just too weak to exploit with only ~767 songs, and the harmonic cues that *do* matter for positivity are already captured by the main music model. I'm keeping the correct encoding (it's the right way to do it), but I now tell the true story instead of the assumed one.
The prototype rebuild was a genuine win. The new learnable ProtoPNet scored about 73% on guessing the right emotion quadrant, comfortably beating both the old fixed-prototype method (~51%) *and* the "always guess Happy" baseline (61%) — the first time anything in my project beat that baseline. Best of all, it went from getting sad songs right only 17% of the time to 69% of the time. And it's still fully transparent: each prototype is, by design, evidence for exactly one emotion, so the model can still explain itself.

**What I learned.** Two lessons. First, the discipline of *testing* a convenient explanation instead of asserting it paid off again — I would have happily written "key needs cyclic encoding" in the thesis as settled fact, and it would have been wrong; running the experiment turned a plausible-but-false story into a true one. (Note for myself: this is the second time a "we assumed X was the cause" turned out false once measured — first the clustering score, now the key encoding. The pattern is clear: measure, don't assume.) Second, the prototype result shows that *how* you build an interpretable component matters enormously — the same idea (match songs to emotion prototypes) went from losing-to-a-baseline to beating-the-baseline purely by letting the prototypes be learned rather than fixed. The explainability of my system is now backed by a component that is both honest and accurate, not one or the other.

### Testing whether my "emotion is continuous" claim is real — and getting a surprise that made it stronger

**What I did.** A central claim of my thesis is that the model's internal space lays emotions out as a smooth continuous gradient (the way psychology says emotion actually works — Russell's circumplex), rather than four tidy separate clusters. I measure this with a "clustering score" (Silhouette ≈ 0.26 — low, meaning not-very-clustered). But a sharp examiner could object: "maybe your score is low not because emotion is continuous, but because your model simply *can't* form clusters." To settle that, I trained the very same model with a completely different goal — explicitly to *classify* songs into the four emotion corners and to push those corners apart — and then measured the clustering score again.

**Why I did it.** If the model, when *told* to make clusters, still couldn't, that's weak evidence either way. But if it could classify well yet still didn't form clean clusters, that would prove the continuous layout isn't a failure — it's the genuine shape of emotion. I wanted to convert a claim I'd been *asserting* into one I'd *tested*.

**What happened — and I'll be honest that my prediction was wrong.** I expected the classification-trained model to show a *higher* clustering score (tighter clusters). It came out *lower* (0.18 vs 0.26) — even though it classified the four emotions correctly 74% of the time. At first this seemed backwards, but it's actually the strongest possible result for my argument. The model recognises each emotion using several scattered "prototype" points, so it can classify a sad song accurately without all sad songs sitting together in one tight clump. In plain terms: **being able to tell emotions apart and having them form neat separate clusters are two different things** — you can do the first without the second. So now I have three different training setups (my main predictor, a cluster-forcing version, and this classifier) and *all three* land in the same narrow low range (0.18–0.29), nowhere near what real separated clusters would score. 

**What I learned.** The continuous-gradient story isn't a limitation of my model — it's a property of emotion itself, and now I can prove it three independent ways instead of asserting it once. This is (I notice) the third time in this project that running the experiment overturned what I assumed and left me with a *better* story than the one I'd have written from intuition — first the clustering claim, then the musical-key fix, now this. I'm increasingly convinced the most valuable habit I've built is refusing to write "X because Y" until I've actually measured Y. (One small honesty note: the script I wrote even printed a conclusion line assuming the result would go the other way — I left the real numbers as the record and corrected the interpretation, rather than quietly rewriting history.)




### Making sure the retrieval system actually uses my best model — and a symmetry I had to build, not just check

**What I did.** I audited my explanation/retrieval system for a worry that had been nagging me: does it use the *same* best model to encode both the song library and a new query song, or did I accidentally leave a simpler model hardcoded somewhere? I traced every retrieval script line by line.

**Why I did it.** If the database were built with one model and queries encoded with another, every "similar song" result would be comparing apples to oranges — silently wrong, and very hard to spot. For an explainable system, that would undermine everything.

**What happened — the reality was different from what I assumed, in two ways.** First, I had assumed there were two encoders (one for the library, one for queries) that might disagree. There weren't: the query path never actually re-encoded anything — it just looked up a pre-computed vector by song ID. So "symmetry" was technically guaranteed, but only because the system could only ever query songs already in its library; there was no way to encode a genuinely new song at all. Second, and more importantly, the retrieval scripts were hardwired to the *simplest* single-encoder MERT, not my best multi-encoder (Enhanced) model, and my new learnable-prototype classifier (ProtoPNet) wasn't connected to the retrieval system at all. So my best work simply wasn't in the deployed pipeline.

**What I did about it.** Rather than patch the old scripts, I built one shared encoder that both the library-building step and the query step call — the exact same function, so they *cannot* drift apart by design. It uses the full best model (music + speech + the corrected cyclic-key music-theory branch), produces the 128-number fingerprint, and normalises it, identically for every song. Then I proved the symmetry empirically: I re-encoded all 767 library songs through the query path and checked they matched the stored library vectors — they matched to six decimal places (the tiny difference is just floating-point rounding). I also wired in the ProtoPNet so the emotion-prototype explanation now uses the accurate learnable version instead of the old weak averaging method. One honesty caveat I'm careful about: the deployed model is trained on all songs (so the system has a fingerprint for every track), which means I must NOT quote its accuracy as a real score — the honest performance numbers stay the held-out cross-validation ones.

**What I learned.** "Are the two encoders the same?" turned out to be the wrong question — the right one was "is there even a real encoder on the query side, and is it the best model?" The answer was no on both counts, and finding that took tracing the actual code rather than trusting my mental model of it. The fix isn't just *checking* symmetry, it's *enforcing* it structurally — one shared function means there's nothing to keep in sync by hand. That's a more robust kind of correctness than "I verified it once."

### Giving my benchmark model its own retrieval score — and finally explaining the clustering-score mystery

**What I did.** After making the explanation system use my best model (the multi-encoder "Enhanced" one) for both the library and the queries, I realised the retrieval-quality number I'd been reporting (how often the songs it pulls back are genuinely emotionally similar) was measured on an *older, simpler* model. So I measured the best model's retrieval quality properly — the honest way, where every song is scored by a model that never trained on it — using the exact same yardstick as the older numbers so they're directly comparable.

**Why I did it.** If I'm going to call the Enhanced model my benchmark and deploy it, its retrieval score has to be a real, measured number for *that* model, not inherited from a different one. Otherwise the thesis claims one thing and the system does another.

**What happened — good news, and a mystery solved.** Good news first: the best model is also the best retriever. Its Precision@5 came out at 0.594, edging out the next-best model (0.585) and more than double random guessing (0.276). So the model I deploy is best at everything — predicting energy, and finding similar songs. Clean story.
The mystery: this same run reported the "clustering score" (Silhouette) as basically zero, even though my careful audit had established it as 0.26. For a moment that looked like a contradiction. But I worked out exactly why, and it's instructive: the retrieval index is assembled by pooling song-fingerprints from *five different models* (one per cross-validation fold). Each of those five models lands its fingerprints in a slightly different orientation, so when you pool them and ask "are the four emotion regions tightly separated *across the whole pool*?", the answer collapses to zero — not because any single model is disorganised, but because you've mixed five differently-oriented maps together. The retrieval score survives this mixing (it only cares about *local* neighbours), but the clustering score doesn't (it's a *global* measure). 

**What I learned.** This finally and precisely explains the "is it zero or is it 0.26?" question that has haunted this project: **zero** is what you get when you pool five models' maps for the retrieval index; **0.26** is the honest within-one-model number. They're both correct, for different measurements — and the difference is a property of *pooling*, not of the model. It also re-confirms the methodological line I've held all along: trust Precision@k (robust, local) over Silhouette (fragile, global) for judging whether the space is emotionally organised. Pleasingly, the experiment I ran to give my benchmark model a fair retrieval score also handed me the cleanest possible explanation of a months-old puzzle.

## Getting the RAG explainer's story straight (Phase C write-up)

When I started writing up the explainable-retrieval part of the thesis, I almost wrote down the wrong tool. An earlier note said the natural-language step used Llama 3.2 through Ollama — but when I actually checked the server, Ollama isn't even installed, and the real exported explanations were produced by **Qwen2.5-3B-Instruct** run straight through HuggingFace `transformers` (no daemon, no admin rights, which is exactly why it works on the university machine). I corrected Chapter 3 and the report to say Qwen, and removed the Llama reference. The lesson, again: check what actually ran, not what I remembered.

Two more things became clear while documenting it. First, the "foils" (the *why-not-these* songs) are, in the code, simply the most dissimilar songs by cosine similarity — not the cleverer "same tempo but opposite mood" hard negatives I might have liked to claim, so I wrote the honest definition. Second, the system splits cleanly into a deterministic Layer 1 (the numbers) and an LLM Layer 2 (the prose), which lets me judge the prose on *faithfulness* alone — does it ever say something the numbers don't support? I found a real failure case (song 282, a track sitting dead-centre on the emotion plane) where the model wrote "high arousal and high valence" even though the retrieved neighbours were mixed and slightly negative. Rather than hide it, I used it as a case study: it shows exactly why keeping the numbers separate and inspectable matters.

**What I learned.** The explanation system's honesty comes from its structure: because Layer 1 can't hallucinate and Layer 2 may only rephrase it, any mistake is visible by reading the prose against the template — and the borderline song made that concrete.

## Running the faithfulness check and fixing the PDF undefined references

**What I did.** Two cleanup tasks: (1) ran `eval_rag_faithfulness.py` — the script that scores how faithfully the Qwen LLM prose stays to the deterministic Layer-1 template, across the 5-song export. (2) Fixed three `\ref{}` labels in the thesis that were producing `??` in the compiled PDF.

**Faithfulness results (n=5, illustrative).** The script recomputed top-5 and foils from `prototypes_dual.npy` and confirmed the recomputed sets exactly match the printed export (sanity check passed for all 5 songs). Scores:
- **ID-grounding precision = 1.00** — song 562 named 6 song IDs in its prose (296, 91, 821, 99, 704, 457) and all 6 are within {query ∪ top-5 ∪ foils}. The other 4 songs used only generic language with no explicit IDs.
- **Directional faithfulness = 4/5 songs, 9/10 axis-claims (0.90)** — the single failure is song 282, which claimed "high valence" while its neighbours' mean valence was 0.438 (mildly negative, spanning 3 quadrants). This is exactly the over-generalisation case study already in the thesis.

The numbers confirm the two case studies written in Chapter 5 are representative, not cherry-picked. Song 562 (clean case) is fully grounded; song 282 (borderline) is the only failure.

**PDF `??` fixes.** Three cross-references in the thesis were pointing to labels that didn't exist yet:
1. `\ref{chap:state_of_art}` in the appendix — simple typo; the real label is `chap:state_of_the_art`. Fixed.
2. `\ref{sec:results-phaseA}` in Chapter 4 — no dedicated Phase A results section existed in Ch5. Added `\label{sec:results-phaseA}` as a secondary label on §5.4 (Cyclic Key Encoding), which is where the Phase A probing null result is reported. Fixed.
3. `\ref{sec:results-protopnet}` in Chapter 4 — pointed to the prototype results discussion; added `\label{sec:results-protopnet}` right before the `\paragraph{The prototype classifier.}` in §5.7. Fixed.

**What I learned.** The faithfulness check produced a clean quantitative summary that now appears in the thesis (4/5, 0.90) as a measured example rather than only the qualitative description of the two songs. The number is honest about its scope: it's illustrative over n=5, not a claim about general LLM reliability.
