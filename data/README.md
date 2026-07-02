# Dataset Setup — PMEmo 2019

This project uses **PMEmo 2019** (Zhang et al., 2018, DOI:10.1145/3206025.3206037),
a corpus of 794 popular-music chorus excerpts with continuous valence–arousal
annotations and simultaneous electrodermal-activity (EDA) recordings.

The dataset is **not** included in this repository. To set it up:

1. Download PMEmo 2019 from its official source.
2. Place it under `data/pmemo/` with the following structure:

   ```
   data/pmemo/
   ├── audio/          # chorus excerpts (MP3)
   ├── annotations/    # static valence–arousal labels (static_annotations.csv)
   └── EDA/            # physiological signals
   ```

3. If the dataset lives elsewhere, point the code at it instead of moving files:

   ```bash
   export PMEMO_ROOT=/path/to/PMEmo2019
   ```

   All paths are resolved through `configs/config.py` (`PMEMO_ROOT`, default `./data/pmemo`).

## Feature extraction

The models consume **pre-extracted, frozen** backbone features (computed once and
cached to disk). Generate them with:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python src/extraction/extract_pmemo.py          # MERT-v1-330M  -> pmemo_mert_all_layers.pt   (25 x 1024)
python src/extraction/extract_pmemo_wav2vec.py  # wav2vec2-base -> pmemo_wav2vec_all_layers.pt (13 x 768)
python src/extraction/extract_pmemo_melspec.py  # log-mel CNN input -> pmemo_melspec.pt
```

Output tensors are `torch.save` dictionaries keyed by PMEmo `musicId`. Their
locations are configurable via `configs/config.py` / the `MER_FEATURES_DIR`
environment variable; large tensors are gitignored.
