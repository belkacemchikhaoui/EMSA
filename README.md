# EMSA: Empathic Multimodal Shopping Assistant

Reference implementation accompanying the paper *"EMSA: An Empathic
Multimodal Shopping Assistant for Context-Aware Recommendations"*
(Chikhaoui & Alabed).

This repository implements:

- **Multimodal Fusion Encoder (MFE)** — ResNet-50 visual encoder, BERT text
  encoder, multi-head cross-attention fusion (Eq. 1).
- **Empathic Dialogue Manager (EDM)** — transformer decoder policy, the
  three-component empathy score (Eq. 2–6), and the empathy-aware PPO reward
  (Eq. 7).
- **Contextual Recommendation Engine (CRE)** — dynamic utility weighting
  over visual similarity, preference similarity, and context similarity
  (Eq. 8–9).
- Seven baselines used for comparison in the paper's tables
  (VisualBERT-CRS, MM-Dialog, EmoRec, CosRec, ReDial-BERT, UniMIND, EMMA),
  implemented as simplified-but-faithful versions of their published
  architectures so that all models are trained and evaluated under an
  identical pipeline.
- Full evaluation suite: HR@k / NDCG@k, dialogue quality (BLEU/ROUGE/PPL),
  the empathy score, ablations, noise-robustness, cold-start analysis,
  computational efficiency, and paired significance tests.

## Important note on results

**This codebase does not hard-code, bias, or guarantee any outcome.**
Every table/figure script in `evaluation/` computes metrics directly from
model outputs on held-out data. If you rerun this pipeline, the numbers
you get are the numbers that come out — they may differ from the
illustrative values currently shown in the manuscript draft (which were
placeholders pending a full run on real hardware). Before submission,
regenerate every reported number from an actual run of this code on the
real VOGUE, PixelRec, and MM-SHOP data, and update the paper's tables to
match the logged output in `results/`. Do not report numbers that were not
produced by this pipeline.

## Datasets

| Dataset | Type | Source |
|---|---|---|
| VOGUE | Human-human fashion dialogues (60 conversations) | Public — see `data/vogue_loader.py` for expected schema |
| PixelRec | 3.9M interactions, 408K product images | Public — see `data/pixelrec_loader.py` for expected schema |
| MM-SHOP | 1,200 simulated shopping dialogues, live video, user profiles | Original, collected by the authors — see `data/mmshop_loader.py` |

None of the raw data is redistributed in this repository. Point the config
at your local copies (see `configs/config.yaml`).

### Expected directory layout

**VOGUE** (confirmed against the real dataset):

```
<VOGUE_ROOT>/
├── conversation_trials/
│   ├── transcripts/
│   │   └── c{conversation_id}_{catalogue}_{scenario}.json   # e.g. c1_a_1.json
│   └── item_ratings/
│       ├── assistant_ratings.csv
│       └── seeker_ratings.csv
├── fashion_profiles/     # not yet wired in -- see data/vogue_loader.py docstring
├── metadata/              # item images/attributes -- path layout not yet confirmed, see
│                           # VOGUELoader._resolve_item_image() for the exact fallback paths tried
├── supplements/           # not used by the loader
└── surveys/                # not used by the loader
```

Each transcript JSON contains `conversation_id`, `catalogue` (`"a"`/`"b"`/`"c"`), `gt_items`/`mentioned_items` (local item indices 1-12 within that catalogue), and `conversation_content` (turns with `role`, `utterances`, and speech-act `tags`). The two ratings CSVs are keyed by `(participant_id, conversation_id, catalogue, scenario)` with `item_1..item_36` rating columns (three 12-item catalogue blocks concatenated, `-1` where not applicable). See the module docstring in `data/vogue_loader.py` for the full mapping between local item indices, catalogue offsets, and the stable item ids used internally (`"{catalogue}_item_{local_idx}"`, e.g. `"a_item_2"`).

`metadata/` (item images/attributes) and `fashion_profiles/` (participant style profiles) have not yet been confirmed against the real files -- `VOGUELoader` currently falls back to a ratings-informed placeholder feature vector per item and skips participant style profiles entirely. **Before trusting any VOGUE results, confirm these two directories' real schema and update `VOGUELoader._resolve_item_image()` / `build_item_feature_lookup()` accordingly** (share a sample file from each, the same way the transcripts/ratings schema was confirmed, and the loader can be extended).

**PixelRec** (confirmed against the real dataset):

```
<PIXELREC_ROOT>/
├── interaction.csv       # item_id, user_id, timestamp  (implicit positive interactions, no rating column)
├── item_meta.csv         # item_id, view_number, comment_number, thumbup_number, share_number,
│                          # coin_number, favorite_number, barrage_number, title, tag, description
├── features/
│   ├── image_feature.json  # pre-extracted image embeddings -- internal layout auto-detected at
│   ├── text_feature.json   # load time (dict-of-vectors or list-of-records); see
│   └── readme.md            # data/pixelrec_loader.py::_load_feature_json() for details
└── cover/                 # cover images, assumed named "<item_id>.jpg"; adjust
                             # PixelRecLoader._resolve_cover_path() if yours differ
```

Every interaction row is treated as an implicit positive engagement (per the PixelRec paper: a comment implies the user saw the cover), used to build per-user interaction sequences for ranking. `PixelRecLoader.build_item_feature_lookup()` uses the real `image_feature.json`/`text_feature.json` vectors where available for an item, and only falls back to a synthetic placeholder for items missing from both files -- check the log line `"PixelRec item features: N/M items have real pre-extracted features"` after loading to confirm real coverage.

**Fast debug runs on a large `interaction.csv`:** set `pixelrec.max_users` in `config.yaml` (e.g. `5000`) to cap the number of users processed instead of running the full dataset -- useful for a quick smoke test before committing to a full multi-hour run. Leave it `null` for the full dataset. `PixelRecLoader.load_all_samples()` also now prints a `tqdm` progress bar over users, so a large dataset shows live progress instead of running silently.

**Memory scaling:** at full PixelRec scale (~3.9M interactions), each sample carries a candidate pool of `num_negatives + 1` item-id strings. The Python object overhead of millions of such samples adds up fast -- with the old default of `num_negatives=99`, this can reach 30-40GB+ and OOM-crash the process (on Windows, with **no Python traceback at all**, which looks identical to a random hang). The default is now `num_negatives=19`, and `PixelRecLoader` logs an upfront memory estimate (and a warning above ~8GB) before building the sample list, so you get a clear signal before committing to a run rather than a silent crash partway through. Raise `num_negatives` back up only if your machine has the RAM headroom the log estimate implies, or combine it with a `max_users` cap.

**Memory note:** `image_feature.json`/`text_feature.json` can be several GB for the full PixelRec catalog. Loading them with a plain `json.load()` reads the whole file into memory as text and then again as Python objects, which can exceed available RAM (`MemoryError`) even on machines with substantial memory. This loader instead streams the file with `ijson`, keeps only vectors for items actually in your `item_meta.csv`, stores them as compact float16 arrays, and caches the result to an `.npz` file next to the source JSON (auto-invalidated if the source file changes) -- so the expensive parse only happens once. If you still hit a `MemoryError` on the first run, your catalog itself may be large enough that even the filtered, streamed version doesn't fit; consider downsampling `item_meta.csv`/`interaction.csv` to a smaller PixelRec variant (e.g. PixelRec50K) first.

**MM-SHOP** (placeholder schema -- not yet confirmed against real files, unlike VOGUE/PixelRec above):

```
<MMSHOP_ROOT>/
├── dialogues.json          # list of {dialogue_id, turns, user_id, context_tags, emotion_label, gold_response}
├── video_features/         # <dialogue_id>_<turn>.npz  (pre-extracted ResNet-50 region features, shape [N, dv])
└── user_profiles.json      # user_id -> historical preference vector / interaction summary
```

If your actual PixelRec/MM-SHOP files use a different schema, adjust `data/pixelrec_loader.py` / `data/mmshop_loader.py` the same way `data/vogue_loader.py` was corrected -- share a sample file from each real directory and the loader gets updated to match, rather than guessed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n emsa python=3.10 && conda activate emsa
```

**Install PyTorch with CUDA support first, before the rest of the requirements.** The plain `torch` wheel on PyPI is CPU-only on Windows/Linux unless installed from PyTorch's own index -- installing it generically will silently give you a CPU build with no error, and `training.utils.get_device()` will just quietly fall back to CPU. Pick the command matching your CUDA driver version from https://pytorch.org/get-started/locally/, e.g. for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # should print ...+cu118 True
```

Then install the rest (this will see `torch` already satisfies the version constraint and won't touch it):

```bash
pip install -r requirements.txt
```

Edit `configs/config.yaml` with your local dataset paths and GPU settings.

## Running the pipeline

```bash
# 1. Train EMSA and all baselines on a chosen dataset
python -m training.train --dataset vogue --config configs/config.yaml
python -m training.train --dataset pixelrec --config configs/config.yaml
python -m training.train --dataset mmshop --config configs/config.yaml

# 2. Run full evaluation (metrics, ablations, robustness, cold-start,
#    efficiency, significance testing) and write results/*.json + figures
python -m evaluation.run_all --config configs/config.yaml
```

or simply:

```bash
bash scripts/run_full_pipeline.sh
```

Every script writes machine-readable results to `results/` (JSON/CSV) and
regenerates the figures under `results/figures/` from those files — figures
are never drawn from numbers that didn't come out of an actual run.

## Repository structure

```
emsa/
├── configs/config.yaml
├── data/           # dataset-specific loaders (isolate all parsing quirks)
├── models/          # MFE, EDM, CRE, and the assembled EMSA model
├── baselines/        # 7 baseline architectures, same training/eval harness
├── training/          # training loop + PPO trainer for the dialogue policy
├── evaluation/        # metrics, ablation, robustness, cold-start, efficiency
├── scripts/            # convenience shell scripts
└── notebooks/           # thin exploratory/demo notebook (not the source of truth)
```

