# Improvement Log

## 2026-06-08 - Generalization-focused training start

### Changed `gpt_lib/config.py`

- Added `weight_decay` so training can use AdamW regularization from config.
- Added `validation_split` so training scripts can reserve validation data.
- Added `early_stopping_patience` and `early_stopping_min_delta` so training can stop when validation loss stops improving.
- Added `restore_best_model` so the best validation weights can be restored before saving.
- Added `seed` for reproducible training runs.
- Included all new fields in `Config.to_dict()` so checkpoints store the full training setup.

### Changed `gpt_lib/trainer.py`

- Added best validation tracking.
- Added a copy of the best model weights during validation.
- Added `restore_best_model()` and call it after validation training finishes.
- Added `min_delta` support for early stopping.
- Kept the existing save/load checkpoint format compatible.

### Rebuilt `train.py`

- Changed the main checkpoint from `output/mini_gpt_advanced.pth` to `output/mini_gpt_generalized.pth`.
- Added an 80/20 train/validation split.
- Build the tokenizer on train tokens instead of the full dataset to avoid validation leakage.
- Added AdamW `weight_decay=1e-4`.
- Reduced learning rate from `1e-3` to `8e-4`.
- Reduced max epochs from `175` to `100`.
- Added early stopping with patience `10`.
- Restores the best validation model before saving.
- Uses safer generation defaults: lower temperature, repetition penalty, and top-p sampling.
- Prints training configuration and dataset split information.

### Goal of this phase

The goal is not to make the tiny dataset magically instruction-following yet. The first target is reducing memorization pressure and making the saved `train.py` model represent the best validation checkpoint instead of the final overfit checkpoint.

## 2026-06-09 - Regularization, metrics, and safety pass

### Changed `gpt_lib/config.py`

- Added `dropout` so model regularization can be configured from training scripts.
- Added `grad_clip` so training can limit exploding gradients from config.
- Included both fields in `Config.to_dict()` so checkpoints record these settings.

### Changed `gpt_lib/model.py`

- Added dropout to attention weights.
- Added dropout after token plus position embeddings.
- Added dropout at the end of each feed-forward block.
- Replaced `ReLU` with `GELU` in the feed-forward block, which is more common in GPT-style transformer blocks.
- Kept constructor defaults backward compatible with `dropout=0.0`.

### Changed `gpt_lib/trainer.py`

- Added optional gradient clipping through `config.grad_clip`.
- Added validation perplexity tracking.
- Added validation token-accuracy tracking.
- Stored `val_perplexities` and `val_token_accuracies` in checkpoint training history.
- Restored these metric histories when loading checkpoints.
- Validation output now prints loss, perplexity, token accuracy, generalization gap, and epoch time.

### Changed `train.py`

- Added `dropout=0.1` to the default improved training config.
- Added `grad_clip=1.0` to the default improved training config.
- Passed dropout into `MiniGPT`.
- Added metadata checks before loading an existing checkpoint.
- Retrains instead of crashing when checkpoint vocab, embedding size, block size, or number of blocks is incompatible.
- Retrains when the stored training strategy does not match the current improved strategy, including dropout, weight decay, learning rate, or validation split.
- Changed validation splitting from "last 20 percent of tokens" to a seeded shuffled sentence-level split when sentence boundaries are available. This avoids making validation a completely different tail section of the tiny dataset.
- Changed default data source from the tiny `data/data.txt` to a bounded sample from `data/train-00031-of-00080.txt`, with fallback to `data/data.txt` if the large file is unavailable.
- Changed data loading to stream tokens until `max_data_size` instead of reading the full source file into memory.
- Reduced `max_vocab` to `5000` and `max_data_size` to `20000` for a more realistic but still CPU-manageable default training run.
- Changed batch size from `64` to `128`, max epochs from `100` to `30`, and early-stopping patience from `10` to `5` for the larger default corpus sample.

### Added `tests/test_core.py`

- Added tokenizer round-trip coverage.
- Added dataloader shape coverage.
- Added model forward-shape coverage.
- Added trainer validation metric and checkpoint-save coverage.

### Goal of this phase

This phase makes overfitting easier to observe and reduces training instability. Dropout and gradient clipping are now available by default in `train.py`, and validation quality is measured with more than loss.

### Changed `evaluation/diagnostic_simple.py`

- Diagnostics now prefer `output/mini_gpt_generalized.pth` and fall back to `output/mini_gpt.pth`.
- Diagnostics pass checkpoint dropout config into `MiniGPT` for architecture consistency.
- Changed the generalization diagnostic from "train for 5 epochs and measure" to pure evaluation on train/validation splits. The diagnostic no longer mutates model weights while measuring.
- Diagnostics now load bounded token samples instead of reading an entire large corpus file into memory.
- Knowledge representation in the diagnostic is now scored from prompt keyword matches instead of being inferred from memorization accuracy.


## 2026-06-20 - MiniGPT Project Organization

### ✅ Documentation Organization
- Created `docs/00_START_HERE.md` - Clear entry point
- Organized 5 main documentation files with numbering
- Created `docs/legacy/` folder for old documents
- All documentation now easily accessible

### ✅ Directory Guides (README files)
- `models/README.md` - Pre-trained models guide
- `examples/README.md` - Usage examples guide
- `data/README.md` - Data management guide
- `datasets/README.md` - Knowledge base guide
- `evaluation/README.md` - Evaluation tools
- `tests/README.md` - Testing guide
- `config/README.md` - Configuration guide

### ✅ Main Project Files
- Created `README_NEW.md` - Professional new README
- Created `ORGANIZATION_GUIDE.md` - Full organization plan
- Created `ORGANIZATION_SUMMARY.md` - What was organized
- Created `CLEANUP_GUIDE.md` - Optional cleanup steps



### 📁 New Structure

```
Mini-GPT/
├── 📄 README.md (or README_NEW.md)         ← START HERE
├── 📄 ORGANIZATION_GUIDE.md                ← Organization info
├── 📄 CLEANUP_GUIDE.md                     ← Optional cleanup
├── 📄 app.py                               ← Web server
├── 📄 requirements.txt                     ← Dependencies
│
├── 📁 docs/                                ← DOCUMENTATION (START HERE)
│   ├── 00_START_HERE.md                    ✨ MAIN ENTRY POINT
│   ├── 01_QUICK_START.md
│   ├── 02_FULL_DOCUMENTATION.md
│   ├── 03_API_REFERENCE.md
│   ├── 04_ARCHITECTURE_DEEP_DIVE.md
│   ├── 05_DATASET_GUIDE.md
│   └── legacy/                             ← Old docs
│
├── 📁 examples/                            ← USAGE EXAMPLES
│   ├── README.md                           ✨ NEW
│   └── *.py files
│
├── 📁 models/                              ← PRE-TRAINED MODELS
│   ├── README.md                           ✨ NEW
│   └── *.pth files
│
├── 📁 data/                                ← TRAINING DATA
│   ├── README.md                           ✨ NEW
│   └── training/
│
├── 📁 datasets/                            ← KNOWLEDGE BASE
│   ├── README.md                           ✨ NEW
│   └── *.md files
│
├── 📁 config/                              ← CONFIGURATION
│   ├── README.md                           ✨ NEW
│   └── *.py files
│
├── 📁 evaluation/                          ← EVALUATION
│   ├── README.md
│   └── *.py files
│
└── 📁 tests/                               ← TESTS
    ├── README.md                           ✨ NEW
    └── test_*.py files
```





## 2026-06-22 - MiniGPT Planning for fixing, adding or improving features.

