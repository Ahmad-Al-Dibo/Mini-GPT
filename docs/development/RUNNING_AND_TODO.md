# Running and TODO Audit

Last checked: 2026-06-08

## Correct way to run

Install dependencies:

```bash
pip install torch numpy
```

Train or load the main modular model:

```bash
python train.py
```

`train.py` uses a bounded token sample from `data/train-00031-of-00080.txt` and saves to `output/mini_gpt_generalized.pth`. If the large data file is unavailable, it falls back to `data/data.txt`. If the checkpoint exists and the vocabulary/architecture/training strategy matches, it loads the model and generates a short sample. If the checkpoint is missing or incompatible, it trains a new model first. Current improved training uses an 80/20 validation split, early stopping, dropout, gradient clipping, and best-validation restoration.

Run basic generation with the existing `output/mini_gpt.pth` checkpoint:

```bash
python examples/example1_generation.py
```

Run diagnostics:

```bash
python evaluation/diagnostic.py data/data.txt
```

Run the improvement phases:

```bash
python evaluation/optimize.py data/data.txt
python evaluation/instruction_tuning.py data/data.txt
python evaluation/roadmap.py data/data.txt diagnose
python evaluation/roadmap.py data/data.txt all
```

`roadmap.py all` can take longer because it runs diagnosis, optimization, and instruction tuning.

## Verified commands

These commands were tested successfully:

```bash
python -m py_compile <all python files>
python train.py
python examples/example1_generation.py
python evaluation/diagnostic.py data/data.txt
python evaluation/roadmap.py data/data.txt diagnose
python evaluation/optimize.py data/data.txt
python evaluation/instruction_tuning.py data/data.txt
python -m unittest discover -s tests
```

## Current model findings

- The model can generate coherent short facts from the small `data/data.txt` dataset.
- Diagnostics show strong overfitting: train loss drops while validation loss stays high.
- Memorization is high, around 85 percent token accuracy on training-data samples.
- Instruction tuning technically runs, but the generated instruction model overfits because only 27 instruction pairs are created from the current small data file.

## Fixed during this audit

- Example scripts can now be run directly from the project root.
- `train.py` now generates a short sample instead of 10,000 tokens.
- Runtime Python files were normalized to ASCII output to avoid Windows `UnicodeEncodeError`.
- `evaluation/diagnostic.py` now delegates to the ASCII-safe diagnostic implementation.
- `evaluation/diagnostic_simple.py` now loads the real `output/mini_gpt.pth` checkpoint format.
- `evaluation/optimize.py` now passes numeric architecture values into advanced training.
- `evaluation/optimize.py` now uses the real checkpoint format in sampling comparison.
- `evaluation/instruction_tuning.py` now defines the generation device correctly during testing.
- `evaluation/roadmap.py` now handles the `all` phase correctly.
- `gpt_v2.py` now reads `data/data.txt` instead of a missing root-level `data.txt`.

## Still to do

- Clean up mojibake in existing Markdown files such as `README.md`, `LIBRARY_README.md`, `ADVANCED_FEATURES.md`, `HUIDIGE_SITUATIE.md`, `GPT-overview.md`, and `evaluation/README.md`.
- Add a `requirements.txt` or `pyproject.toml` with at least `torch` and `numpy`.
- Add automated tests for tokenizer, dataset, model forward pass, checkpoint load/save, and generator sampling.
- Add broader automated tests for generator sampling and evaluation scripts.
- Add resume-training support from checkpoint.
- Improve instruction data generation; 27 pairs is too small for useful instruction following.
- Add perplexity, token accuracy, vocabulary coverage, and response diversity metrics.
- Compare `output/mini_gpt_generalized.pth` against older checkpoints after several training runs.
- Decide whether `gpt_v1.py` and `gpt_v2.py` are legacy scripts or should be maintained alongside `gpt_lib`.
