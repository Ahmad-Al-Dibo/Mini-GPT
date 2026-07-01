# ArcLM

ArcLM is a compact PyTorch library for local causal language-model training,
fine-tuning, checkpoint loading, and text generation.

The project is intentionally small: it focuses on readable training workflows
for compact models, local datasets, teaching, experiments, and checkpoint
adaptation. It is not a distributed training system.

## Implemented Scope

- Train a compact ArcLM causal language model from local text data.
- Fine-tune or continue training from compatible checkpoints.
- Save checkpoints with model weights, optimizer state, config, training
  history, vocabulary, token mappings, and tokenizer metadata.
- Reload saved checkpoints for text generation.
- Load ArcLM checkpoints, raw PyTorch state dicts, safetensors files, and
  Hugging Face causal language model sources through the loader registry.
- Write training metrics through a queue-backed asynchronous JSONL logger.
- Run focused diagnostics, tokenizer coverage reports, perplexity calculation,
  and top-k prediction helpers.
- Use a small optional Flask prediction API from `app.py`.

## Requirements

ArcLM is packaged as a Python library.

Current package constraints:

| Dependency | Constraint | Purpose |
| --- | --- | --- |
| Python | `>=3.9` | Runtime |
| PyTorch | `>=2.1,<3` | Model, tensors, training loop |
| NumPy | `>=1.24,<3` | Numeric utilities |
| SentencePiece | `>=0.2,<0.3` | Optional SentencePiece tokenizer |
| Transformers | `>=4.38,<6` | Optional Hugging Face source loading |
| Flask | `>=3,<4` | Optional local prediction API |
| Pytest | `>=8` | Tests |
| Build | `>=1.2,<2` | Package build |
| Twine | `>=5,<7` | Package validation/upload |

Dependency check performed in 2026-06 against PyPI:

| Package | Latest observed stable version | Python requirement |
| --- | --- | --- |
| torch | `2.12.1` | `>=3.10` |
| numpy | `2.5.0` | `>=3.12` |
| sentencepiece | `0.2.1` | `>=3.9` |
| transformers | `5.12.1` | `>=3.10.0` |
| pytest | `9.1.1` | `>=3.10` |
| build | `1.5.0` | `>=3.10` |
| twine | `6.2.0` | `>=3.9` |
| flask | `3.1.3` | `>=3.9` |
| safetensors | `0.8.0` | `>=3.10` |

The package keeps broad version ranges in `pyproject.toml` so compatible
installers can resolve versions for the active Python runtime.

## Installation

Install the package:

```bash
pip install arclm
```

For local development:

```bash
pip install -e .[dev]
```

For the optional Flask API:

```bash
pip install -e .[web]
```

## Verifiable Dataflow

ArcLM is a library/API workflow, not a GUI workflow. The core dataflow is:

1. A local text file is provided as training data.
2. `prepare_data()` reads tokens, creates train/validation splits, and builds
   or reuses a tokenizer.
3. `build_model()` creates an `ArcLM` model from `Config`.
4. `build_trainer()` creates the optimizer, loss function, and `Trainer`.
5. `Trainer.train()` runs pretraining, fine-tuning, or continued training.
6. `Trainer.save()` writes a checkpoint with model, optimizer, config,
   history, vocab, token maps, and tokenizer metadata.
7. `load_model()` reloads the checkpoint.
8. `LoadedModel.predict()` generates text from a prompt.

The flow is verified by the test suite with tokenizer, model-shape,
training/save, batch checkpointing, external loading, and metrics logging tests.

## Quick Start

Train a small model from text:

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/data.txt",
    output="models/arclm.pth",
    tokenizer_type="word",
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
    checkpoint_batch_interval=100,
)

print(result.model_path)
```

Generate text from the saved checkpoint:

```python
from arclm import load_model

model = load_model("models/arclm.pth")
print(model.predict("machine learning", max_new_tokens=50, top_p=0.9))
```

## Fine-Tuning

Fine-tune from an ArcLM checkpoint:

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    checkpoint="models/arclm.pth",
    data="data/domain_text.txt",
    output="models/arclm_domain.pth",
    num_epochs=2,
    learning_rate=2e-5,
    freeze_backbone=True,
    checkpoint_batch_interval=50,
)
```

For native ArcLM checkpoints, tokenizer compatibility is checked before
fine-tuning or continued training.

## Lower-Level API

Use the lower-level pieces when the training loop needs direct control:

```python
import torch
from arclm import (
    Config,
    build_model,
    build_trainer,
    create_checkpoint_callback,
    prepare_data,
)

config = Config(
    data_path="data/data.txt",
    model_path="models/arclm.pth",
    tokenizer_type="word",
    max_vocab=8000,
    embed_dim=128,
    num_blocks=4,
    block_size=128,
    batch_size=32,
    num_epochs=5,
    learning_rate=3e-4,
    validation_split=0.1,
    checkpoint_batch_interval=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

data = prepare_data(config)
config.vocab_size = data.vocab_size

model = build_model(config, data.vocab_size)
trainer = build_trainer(model, config)

trainer.train(
    data.train_loader,
    config.num_epochs,
    val_loader=data.val_loader,
    early_stopping_patience=2,
    min_delta=1e-4,
    checkpoint_callback=create_checkpoint_callback(config, data.tokenizer, data.vocab_size),
)

trainer.save(
    config,
    vocab=data.tokenizer.vocab,
    stoi=data.tokenizer.stoi,
    itos=data.tokenizer.itos,
    tokenizer_metadata=data.tokenizer.to_checkpoint(),
)
```

## External Model Loading

External sources are normalized through `arclm.loaders`:

```python
from arclm import adapt_for_training, load_external_model

loaded = load_external_model("external/model.pth")
bundle = adapt_for_training(loaded)

print(loaded.source_type)
print(bundle.config)
```

Supported source types:

- ArcLM checkpoints saved by `Trainer.save()`.
- Raw PyTorch state dict files: `.pth`, `.pt`, `.bin`, `.ckpt`.
- `.safetensors` files when `safetensors` is installed.
- Hugging Face model folders or model IDs when `transformers` is installed.

## Logging

`AsyncTrainingLogger` provides non-blocking event logging for training metrics.
It writes JSONL records from a background thread and supports the essential
levels `INFO`, `WARNING`, `ERROR`, and `METRIC`.

The high-level `train_model()` API writes metrics to `metrics_log_path` when
provided. If no path is provided, it creates a run-specific metrics file under
the output directory.

## Optional Flask API

`app.py` exposes a small local prediction API for an existing checkpoint:

- `GET /health`
- `POST /predict`
- `POST /generate`

Run it after installing the web extra:

```bash
python app.py
```

The checkpoint path is read from `MODEL_PATH` and defaults to
`models/arclm.pth`.

## Project Layout

```text
arclm/      Library source code
docs/       Usage and versioning notes
examples/   Example scripts for data preparation, training, and loading
tests/      Pytest suite
app.py      Optional Flask prediction API
train.py    Local training script
```

## Tests

```bash
python -m pytest tests
```

Current verified baseline: `python -m pytest tests` passes.

## Documentation

See `docs/FULL_FEATURE_GUIDE.md` for the complete feature guide with
end-to-end examples for data processing, special tokens, training,
fine-tuning, SmartLoader, external loading, inference, diagnostics, config
files, and tracking.

See `docs/USAGE.md` for a shorter usage guide.

See `PROJECT_MAP.md` for the external project map and current architecture
state.

## License

ArcLM is licensed under the Apache License, Version 2.0. See `LICENSE`.
