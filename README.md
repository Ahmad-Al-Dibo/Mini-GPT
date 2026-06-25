# ArcLM

ArcLM is a compact PyTorch library for training, loading, generating with, and fine-tuning small causal language models.

It is designed for learning, experiments, and small local workflows rather than serving giant production LLMs.

## Install From Source

```bash
pip install -e .
```

For development and packaging:

```bash
pip install -e ".[dev]"
```

## Quick Training Run

```bash
python train.py
```

The default training script reads `data/data.txt` and saves a checkpoint to `models/arclm.pth`.

## Generate Text

```python
from arclm import load_model

model = load_model("models/arclm.pth")
print(model.predict("machine learning", max_new_tokens=50))
```

## Repository Layout

```text
arclm/              library source code
data/data.txt       tiny sample dataset for smoke training
examples/           minimal runnable usage example
tests/              focused smoke tests
docs/USAGE.md       training, loading, and fine-tuning guide
models/             local checkpoints, ignored except .gitkeep
```

## Build For PyPI

```bash
python -m build
twine check dist/*
```

Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

Then upload to PyPI:

```bash
twine upload dist/*
```

## Main APIs

- `Config` / `create_config`
- `prepare_data`
- `build_model`
- `build_trainer`
- `Trainer`
- `Generator`
- `load_model`
- `UnifiedPipeline`

See [docs/USAGE.md](docs/USAGE.md) for full usage examples, including fine-tuning a checkpoint and working with external model files.
