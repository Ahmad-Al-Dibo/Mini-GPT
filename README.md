# ArcLM

ArcLM is a lightweight PyTorch library for training, fine-tuning, and generating text with compact causal language models.

It is designed for learning, experimentation, research, and local AI workflows.

## Features

* Train compact causal language models from your own text
* Generate text with a simple API
* Fine-tune existing ArcLM checkpoints
* Lightweight and easy to understand
* Built on PyTorch

## Installation

Install ArcLM from PyPI:

```bash
pip install arclm
```

Or install the latest development version:

```bash
git clone https://github.com/Ahmad-Al-Dibo/ArcLM.git
cd ArcLM
pip install -e ".[dev]"
```

## Quick Start

Train a model:

```bash
python train.py
```

Generate text:

```python
from arclm import load_model

model = load_model("models/arclm.pth")
print(model.predict("machine learning", max_new_tokens=50))
```

## Documentation

The complete documentation is available in **docs/USAGE.md**.

It includes:

* Training a model from scratch
* Loading saved checkpoints
* Fine-tuning existing ArcLM models
* Working with external checkpoints
* Configuration options
* Best practices and notes

## Project Structure

```text
arclm/              Library source code
data/               Example datasets
docs/               Documentation
examples/           Example scripts
models/             Local checkpoints
tests/              Tests
```

## License

ArcLM is licensed under the Apache License 2.0. See the LICENSE file for details.