# ArcLM Full Feature Guide

This guide documents the current ArcLM library surface with practical examples.
It is written for local experimentation: small causal language models, local
datasets, checkpoint reuse, inspection, diagnostics, and simple deployment.

## What ArcLM Is

ArcLM is a compact PyTorch library for:

- training small causal language models from local text;
- fine-tuning or continuing compatible ArcLM checkpoints;
- preparing instruction/chat/pretraining datasets;
- preserving tokenizer metadata inside checkpoints;
- inspecting local/Hugging Face-style model sources before loading;
- adapting raw PyTorch, safetensors, ArcLM, and Hugging Face sources;
- generating text from saved checkpoints;
- logging, diagnostics, and experiment tracking.

It is not a distributed trainer and it does not try to hide model compatibility
rules. Tokenizer and architecture compatibility still matter.

## Installation

For local development from this repository:

```bash
pip install -e .[dev]
```

For the optional Flask API:

```bash
pip install -e .[web]
```

Run the test suite:

```bash
python -m pytest tests
```

## Complete Runnable Example

The broadest runnable example is:

```bash
python examples/full_feature_guide.py
```

It creates tiny files under `runs/full_feature_guide/`, then demonstrates:

- JSONL instruction-data preprocessing with `DataProcessor`;
- high-level pretraining with `train_model`;
- fine-tuning from the saved checkpoint;
- inference with `load_model`;
- top-k diagnostics;
- lower-level `Config`/`prepare_data`/`build_model`/`build_trainer`;
- `SmartLoader.inspect`;
- `load_external_model`;
- config save/load;
- local experiment tracking.

## Special Tokens

Use one shared token list for pretraining, fine-tuning, and inference prompts:

```python
SPECIAL_TOKENS = [
    "<|qa_start|>",
    "<|res_start|>",
    "<|end|>",
    "<|pad|>",
]
```

Pass it during pretraining:

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/data.txt",
    output="models/arclm_pretrained.pth",
    tokenizer_type="word",
    user_defined_symbols=SPECIAL_TOKENS,
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
)
```

For `word` tokenizers, ArcLM reserves `user_defined_symbols` as real vocabulary
items even if they are rare. For `sentencepiece`, ArcLM passes them to
SentencePiece training and stores them in tokenizer metadata.

Use `encode_text()` for raw strings:

```python
ids = result.tokenizer.encode_text("<|qa_start|> wat is minigpt? <|res_start|>")
```

Use `encode()` only when you already have a token list:

```python
ids = result.tokenizer.encode(["<|qa_start|>", "wat", "is", "minigpt?"])
```

Important: for fine-tuning, start from a checkpoint that already contains the
special tokens you want. Adding new tokens later changes the vocabulary and
usually requires resizing/retraining embeddings and the output head.

## Data Processing

`DataProcessor` loads JSON, JSONL, CSV, TXT, or custom sources into an in-memory
`ProcessedDataset`.

```python
from arclm import DataProcessor

dataset = DataProcessor.load("data/qa.jsonl")

processed = (
    dataset.clean()
    .filter(lambda row: bool(row.get("question")))
    .transform(
        format="instruction",
        mapping={
            "instruction": "question",
            "input": "context",
            "output": "answer",
        },
        template="<|qa_start|> {instruction}\n{input}\n<|res_start|> {output} <|end|>",
    )
)

splits = processed.split(train=0.8, validation=0.1, test=0.1, seed=42)
```

Write transformed samples to a text file for `train_model`:

```python
from pathlib import Path

Path("data/finetune.txt").write_text(
    "\n".join(sample["text"] for sample in processed.samples),
    encoding="utf-8",
)
```

Optional tokenization:

```python
tokenized = processed.tokenize(result.tokenizer)
print(tokenized.samples[0]["tokens"])
```

Custom loader:

```python
from arclm import DataProcessor

def load_custom(path):
    yield {"text": path.read_text(encoding="utf-8")}

dataset = DataProcessor.load("custom.data", loader=load_custom)
```

## High-Level Training

`train_model()` owns the whole flow: data loading, tokenizer creation/reuse,
model creation/adaptation, trainer creation, training, metrics, and checkpoint
saving.

Pretraining:

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/data.txt",
    output="models/arclm_pretrained.pth",
    tokenizer_type="sentencepiece",
    user_defined_symbols=SPECIAL_TOKENS,
    max_vocab=8000,
    embed_dim=128,
    num_blocks=4,
    block_size=128,
    batch_size=16,
    num_epochs=5,
    learning_rate=3e-4,
    validation_split=0.1,
    checkpoint_batch_interval=100,
)
```

Fine-tuning:

```python
result = train_model(
    mode="finetune",
    checkpoint="models/arclm_pretrained.pth",
    data="data/finetune.txt",
    output="models/arclm_finetuned.pth",
    user_defined_symbols=SPECIAL_TOKENS,
    num_epochs=2,
    learning_rate=2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True,
)
```

Continue training:

```python
result = train_model(
    mode="continue_training",
    checkpoint="models/arclm_pretrained.pth",
    data="data/data.txt",
    output="models/arclm_continued.pth",
    num_epochs=5,
)
```

For native ArcLM checkpoints, tokenizer compatibility is checked before
fine-tuning/continuing so token IDs are not silently corrupted.

## Lower-Level Training

Use lower-level pieces when you need explicit control over the model, trainer,
callbacks, or diagnostics.

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
    model_path="models/arclm_low_level.pth",
    tokenizer_type="word",
    user_defined_symbols=SPECIAL_TOKENS,
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
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
    checkpoint_callback=create_checkpoint_callback(config, data.tokenizer, data.vocab_size),
    checkpoint_batch_interval=config.checkpoint_batch_interval,
)

trainer.save(
    config,
    vocab=data.tokenizer.vocab,
    stoi=data.tokenizer.stoi,
    itos=data.tokenizer.itos,
    tokenizer_metadata=data.tokenizer.to_checkpoint(),
)
```

## Inference

Load a saved ArcLM checkpoint:

```python
from arclm import load_model

loaded = load_model("models/arclm_finetuned.pth")

prompt = "<|qa_start|> Wat is MiniGPT? <|res_start|>"
print(loaded.predict(prompt, max_new_tokens=50, top_p=0.9))
```

Cached prediction helper:

```python
from arclm import predict

text = predict(
    "<|qa_start|> Wat is MiniGPT? <|res_start|>",
    model_path="models/arclm_finetuned.pth",
    max_new_tokens=50,
)
```

Generation options:

- `max_new_tokens`: maximum generated tokens;
- `temperature`: lower is more conservative;
- `repetition_penalty`: penalizes repeated tokens;
- `top_k`: keep only the top K logits;
- `top_p`: nucleus sampling threshold.

## SmartLoader

`SmartLoader` inspects before loading. It can detect local model folders,
checkpoint files, Hugging Face-style folders, tokenizer files, weight formats,
adapter files, optimizer/scheduler state, and trainer state.

```python
from arclm import SmartLoader

plan = SmartLoader.inspect(
    "models/arclm_pretrained.pth",
    auto_detect=True,
    precision="fp32",
    load_optimizer=False,
)

print(plan.report)
print(plan.to_dict())
```

Manual mode:

```python
plan = SmartLoader.inspect(
    "Qwen/Qwen2.5-7B-Instruct",
    auto_detect=False,
    model_type="qwen",
    tokenizer="auto",
    weight_format="safetensors",
    precision="bf16",
    load_as="full_model",
)
```

Load with inspection metadata attached:

```python
loaded = SmartLoader.load(
    "models/arclm_pretrained.pth",
    load_optimizer=False,
)

print(loaded.metadata["load_report"])
```

Register a custom inspector:

```python
from arclm.loaders import ModelInspector, register_model_inspector

class MyInspector(ModelInspector):
    def can_inspect(self, source):
        return str(source).endswith(".myformat")

    def inspect(self, source):
        plan = super().inspect(source)
        plan.source_type = "myformat"
        plan.weight_format = "myformat"
        return plan

register_model_inspector(MyInspector())
```

## External Model Loading

The loader registry normalizes supported sources into `LoadedCheckpoint`.

```python
from arclm import adapt_for_training, load_external_model

loaded = load_external_model("external/model.pth")
bundle = adapt_for_training(loaded)

print(loaded.source_type)
print(bundle.config)
```

Supported sources:

- ArcLM checkpoints saved by `Trainer.save`;
- raw PyTorch state dicts: `.pth`, `.pt`, `.bin`, `.ckpt`;
- safetensors files when `safetensors` is installed;
- Hugging Face model folders or model IDs when `transformers` is installed.

Hugging Face loading can download and instantiate large models. Use
`SmartLoader.inspect()` first when you only want to understand a source.

## Diagnostics

Top-k next-token predictions:

```python
from arclm import predict_top_k, format_top_k_predictions

predictions = predict_top_k(
    loaded.model,
    loaded.generator.stoi,
    loaded.generator.itos,
    loaded.config.block_size,
    loaded.device,
    "machine learning",
    k=5,
    tokenizer=loaded.generator.tokenizer,
)

print(format_top_k_predictions("machine learning", predictions))
```

Tokenizer coverage:

```python
coverage = data.tokenizer.analyze_coverage(data.tokens)
from arclm import format_tokenizer_coverage_report

print(format_tokenizer_coverage_report(coverage))
```

Training diagnostics:

```python
from arclm import build_training_diagnostics_report

print(build_training_diagnostics_report(model, data, config))
```

Evaluation metrics:

```python
from arclm import calculate_metrics, export_metrics_to_json

metrics = calculate_metrics(model, data.val_loader, config, device=config.device)
export_metrics_to_json(metrics, "reports/metrics.json")
```

## Config Files

```python
from arclm import Config
from arclm.config_loader import save_config_json, load_config_json

config = Config(embed_dim=64, block_size=32)
save_config_json(config, "configs/small.json")
loaded_config = load_config_json("configs/small.json")
```

YAML is supported when PyYAML is installed:

```python
from arclm.config_loader import save_config_yaml, load_config_yaml
```

## Experiment Tracking

Local tracking requires no external service:

```python
from arclm.tracking import create_experiment

tracker = create_experiment(
    "baseline",
    experiment_dir="experiments",
    tags={"tokenizer": "word"},
)
tracker.log_config(config)
tracker.log_metrics({"loss": 2.5}, step=1)
tracker.log_artifact("models/arclm_pretrained.pth")
tracker.end()
```

Optional backends exist for MLflow and Weights & Biases if installed.

## Flask API

`app.py` exposes:

- `GET /health`
- `POST /predict`
- `POST /generate`

Run:

```bash
set MODEL_PATH=models/arclm_finetuned.pth
python app.py
```

## Example Scripts

- `examples/full_feature_guide.py`: broad end-to-end local demo.
- `examples/prepare_data.py`: convert raw TXT/JSONL into training files.
- `examples/train_pretrain.py`: CLI pretraining example.
- `examples/train_finetune.py`: CLI fine-tuning example.
- `examples/load_external_and_finetune.py`: external source fine-tuning.
- `examples/test/test-train.py`: compact pretraining smoke example.
- `examples/test/test-finetuing.py`: compact fine-tuning smoke example.
- `examples/test/test-usage.py`: compact inference smoke example.

## Common Problems

`Special tokens become <UNK>`

Pass the same `user_defined_symbols` during pretraining, and use `encode_text`
for raw strings. Fine-tuning cannot safely add new tokens to an old checkpoint
without changing model shapes.

`Fine-tuning says tokenizer is incompatible`

Use the checkpoint tokenizer. Do not rebuild a tokenizer with a different
vocabulary unless you intentionally adapt or retrain the model head.

`SentencePiece output looks different from word-tokenizer output`

SentencePiece uses subword pieces and its own spacing markers internally.
Use `decode_string()` for text and keep tokenizer metadata in checkpoints.

`Hugging Face loading is slow or uses too much memory`

`load_external_model()` instantiates the model. Use `SmartLoader.inspect()`
first when you only need to inspect files or metadata.

`No checkpoint is written during training`

The high-level `train_model()` always saves at the end. Automatic intermediate
saves require `checkpoint_interval` or `checkpoint_batch_interval`.

`Validation metrics are missing`

Set `validation_split > 0.0` and ensure the dataset is large enough to create a
validation loader.
