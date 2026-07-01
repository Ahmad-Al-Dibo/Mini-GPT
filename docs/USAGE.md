# ArcLM Usage Guide

This is a short usage guide for using ArcLM. For the complete feature guide,
including SmartLoader, DataProcessor, diagnostics, config files, tracking, and
full runnable examples, see `FULL_FEATURE_GUIDE.md`.

## Train From Scratch

The high-level API owns the full flow: data, tokenizer, model, trainer, metrics
logging, and checkpoint writing.

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/pretrain.txt",
    output="models/arclm_pretrained.pth",
    tokenizer_type="word",
    user_defined_symbols=["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
    learning_rate=3e-4,
    validation_split=0.2,
    checkpoint_batch_interval=100,
)

print(result.model_path)
```

`train_model()` writes a checkpoint and JSONL metrics under `models/runs/...`
unless `metrics_log_path` is provided. Set `checkpoint_batch_interval=N` to
save the latest checkpoint every N training batches, or `checkpoint_interval=N`
to save every N completed epochs.

The lower-level workflow remains available when you need full control:

```python
import torch
from arclm import Config, build_model, build_trainer, create_checkpoint_callback, prepare_data

config = Config(
    data_path="data/data.txt",
    model_path="models/arclm.pth",
    tokenizer_type="word",
    user_defined_symbols=["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
    learning_rate=3e-4,
    validation_split=0.2,
    training_log_interval=10,
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

## Load A Checkpoint

```python
from arclm import load_model

loaded = load_model("models/arclm.pth")
print(loaded.predict("machine learning", max_new_tokens=50, top_p=0.9))
```

`load_model()` returns a bundle with:

- `loaded.model`
- `loaded.generator`
- `loaded.config`
- `loaded.predict(...)`

## Fine-Tune An ArcLM Checkpoint

Fine-tuning should reuse the checkpoint tokenizer. Do not build a new tokenizer unless you are intentionally training a new model head or adapting weights.

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    checkpoint="models/arclm_pretrained.pth",
    data="data/finetune.txt",
    output="models/arclm_finetuned.pth",
    user_defined_symbols=["<|qa_start|>", "<|res_start|>", "<|end|>", "<|pad|>"],
    num_epochs=2,
    learning_rate=2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True,
)
```

## External Models

External sources flow through the loader registry:

```python
from arclm import adapt_for_training, load_external_model

loaded = load_external_model("external/model.pth")
bundle = adapt_for_training(loaded)

print(loaded.source_type)
print(bundle.config)
```

Supported sources:

- ArcLM checkpoints saved by `Trainer.save`
- raw PyTorch state dict files: `.pth`, `.pt`, `.bin`, `.ckpt`
- `.safetensors` files when `safetensors` is installed
- Hugging Face model folders or model IDs when `transformers` is installed

Fine-tune an external source without writing checkpoint surgery:

```python
from arclm import train_model

train_model(
    mode="finetune",
    checkpoint="external/model.pth",
    data="data/finetune.txt",
    output="models/arclm_external_finetuned.pth",
    num_epochs=2,
)
```

For native ArcLM checkpoints, ArcLM checks the tokenizer mapping before
fine-tuning or continuing training. If the mapping differs, training stops
instead of silently corrupting token IDs.

## Data Preparation Examples

Use `DataProcessor` directly when you need explicit mappings/templates:

```python
from arclm import DataProcessor

dataset = DataProcessor.load("data/qa.jsonl")
processed = dataset.transform(
    format="instruction",
    mapping={
        "instruction": "question",
        "input": "context",
        "output": "answer",
    },
    template="<|qa_start|> {instruction}\n{input}\n<|res_start|> {output} <|end|>",
)
```

```bash
python examples/prepare_data.py raw.txt qa.jsonl --output-dir data
python examples/full_feature_guide.py
python examples/train_pretrain.py --data data/pretrain.txt
python examples/train_finetune.py --checkpoint models/arclm_pretrained.pth
python examples/load_external_and_finetune.py external/model.pth
```

`examples/prepare_data.py` accepts `.txt` and `.jsonl`. JSONL records with
`question`/`answer`, `prompt`/`response`, or `instruction`/`completion` are
formatted for fine-tuning.

## Notes

- The current config field is `num_epochs`, not `epochs`.
- The current save call is `trainer.save(config, ...)`; it writes to `config.model_path`.
- Use `encode_text()` for raw prompt strings; use `encode()` only for a list of tokens.
- Keep the same `user_defined_symbols` across pretraining, fine-tuning, and inference prompts.
- Keep `data/` for local datasets and `models/` for local checkpoints.
- Keep large downloaded datasets and model weights out of git.
