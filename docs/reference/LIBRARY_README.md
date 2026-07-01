# arclm Library Usage Guide

This guide shows the real library workflow for:

- training a new arclm model from scratch
- loading a trained/pretrained arclm checkpoint for inference
- fine-tuning a arclm checkpoint on new data
- fine-tuning a model downloaded from another source, such as Kaggle

The examples are intentionally small so you can run them locally first, then replace the demo data with your real dataset.

## Install And Import

From the repository root:

```bash
pip install -e . # this installs the library in editable mode
pip install -r requirements.txt
```

After installation, prefer top-level imports:

```python
from arclm import Config, prepare_data, build_model, build_trainer, load_model
```

If you run scripts directly inside this repository without installing the package, the older local import style also works:

```python
from src.arclm import Config, prepare_data, build_model, build_trainer, load_model
```

## Important Concepts

arclm checkpoints are PyTorch `.pth` files. A fully reusable checkpoint should contain:

- model weights
- optimizer state
- config
- `vocab`, `stoi`, and `itos`
- tokenizer metadata when using SentencePiece
- `block_size` and `vocab_size`

For fine-tuning, keep the tokenizer and architecture compatible with the checkpoint. A checkpoint trained with one vocabulary should normally be fine-tuned with that same vocabulary, not a new tokenizer.

## Example Data

For quick tests, create small text files like these.

General pretraining data:

```text
Machine learning is a field of artificial intelligence.
Language models learn patterns from text.
Transformers use attention to connect tokens in context.
arclm is a small educational GPT model.
Training from scratch starts with random weights.
```

Fine-tuning data:

```text
<qa> question: what is arclm? answer: arclm is a small GPT-style language model library. </qa>
<qa> question: what is fine-tuning? answer: Fine-tuning adapts a pretrained model to new data. </qa>
<qa> question: when should I freeze layers? answer: Freeze layers when the new dataset is small. </qa>
```

Use plain `.txt` files for normal language-model training. The model learns next-token prediction, so instruction or Q/A examples should be formatted exactly how you want the model to see them.

## Train A New Model From Scratch

This script creates demo data, trains a new model with random weights, saves the checkpoint, and generates a small sample.

Save as `examples/train_from_scratch_real.py` or run it in a notebook cell.

```python
from pathlib import Path
import torch

from arclm import Config, Generator, build_model, build_trainer, prepare_data


project_dir = Path("runs/scratch_demo")
project_dir.mkdir(parents=True, exist_ok=True)

data_path = project_dir / "train.txt"
data_path.write_text(
    """
    Machine learning is a field of artificial intelligence.
    Language models learn patterns from text.
    Transformers use attention to connect tokens in context.
    arclm is a small educational GPT model.
    Training from scratch starts with random weights.
    Fine-tuning adapts a pretrained model to a new dataset.
    """ * 50,
    encoding="utf-8",
)

config = Config(
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=5,
    learning_rate=3e-4,
    weight_decay=0.01,
    dropout=0.1,
    data_path=str(data_path),
    model_path=str(project_dir / "scratch_model.pth"),
    tokenizer_type="word",
    max_vocab=2000,
    validation_split=0.2,
    early_stopping_patience=2,
    early_stopping_min_delta=1e-4,
    restore_best_model=True,
    training_log_interval=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Load text, split train/validation, build tokenizer, and create DataLoaders.
data = prepare_data(config)
config.vocab_size = data.vocab_size

# Build a fresh randomly initialized arclm model.
model = build_model(config, vocab_size=data.vocab_size)
trainer = build_trainer(model, config)

trainer.train(
    data.train_loader,
    config.num_epochs,
    val_loader=data.val_loader,
    early_stopping_patience=config.early_stopping_patience,
    min_delta=config.early_stopping_min_delta,
)

tokenizer_metadata = (
    data.tokenizer.to_checkpoint()
    if hasattr(data.tokenizer, "to_checkpoint")
    else None
)

trainer.save(
    config,
    vocab=data.tokenizer.vocab,
    stoi=data.tokenizer.stoi,
    itos=data.tokenizer.itos,
    tokenizer_metadata=tokenizer_metadata,
)

generator = Generator(
    model=model,
    stoi=data.tokenizer.stoi,
    itos=data.tokenizer.itos,
    block_size=config.block_size,
    device=torch.device(config.device),
    tokenizer=data.tokenizer,
)

print(generator.generate_string("machine learning", max_new_tokens=30, top_p=0.9))
print(f"Saved checkpoint: {config.model_path}")
```

Notes:

- Use `num_epochs`, not `epochs`, when configuring the current `Config` class.
- `trainer.save(config, ...)` saves to `config.model_path`.
- Increase `max_vocab`, `block_size`, `embed_dim`, and `num_blocks` for serious training.
- Use `tokenizer_type="sentencepiece"` for better subword handling on larger datasets.

## Load A Pretrained arclm Checkpoint

Use `load_model()` when the checkpoint was saved in arclm format with tokenizer data.

```python
from arclm import load_model

loaded = load_model("runs/scratch_demo/scratch_model.pth")

text = loaded.predict(
    "machine learning",
    max_new_tokens=50,
    temperature=0.8,
    repetition_penalty=1.2,
    top_p=0.9,
)

print(text)
```

`load_model()` returns a `LoadedModel` bundle:

- `loaded.model`: the PyTorch model
- `loaded.generator`: generation helper
- `loaded.config`: checkpoint config
- `loaded.predict(...)`: quick generation method

## Fine-Tune A Pretrained arclm Checkpoint

Fine-tuning should reuse the checkpoint tokenizer. The helper below rebuilds a tokenizer object from the loaded checkpoint when needed.

```python
from pathlib import Path
import torch

from arclm import Tokenizer, build_trainer, load_model, prepare_data


def tokenizer_from_loaded_model(loaded):
    tokenizer = loaded.generator.tokenizer
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return tokenizer

    # Fallback for word-token checkpoints loaded from stoi/itos.
    tokenizer = Tokenizer(max_vocab=len(loaded.generator.stoi))
    tokenizer.stoi = loaded.generator.stoi
    tokenizer.itos = loaded.generator.itos
    tokenizer.vocab = [
        tokenizer.itos[i]
        for i in range(len(tokenizer.itos))
        if i in tokenizer.itos
    ]
    tokenizer.vocab_size = len(tokenizer.vocab)
    return tokenizer


project_dir = Path("runs/finetune_demo")
project_dir.mkdir(parents=True, exist_ok=True)

fine_tune_data_path = project_dir / "qa_data.txt"
fine_tune_data_path.write_text(
    """
    <qa> question: what is arclm? answer: arclm is a small GPT-style language model library. </qa>
    <qa> question: what is fine-tuning? answer: Fine-tuning adapts a pretrained model to new data. </qa>
    <qa> question: when should I freeze layers? answer: Freeze layers when the new dataset is small. </qa>
    """ * 80,
    encoding="utf-8",
)

loaded = load_model("runs/scratch_demo/scratch_model.pth")
tokenizer = tokenizer_from_loaded_model(loaded)

config = loaded.config
config.data_path = str(fine_tune_data_path)
config.model_path = str(project_dir / "finetuned_model.pth")
config.num_epochs = 3
config.batch_size = 8
config.learning_rate = 2e-5
config.weight_decay = 0.01
config.validation_split = 0.2
config.vocab_size = len(loaded.generator.stoi)
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.use_discriminative_lr = True
config.lr_multiplier = {
    "embeddings": 0.1,
    "blocks": 0.1,
    "head": 1.0,
}

data = prepare_data(config, existing_tokenizer=tokenizer)

model = loaded.model.to(torch.device(config.device))
trainer = build_trainer(model, config)

# Common strategy for small fine-tuning datasets.
trainer.freeze_layers("blocks")

trainer.train(
    data.train_loader,
    config.num_epochs,
    val_loader=data.val_loader,
    early_stopping_patience=2,
    min_delta=1e-4,
)

tokenizer_metadata = (
    tokenizer.to_checkpoint()
    if hasattr(tokenizer, "to_checkpoint")
    else None
)

trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer_metadata,
)

print(f"Saved fine-tuned checkpoint: {config.model_path}")
```

After fine-tuning:

```python
from arclm import load_model

model = load_model("runs/finetune_demo/finetuned_model.pth")
print(model.predict("<qa> question: what is fine-tuning? answer:", max_new_tokens=40))
```

## Fine-Tune A Model Downloaded From Kaggle Or Another Source

Kaggle is usually just a source for files. Download the model artifact first, then treat it as a local checkpoint.

Recommended folder layout:

```text
external_models/
  kaggle_author_model/
    model.pth
    README.md
```

### Case 1: The Kaggle File Is A arclm Checkpoint

If the downloaded `.pth` file contains arclm keys such as `model_state_dict`, `config`, `stoi`, `itos`, and `vocab_size`, use the same fine-tuning flow as above:

```python
loaded = load_model("external_models/kaggle_author_model/model.pth")
```

Then reuse `loaded.model`, `loaded.config`, and the checkpoint tokenizer for fine-tuning.

### Case 2: The Kaggle File Is A Raw PyTorch Checkpoint

Some files only contain weights. Inspect the checkpoint first:

```python
import torch

try:
    checkpoint = torch.load(
        "external_models/kaggle_author_model/model.pth",
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(
        "external_models/kaggle_author_model/model.pth",
        map_location="cpu",
    )

print(checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint))
```

If it has compatible arclm weights under `model_state_dict`, `model`, or `state_dict`, you can load it manually:

```python
from arclm import Config, build_model

config = Config(
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    vocab_size=2000,
    device="cpu",
)

model = build_model(config, vocab_size=config.vocab_size)
state_dict = checkpoint
if isinstance(checkpoint, dict):
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model")
        or checkpoint.get("state_dict")
    )
if state_dict is None:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
```

After that, create a compatible tokenizer and use `build_trainer()` to fine-tune. If the checkpoint does not include tokenizer information, you must know which tokenizer was used originally. A model trained with unknown token IDs cannot be reliably fine-tuned or used for text generation.

### Case 3: The External Model Is Not arclm

For Hugging Face causal language models, `UnifiedPipeline` can attempt to adapt weights into a arclm model:

```python
from pathlib import Path
import torch

from arclm import Config, StoppingCriteria, UnifiedPipeline, prepare_data


project_dir = Path("runs/external_hf_demo")
project_dir.mkdir(parents=True, exist_ok=True)

data_path = project_dir / "domain_data.txt"
data_path.write_text(
    """
    <qa> question: what is deployment? answer: Deployment is releasing software to users. </qa>
    <qa> question: what is monitoring? answer: Monitoring tracks system behavior in production. </qa>
    """ * 100,
    encoding="utf-8",
)

config = Config(
    embed_dim=128,
    num_blocks=4,
    block_size=64,
    batch_size=8,
    num_epochs=2,
    learning_rate=2e-5,
    tokenizer_type="sentencepiece",
    max_vocab=8000,
    data_path=str(data_path),
    model_path=str(project_dir / "adapted_finetuned.pth"),
    validation_split=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_discriminative_lr=True,
)

data = prepare_data(config)
config.vocab_size = data.vocab_size

pipeline = UnifiedPipeline(
    config=config,
    mode="fine_tuning",
    pretrained_source="gpt2",
    stopping_criteria=StoppingCriteria(
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
    ),
)

pipeline.build(vocab_size=data.vocab_size)
results = pipeline.train(
    data.train_loader,
    val_loader=data.val_loader,
    num_epochs=config.num_epochs,
)

pipeline.save_checkpoint(config.model_path)
print(results)
```

This is an adaptation path, not a guarantee that every external architecture maps cleanly. For production work, inspect the loading logs and evaluate the output carefully.

## Choosing A Fine-Tuning Strategy

Use these defaults as a starting point:

```python
config.learning_rate = 2e-5
config.num_epochs = 3
config.batch_size = 8
config.use_discriminative_lr = True
config.lr_multiplier = {
    "embeddings": 0.1,
    "blocks": 0.1,
    "head": 1.0,
}
```

Layer freezing:

```python
# Very small dataset: train mostly the output head.
trainer.freeze_layers("embedding")
trainer.freeze_layers("blocks")

# Small/medium dataset: keep embeddings trainable but freeze transformer blocks.
trainer.freeze_layers("blocks")

# Larger dataset: train all layers with a low learning rate.
trainer.unfreeze_layers(None)
```

Check trainable parameters:

```python
info = trainer.get_frozen_layers_info()
print(info)
```

## Tokenizer Guidance

Use `tokenizer_type="word"` for tiny educational runs:

```python
config = Config(tokenizer_type="word", max_vocab=5000)
```

Use `tokenizer_type="sentencepiece"` for real datasets:

```python
config = Config(
    tokenizer_type="sentencepiece",
    sentencepiece_model_type="bpe",
    max_vocab=16000,
)
```

When continuing or fine-tuning a checkpoint, reuse the checkpoint tokenizer. Changing tokenizers changes token IDs and usually makes the old weights meaningless.

## Saving And Loading Checklist

When saving after direct training or fine-tuning:

```python
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

When loading for inference:

```python
loaded = load_model("path/to/model.pth")
print(loaded.predict("your prompt", max_new_tokens=80))
```

## Common Problems

`AttributeError: Config has no attribute epochs`

Use `config.num_epochs`. The current `Config` class stores the training length as `num_epochs`.

`ValueError: checkpoint is missing stoi or itos`

The checkpoint does not contain tokenizer mappings. You need the original tokenizer files or a converted arclm checkpoint.

`size mismatch` while loading weights

The checkpoint architecture does not match the config. Check `embed_dim`, `num_blocks`, `block_size`, and `vocab_size`.

Generated text is poor after fine-tuning

Use more data, lower the learning rate, train for a few more epochs, and make sure the prompt format matches the fine-tuning examples.

Out of memory

Lower `batch_size`, `block_size`, `embed_dim`, or `num_blocks`.

## Minimal Workflow Summary

Train from scratch:

```python
data = prepare_data(config)
config.vocab_size = data.vocab_size
model = build_model(config, data.vocab_size)
trainer = build_trainer(model, config)
trainer.train(data.train_loader, config.num_epochs, val_loader=data.val_loader)
trainer.save(config, vocab=data.tokenizer.vocab, stoi=data.tokenizer.stoi, itos=data.tokenizer.itos)
```

Load and predict:

```python
loaded = load_model("model.pth")
print(loaded.predict("hello", max_new_tokens=50))
```

Fine-tune:

```python
loaded = load_model("base_model.pth")
model = loaded.model
config = loaded.config
config.data_path = "new_data.txt"
config.model_path = "fine_tuned_model.pth"
config.learning_rate = 2e-5
config.num_epochs = 3
```

Then prepare data with the checkpoint tokenizer, train with `build_trainer()`, and save a new checkpoint.