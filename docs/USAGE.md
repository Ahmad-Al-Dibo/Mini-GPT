# ArcLM Usage Guide

This is the single detailed guide for using ArcLM.

## Train From Scratch

```python
import torch
from arclm import Config, build_model, build_trainer, prepare_data

config = Config(
    data_path="data/data.txt",
    model_path="models/arclm.pth",
    tokenizer_type="word",
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=32,
    batch_size=8,
    num_epochs=3,
    learning_rate=3e-4,
    validation_split=0.2,
    training_log_interval=10,
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
import torch
from arclm import Tokenizer, build_trainer, load_model, prepare_data


def tokenizer_from_loaded_model(loaded):
    tokenizer = loaded.generator.tokenizer
    if tokenizer is not None:
        return tokenizer

    tokenizer = Tokenizer(max_vocab=len(loaded.generator.stoi))
    tokenizer.stoi = loaded.generator.stoi
    tokenizer.itos = loaded.generator.itos
    tokenizer.vocab = [tokenizer.itos[i] for i in range(len(tokenizer.itos))]
    tokenizer.vocab_size = len(tokenizer.vocab)
    return tokenizer


loaded = load_model("models/arclm.pth")
tokenizer = tokenizer_from_loaded_model(loaded)

config = loaded.config
config.data_path = "data/data.txt"
config.model_path = "models/arclm_finetuned.pth"
config.num_epochs = 2
config.batch_size = 8
config.learning_rate = 2e-5
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

trainer.freeze_layers("blocks")
trainer.train(data.train_loader, config.num_epochs, val_loader=data.val_loader)

trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

## External Models

Downloaded models are local files. First inspect the checkpoint:

```python
import torch

checkpoint = torch.load("external/model.pth", map_location="cpu", weights_only=False)
print(checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint))
```

If the file contains ArcLM checkpoint fields like `model_state_dict`, `config`, `stoi`, `itos`, and `vocab_size`, load it with:

```python
from arclm import load_model

loaded = load_model("external/model.pth")
```

If it is not an ArcLM checkpoint, you need the original architecture and tokenizer. Unknown token IDs cannot be safely fine-tuned as text.

## Notes

- The current config field is `num_epochs`, not `epochs`.
- The current save call is `trainer.save(config, ...)`; it writes to `config.model_path`.
- Keep `data/` for local datasets and `models/` for local checkpoints.
- Keep large downloaded datasets and model weights out of git.
