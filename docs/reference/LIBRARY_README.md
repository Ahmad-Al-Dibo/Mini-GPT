# GPT Library - Modulaire LLM Framework

Een herbruikbare Python library voor het trainen en gebruiken van kleine Transformer-gebaseerde taalmodellen.

## 📁 Project Structuur

```
gpt_lib/
├── __init__.py           # Library exports
├── config.py             # Configuratie management
├── tokenizer.py          # Tekst naar tokens conversie
├── dataset.py            # Dataset en DataLoader
├── model.py              # MiniGPT architectuur
├── trainer.py            # Training loop en model management
└── generator.py          # Text generation utilities

train.py                  # Standaard training script
examples/
├── example1_generation.py       # Basis tekstgeneratie
├── example2_custom_training.py  # Custom modelgroottes
└── example3_instruction_training.py  # Instruction-following
```

## 🚀 Snel Starten

### Installatie

```bash
# Vereisten
pip install torch numpy

# Library is klaar om te gebruiken (lokaal)
```

### Training

```bash
python train.py
```

Dit traint een model met default config:
- 64 embedding dim
- 2 transformer blokken
- 100 epochs
- Saved to `output/mini_gpt.pth`

### Tekstgeneratie

```bash
python examples/example1_generation.py
```

## 🎯 Advanced Features (NEW!)

### Repetition Penalty

```python
# Voorkomen van woord herhaling
text = generator.generate_string(
    "hello",
    repetition_penalty=1.5  # 1.0=no penalty, 2.0=strong penalty
)
```

### Sampling Strategieën

```python
# Top-K sampling
text = generator.generate_string(
    prompt,
    top_k=50  # Keep top 50 tokens
)

# Nucleus (Top-P) sampling
text = generator.generate_string(
    prompt,
    top_p=0.9  # Cumulative probability threshold
)

# Combined
text = generator.generate_string(
    prompt,
    temperature=1.2,
    repetition_penalty=1.5,
    top_p=0.9
)
```

### Generalisatie Monitoring

```python
from gpt_lib import GeneralizationMonitor

monitor = GeneralizationMonitor()
for epoch in range(epochs):
    train_loss = ...
    val_loss = ...
    monitor.update(train_loss, val_loss)

print(monitor.get_report())  # Overfitting detection
```

### Early Stopping

```python
from gpt_lib import EarlyStopping

early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

for epoch in range(epochs):
    val_loss = validate()
    if early_stopper.check(val_loss):
        print("Stop training!")
        break
```

### Regularisatie Techniques

```python
from gpt_lib import L2Regularization, LearningRateScheduler, LabelSmoothing

# L2 Regularization
l2_reg = L2Regularization(lambda_l2=1e-4)
reg_loss = l2_reg.compute_loss(model)

# Learning Rate Scheduling
scheduler = LearningRateScheduler(optimizer, "cosine", total_epochs=100)
scheduler.step(epoch)

# Label Smoothing
criterion = LabelSmoothing(num_classes=50000, smoothing=0.1)
```

---

## 📚 Library API

### 1. **Config** - Configuratie Management

```python
from gpt_lib import Config

config = Config(
    embed_dim=64,
    block_size=8,
    batch_size=64,
    epochs=100,
    learning_rate=1e-3,
    num_blocks=2,
    model_path="output/model.pth",
    data_path="data/data.txt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Print config
print(config)

# Access values
print(config.embed_dim)
```

### 2. **Tokenizer** - Tekst Verwerking

```python
from gpt_lib import Tokenizer

# Create tokenizer
tokenizer = Tokenizer(max_vocab=50000)

# Build vocabulary from text
text = "your training text here..."
tokenizer.build(text)

# Encode tekst naar indices
tokens = text.split()
encoded = tokenizer.encode(tokens)  # List of integers

# Decode terug naar tekst
decoded = tokenizer.decode(encoded)  # List of words
decoded_str = tokenizer.decode_string(encoded)  # String

# Get info
vocab_size = tokenizer.get_vocab_size()

# Save/Load vocab
tokenizer.save_vocab("vocab.txt")
tokenizer.load_vocab("vocab.txt")
```

### 3. **Dataset** - Data Handling

```python
from gpt_lib import create_dataloader

# Create DataLoader
loader = create_dataloader(
    encoded_data=encoded,
    block_size=8,
    batch_size=64,
    shuffle=True
)

# Use in training
for x, y in loader:
    # x: [batch, block_size]
    # y: [batch, block_size]
    pass
```

### 4. **MiniGPT** - Model

```python
from gpt_lib import MiniGPT
import torch

# Create model
model = MiniGPT(
    vocab_size=50001,
    embed_dim=64,
    block_size=8,
    num_blocks=2
)

# Move to device
model = model.to("cpu")

# Forward pass
input_ids = torch.randint(0, 50001, (16, 8))  # [batch, seq_len]
logits = model(input_ids)  # [batch, seq_len, vocab_size]

# Get parameter count
params = model.get_num_parameters()
print(f"Total parameters: {params:,}")
```

### 5. **Trainer** - Training Management

```python
from gpt_lib import Trainer
import torch.nn as nn

# Create trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion, config)

# Train
trainer.train(loader, epochs=100)

# Save checkpoint
trainer.save(
    "output/model.pth",
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    block_size=8,
    vocab_size=50001
)

# Load checkpoint
if trainer.exists("output/model.pth"):
    checkpoint = trainer.load("output/model.pth")
```

### 6. **Generator** - Text Generation

```python
from gpt_lib import Generator

# Create generator
generator = Generator(
    model=model,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    block_size=8,
    device="cpu"
)

# Generate tokens
tokens = generator.generate(
    start_text="hello world",
    max_new_tokens=20,
    temperature=1.0
)

# Generate string
text = generator.generate_string(
    start_text="hello world",
    max_new_tokens=20,
    temperature=1.0
)
print(text)
```

## 💡 Use Cases & Voorbeelden

### Use Case 1: Standaard Training

```python
import torch
from gpt_lib import Config, Tokenizer, create_dataloader, MiniGPT, Trainer

# Config
config = Config(embed_dim=64, epochs=50)

# Data
with open(config.data_path) as f:
    text = f.read()
tokenizer = Tokenizer()
tokenizer.build(text)
encoded = tokenizer.encode(text.split())

# Model
loader = create_dataloader(encoded, config.block_size, config.batch_size)
model = MiniGPT(tokenizer.get_vocab_size(), config.embed_dim, config.block_size, config.num_blocks)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion, config)
trainer.train(loader, config.epochs)
trainer.save(config.model_path, tokenizer.vocab, tokenizer.stoi, tokenizer.itos, config.block_size, tokenizer.get_vocab_size())
```

### Use Case 2: Ander Model Formaat

```python
# Train met groter model
config = Config(
    embed_dim=256,        # 4x groter
    num_blocks=8,         # 4x meer lagen
    batch_size=32,        # Kleinere batches
    epochs=200,
    learning_rate=5e-4,
    model_path="output/large_model.pth"
)

# Rest is hetzelfde! ✓
```

### Use Case 3: Instruction-Following

```python
# Trainingsdata in format:
# [INST] What is Python? [/INST] Python is a programming language...

# Alles gelijk, maar:
config = Config(
    block_size=16,        # Langere context
    num_blocks=4,         # Meer capacity
    epochs=100,
    model_path="output/instruction_model.pth"
)

# Training code identiek!
```

### Use Case 4: Fine-tuning op Bestaand Model

```python
# Load checkpoint
checkpoint = trainer.load("output/model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Continue training met nieuwe data
new_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Lower LR
trainer_new = Trainer(model, new_optimizer, criterion, config)
trainer_new.train(new_loader, epochs=20)

trainer_new.save("output/finetuned_model.pth", ...)
```

## 🎯 Advanced Features

### Custom Data Processing

```python
from gpt_lib import Tokenizer

tokenizer = Tokenizer(max_vocab=100000)

# Build on multiple files
combined_text = ""
for file in ["data1.txt", "data2.txt", "data3.txt"]:
    with open(file) as f:
        combined_text += f.read() + " "

tokenizer.build(combined_text)
```

### Model Evaluation

```python
# Bereken loss op validation set
model.eval()
val_loss = 0
with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
print(f"Validation Loss: {avg_val_loss:.4f}")
```

### Batch Generation

```python
# Generate multiple samples
prompts = [
    "machine learning",
    "artificial intelligence",
    "deep learning"
]

for prompt in prompts:
    text = generator.generate_string(prompt, max_new_tokens=15)
    print(f"{prompt} -> {text}")
```

## 📊 Voorbeeld Outputs

### Model Stats
```
Config(
  embed_dim: 64
  block_size: 8
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  num_blocks: 2
  model_path: output/mini_gpt.pth
  data_path: data/data.txt
  max_vocab: 50000
  max_data_size: 1000000
  device: cpu
)

Parameters: 649,201
```

### Training Output
```
Epoch 1/100: Loss = 10.2341 | Time = 38.42s
Epoch 2/100: Loss = 9.1234 | Time = 37.89s
...
Epoch 100/100: Loss = 2.1234 | Time = 35.21s

✓ Training completed in 0h 59m 15s
✓ Model saved to output/mini_gpt.pth
```

### Generation Output
```
Prompt: 'english is'
Temp 0.5: english is a simple and powerful language for programming
Temp 1.0: english is the most common language used in computer science
Temp 1.5: english is like mathematical structure programming very complex modern

Generation time: 0.23s
```

## 🔧 Troubleshooting

### Out of Memory
- Verklein `batch_size` (64 → 32)
- Verklein `embed_dim` (64 → 32)
- Verklein `block_size` (8 → 4)

### Slow Training
- Verhoog `batch_size` voor snellere processing
- Gebruik GPU (`device="cuda"`)
- Verklein dataset (`max_data_size`)

### Bad Generated Text
- Meer training epochs
- Groter model (meer `num_blocks`)
- Betere trainingsdata

## 📖 Documentatie Links

- [PyTorch Documentation](https://pytorch.org)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1905.04467)

## 📚 Advanced Features Documentation

Voor volledige documentatie over repetition penalty, sampling, en generalisatie:
→ Zie [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)

## 📝 Licentie

MIT License - Vrij te gebruiken en aan te passen

---

**Version:** 1.0.0  
**Last Updated:** 2026-06-08
