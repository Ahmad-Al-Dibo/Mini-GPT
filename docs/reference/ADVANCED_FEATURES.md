# Advanced Features Guide

## 🎯 Repetition Penalty

**Doel:** Voorkomen dat het model dezelfde woorden herhaaldelijk genereert

### Hoe het werkt

```
Normale generatie:
"programming is programming programming..."  ← Repetitief!

Met repetition_penalty=2.0:
"programming is a powerful language..."  ← Beter!
```

### Mathematische Formule

$$\text{logits}_{\text{penalized}}[t] = \begin{cases}
\frac{\text{logits}[t]}{\text{penalty}} & \text{if } t \text{ in recent tokens} \\
\text{logits}[t] & \text{otherwise}
\end{cases}$$

### Gebruik

```python
from gpt_lib import Generator

# Basis (geen penalty)
text = generator.generate_string(
    "hello world",
    max_new_tokens=20,
    repetition_penalty=1.0  # No penalty
)

# Met penalty
text = generator.generate_string(
    "hello world",
    max_new_tokens=20,
    repetition_penalty=1.5  # Moderate penalty
)

# Hoge penalty
text = generator.generate_string(
    "hello world",
    max_new_tokens=20,
    repetition_penalty=2.0  # Strong penalty
)
```

### Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `1.0` | baseline | Geen penalty |
| `1.2-1.5` | mild | Minder repetities |
| `1.5-2.0` | moderate | Veel minder repetities |
| `2.0+` | strong | Zeer weinig repetities |

---

## 🎲 Sampling Strategieën

Verschillende methoden om volgende token te selecteren

### 1. Standard Sampling

```python
text = generator.generate_string(
    "hello",
    temperature=1.0  # Default
)
```

**Effect:** Willekeurig samplen volgens model kansen

### 2. Top-K Sampling

```python
text = generator.generate_string(
    "hello",
    top_k=50  # Keep only top 50 tokens
)
```

**Effect:** Filter eerst tot top K tokens met hoogste waarschijnlijkheid

### 3. Top-P (Nucleus) Sampling

```python
text = generator.generate_string(
    "hello",
    top_p=0.9  # Keep tokens with cum_prob >= 0.9
)
```

**Effect:** Selecteer tokens totdat cumulatieve waarschijnlijkheid >= p

### Combinatie

```python
text = generator.generate_string(
    "hello",
    temperature=1.2,        # Wat meer variatie
    repetition_penalty=1.5, # Minder herhaling
    top_p=0.9,              # Nucleus sampling
    top_k=50                # Top-k fallback
)
```

---

## 📊 Generalisatie Monitoring

**Doel:** Detecteren wanneer model begint te overfitten

### Concept

```
Trainings Loop:
  Train Loss: 5.2 → 3.1 → 2.4 → 1.8 (Daalt)
  Val Loss:   5.5 → 3.4 → 2.8 → 3.2 (Stijgt) ⚠️ OVERFITTING!
  
  Generalization Gap = Val Loss - Train Loss = 1.4
```

### Gebruik

```python
from gpt_lib import GeneralizationMonitor

# Create monitor
monitor = GeneralizationMonitor()

# During training
for epoch in range(epochs):
    train_loss = ...  # Calculate
    val_loss = ...    # Calculate
    
    monitor.update(train_loss, val_loss)

# Check status
gap = monitor.get_generalization_gap()
is_overfitting = monitor.is_overfitting(threshold=0.5)

# Get report
print(monitor.get_report())
```

### Output

```
Generalization Report:
  Latest Train Loss: 1.8234
  Latest Val Loss:   3.1456
  Generalization Gap: 1.3222
  Avg Gap (10-epoch): 1.1234
  Status: ⚠️  OVERFITTING
```

---

## 🛑 Early Stopping

**Doel:** Stop training wanneer model niet meer verbetert

### Concept

```
Epoch 1: Val Loss = 5.0 ✓ (New best!)
Epoch 2: Val Loss = 4.2 ✓ (New best!)
Epoch 3: Val Loss = 4.1 ✓ (New best!)
Epoch 4: Val Loss = 4.3   (No improvement) [counter=1]
Epoch 5: Val Loss = 4.4   (No improvement) [counter=2]
...
Epoch 16: Val Loss = 4.5  (No improvement) [counter=12]
Epoch 17: Val Loss = 4.6  (No improvement) [counter=13]
Epoch 18: Val Loss = 4.7  (No improvement) [counter=14]
Epoch 19: Val Loss = 4.8  (No improvement) [counter=15]
          STOP! (reached patience=15)
```

### Gebruik

```python
from gpt_lib import EarlyStopping, Trainer

# Create early stopper
early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

# In training loop
for epoch in range(epochs):
    val_loss = validate()
    
    if early_stopper.check(val_loss):
        print(f"Stop at epoch {epoch}")
        break

# In trainer
trainer.train(
    loader,
    epochs=100,
    val_loader=val_loader,
    early_stopping_patience=15  # Built-in!
)
```

---

## 🔄 Learning Rate Scheduling

**Doel:** Aanpassen learning rate tijdens training

### Strategieën

#### 1. Cosine Decay

```python
from gpt_lib import LearningRateScheduler

scheduler = LearningRateScheduler(
    optimizer,
    strategy="cosine",
    total_epochs=100
)

for epoch in range(100):
    # Train...
    scheduler.step(epoch)  # LR smoothly decreases
```

**Effect:** Learning rate volgt cosine curve

#### 2. Linear Decay

```python
scheduler = LearningRateScheduler(
    optimizer,
    strategy="linear",
    total_epochs=100
)
```

**Effect:** Learning rate lineair naar 0

#### 3. Exponential Decay

```python
scheduler = LearningRateScheduler(
    optimizer,
    strategy="exponential",
    total_epochs=100
)
```

**Effect:** Learning rate exponentieel afnemen

---

## 🎯 Regularisatie Technieken

### L2 Regularization (Weight Decay)

```python
from gpt_lib import L2Regularization

l2_reg = L2Regularization(lambda_l2=1e-4)

# In training loop
ce_loss = criterion(...)
reg_loss = l2_reg.compute_loss(model)
total_loss = ce_loss + reg_loss
total_loss.backward()
```

**Effect:** Kleine gewichten worden geprefereerd

### L1 Regularization

```python
from gpt_lib import L1Regularization

l1_reg = L1Regularization(lambda_l1=1e-5)

# In training loop
ce_loss = criterion(...)
reg_loss = l1_reg.compute_loss(model)
total_loss = ce_loss + reg_loss
```

**Effect:** Sparsity-inducing (veel nul gewichten)

### Label Smoothing

```python
from gpt_lib import LabelSmoothing

criterion = LabelSmoothing(num_classes=50000, smoothing=0.1)

# Use in training
loss = criterion(logits, targets)
```

**Effect:** Voorkomt overconfident voorspellingen

### Mixup Augmentation

```python
from gpt_lib import MixupAugmentation

mixup = MixupAugmentation(alpha=1.0)

# In training loop
x_mixed, y_mixed = mixup(x, y)
logits = model(x_mixed)
loss = criterion(logits, y_mixed)
```

**Effect:** Smooth decision boundaries

---

## 📋 Volledige Training Voorbeeld

```python
import torch
import torch.nn as nn
from gpt_lib import (
    Config, Tokenizer, create_dataloader,
    MiniGPT, Trainer,
    L2Regularization, EarlyStopping,
    GeneralizationMonitor, LearningRateScheduler
)

# Config
config = Config(
    embed_dim=64,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3
)

# Data
tokenizer = Tokenizer()
tokenizer.build(text)
train_loader = create_dataloader(train_encoded, 8, 64)
val_loader = create_dataloader(val_encoded, 8, 64)

# Model
model = MiniGPT(tokenizer.get_vocab_size(), config.embed_dim)
model = model.to("cuda")

# Regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
l2_reg = L2Regularization(1e-4)
scheduler = LearningRateScheduler(optimizer, "cosine", 100)
early_stopper = EarlyStopping(patience=15)
monitor = GeneralizationMonitor()

# Training
trainer = Trainer(model, optimizer, criterion, config)
trainer.train(train_loader, 100, val_loader, early_stopping_patience=15)
trainer.save(model_path, ...)

# Report
print(monitor.get_report())
```

---

## 🚀 Best Practices

| Feature | When to Use | Recommended Value |
|---------|------------|-------------------|
| **Repetition Penalty** | Veel repetitie in tekst | `1.2-1.5` |
| **Top-P Sampling** | Meer kontrole over output | `0.8-0.95` |
| **Top-K Sampling** | Beperkt vocab | `top_k=50` |
| **Early Stopping** | Overfitting | `patience=10-20` |
| **L2 Regularization** | Grote modellen | `lambda=1e-4` |
| **Learning Rate Scheduling** | Lang training | `cosine` strategy |
| **Label Smoothing** | Overconfident model | `smoothing=0.1` |

---

## 🔗 Voorbeelden

Run voorbeelden:

```bash
# Repetition penalty & sampling demo
python examples/example4_advanced.py penalty
python examples/example4_advanced.py sampling

# Advanced training met validatie en early stopping
python examples/example5_advanced_training.py data/data.txt

# Generalization monitoring
python examples/example4_advanced.py generalization
```

---

**Versie:** 1.0.0  
**Last Updated:** 2026-06-08
