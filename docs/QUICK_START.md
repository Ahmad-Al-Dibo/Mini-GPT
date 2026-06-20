# Quick Start Guide

Get started with MiniGPT in 5 minutes!

## Installation (2 minutes)

```bash
# 1. Navigate to project
cd Mini-GPT

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Generate Text (1 minute)

### Option A: Use Pre-trained Model

```python
from gpt_lib.inference import LoadedModel

# Load model
model = LoadedModel("output/MiniGPT.pth")

# Generate text
result = model.predict(
    "The future of artificial intelligence is",
    max_new_tokens=50
)
print(result)
```

**Output:**
```
The future of artificial intelligence is bright and full of possibilities.
It will transform how we work and live.
```

### Option B: Web Interface

```bash
# Start server
python app.py

# Open browser
# http://localhost:5000
```

## Train Your Own Model (2 minutes)

### Minimal Training Example

```python
from gpt_lib.pipeline import Pipeline
from gpt_lib.dataset import TextDataset
from torch.utils.data import DataLoader

# 1. Load pipeline
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# 2. Create dataset
dataset = TextDataset("data/data.txt", block_size=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Train (on same data for demo)
train_losses, val_losses = trainer.train(
    loader, loader,  # Use same data for train/val for demo
    epochs=5,
    checkpoint_path="output/my_model.pth"
)

print(f"Final training loss: {train_losses[-1]:.4f}")
```

## Use Your Trained Model

```python
from gpt_lib.inference import LoadedModel

# Load your trained model
model = LoadedModel("output/my_model.pth")

# Generate text
result = model.predict("Once upon a time", max_new_tokens=50)
print(result)
```

## Common Tasks

### Task 1: Generate Different Styles

```python
# Deterministic (always same output)
text1 = model.predict(prompt, temperature=0.1)

# Creative (varied output)
text2 = model.predict(prompt, temperature=1.5)

# Balanced (default)
text3 = model.predict(prompt, temperature=0.7)
```

### Task 2: Fine-tune on Your Data

```python
from gpt_lib.pipeline import Pipeline
from gpt_lib.dataset import TextDataset
from torch.utils.data import DataLoader, random_split

# Load pre-trained model
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# Your data
dataset = TextDataset("my_data.txt", block_size=128)

# Split 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Create loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Fine-tune with lower learning rate
trainer.learning_rate = 1e-4

train_losses, val_losses = trainer.train(
    train_loader, val_loader,
    epochs=10,
    checkpoint_path="output/fine_tuned.pth"
)
```

### Task 3: Generate Multiple Texts

```python
prompts = [
    "Machine learning is",
    "Python programming is",
    "Artificial intelligence will"
]

results = model.predict_batch(prompts, max_new_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}\n")
```

### Task 4: Evaluate Your Model

```python
from evaluation.diagnostic import calculate_perplexity, calculate_accuracy

# Calculate metrics
perplexity = calculate_perplexity(model, val_loader)
accuracy = calculate_accuracy(model, val_loader)

print(f"Perplexity: {perplexity:.2f}")  # Lower is better
print(f"Accuracy: {accuracy:.2%}")      # Higher is better
```

## API Summary

### Loading & Generating

```python
from gpt_lib.inference import LoadedModel

model = LoadedModel("path/to/model.pth")

# Generate once
result = model.predict(prompt, max_new_tokens=50)

# Generate multiple
results = model.predict_batch([prompt1, prompt2], max_new_tokens=50)
```

### Training

```python
from gpt_lib.pipeline import Pipeline

pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# Train
train_losses, val_losses = trainer.train(
    train_loader, val_loader,
    epochs=20,
    checkpoint_path="model.pth"
)
```

### Configuration

```python
from gpt_lib.config import Config

config = Config(
    embed_dim=512,        # Model size
    learning_rate=5e-4,   # Training rate
    batch_size=64,        # Batch size
    num_epochs=20         # Epochs
)
```

## Next Steps

1. **Read Full Documentation**: See `FULL_DOCUMENTATION.md`
2. **Explore Examples**: Check `examples/` folder
3. **Try API Reference**: See `API_REFERENCE.md`
4. **Learn Architecture**: See `ARCHITECTURE_DEEP_DIVE.md`

## Troubleshooting

### "Module not found" Error

```bash
pip install --upgrade -r requirements.txt
```

### Out of Memory (OOM)

```python
# Reduce batch size
config.batch_size = 16

# Or reduce model size
model, trainer = pipeline.build_model_and_trainer("MiniGPT")  # Smaller model
```

### Model generates gibberish

```python
# Train longer
trainer.train(..., epochs=50)  # More epochs

# Lower temperature
model.predict(prompt, temperature=0.3)  # More focused
```

## Model Sizes

| Model | Speed | Quality | Memory |
|-------|-------|---------|--------|
| MiniGPT | Fast | Good | Low |
| MediumGPT | Slow | Better | High |

## File Locations

- **Models**: `output/*.pth`
- **Data**: `data/*.txt`, `data/*.csv`
- **Examples**: `examples/example*.py`
- **Web UI**: Templates in `templates/`

## Generation Parameters

```python
model.predict(
    prompt,
    max_new_tokens=50,      # How many tokens to generate
    temperature=0.7,        # 0.1=deterministic, 2.0=random
    top_k=40,              # Keep top 40 tokens
    top_p=0.9,             # Keep until 90% probability
    repetition_penalty=1.2  # Avoid repeating tokens
)
```

## Performance

- **CPU**: ~30 tokens/second
- **GPU**: ~500 tokens/second (RTX 3090)
- **Training**: ~30 seconds per epoch on RTX 3090

---

**Next:** Read `FULL_DOCUMENTATION.md` for complete guide
