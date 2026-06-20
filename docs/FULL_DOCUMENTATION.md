# MiniGPT - Complete Project Documentation

**Version:** 1.0  
**Last Updated:** 2026-06-20  
**Project Type:** Educational GPT-Style Language Model Framework

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Training Guide](#training-guide)
7. [Inference & Generation](#inference--generation)
8. [API Reference](#api-reference)
9. [Web Interface](#web-interface)
10. [Examples](#examples)
11. [Data Preparation](#data-preparation)
12. [Evaluation & Diagnostics](#evaluation--diagnostics)
13. [FAQ](#faq)
14. [Troubleshooting](#troubleshooting)

---

## Overview

### What is MiniGPT?

MiniGPT is an **educational framework** for understanding and implementing Transformer-based language models. It demonstrates how GPT-style models work by providing a clean, readable implementation of core concepts including:

- **Multi-Head Self-Attention**: Allows the model to focus on different parts of input sequences
- **Transformer Blocks**: Core building blocks combining attention and feed-forward networks
- **Autoregressive Generation**: Predicts one token at a time, conditioning on previous tokens
- **Instruction Following**: Fine-tuning for question-answering tasks

### Key Features

✅ **Educational Focus**: Clean, well-commented code prioritizing understanding over performance  
✅ **Complete Pipeline**: From raw text to trained model to inference  
✅ **Multiple Model Sizes**: MiniGPT, MediumGPT, and variants for different use cases  
✅ **Flexible Training**: Support for standard language modeling, instruction tuning, and domain-specific fine-tuning  
✅ **Advanced Generation**: Temperature sampling, top-k/top-p, and repetition penalty  
✅ **Web Interface**: Flask-based REST API for easy integration  
✅ **Diagnostic Tools**: Built-in evaluation and benchmarking utilities  
✅ **Pre-trained Models**: Multiple checkpoints available in `output/` directory

### Ideal For

- **Students** learning about Transformer architecture
- **Researchers** experimenting with language models
- **Developers** building NLP applications
- **Educational Projects** teaching deep learning concepts

---

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd Mini-GPT

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Text (Pre-trained Model)

```python
from gpt_lib.inference import LoadedModel

# Load a pre-trained model
model = LoadedModel("output/MiniGPT.pth")

# Generate text
prompt = "The future of artificial intelligence is"
generated = model.predict(prompt, max_new_tokens=50)
print(generated)
```

### 3. Start Web Interface

```bash
python app.py
```

Access the web UI at `http://localhost:5000`

### 4. Train Your Own Model

```python
from gpt_lib.pipeline import Pipeline

# Create and train a model
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# Train on your data
train_losses, val_losses = trainer.train(
    train_loader,
    val_loader,
    epochs=10,
    checkpoint_path="output/my_model.pth"
)
```

---

## Architecture

### Model Architecture Overview

```
Input Text
    ↓
Tokenization (SentencePiece BPE)
    ↓
Token Embedding + Positional Embedding
    ↓
GPT Block Stack (N blocks of):
    ├─ Multi-Head Self-Attention
    ├─ Feed-Forward Network
    ├─ Layer Normalization
    └─ Residual Connections
    ↓
Output Linear Layer
    ↓
Logits (probability distribution over vocabulary)
    ↓
Next Token Prediction / Text Generation
```

### Core Components

#### 1. **Self-Attention Mechanism**
- Computes query, key, value projections
- Applies causal masking (can only attend to past tokens)
- Performs softmax normalization
- Outputs weighted sum of values

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k + mask) · V
```

#### 2. **Multi-Head Attention**
- Multiple attention heads operate in parallel
- Allows model to attend to different representation subspaces
- Outputs concatenated and projected

```
MultiHeadAttention = Concat(head_1, ..., head_h) · W^O
```

#### 3. **Feed-Forward Network**
- Two linear layers with activation
- Expands to 4× hidden dimension then contracts back
- Applied position-wise to each token

```
FFN(x) = Dropout(ReLU(x · W_1 + b_1)) · W_2 + b_2
```

#### 4. **GPT Block**
- Combines attention and FFN with residual connections
- Layer normalization applied before each sublayer (Pre-LN architecture)

```
Block(x) = x + FFN(LayerNorm(x + Attention(LayerNorm(x))))
```

### Model Variants

| Variant | Hidden Dim | Heads | Blocks | Parameters | Use Case |
|---------|-----------|-------|--------|-----------|----------|
| MiniGPT | 256 | 4 | 4 | ~2M | Learning, experimentation |
| MediumGPT | 512 | 8 | 8 | ~11M | General purpose, better quality |
| MediumGPT-T | 512 | 8 | 8 | ~11M | Instruction-following variant |

### Configuration Parameters

```python
# Core Model
embed_dim = 256          # Embedding dimension
num_heads = 4            # Number of attention heads
num_blocks = 4           # Number of transformer blocks
vocab_size = 4096        # Vocabulary size (BPE)
block_size = 128         # Context length / sequence length

# Training
learning_rate = 1e-3     # Adam learning rate
weight_decay = 1e-4      # L2 regularization
dropout = 0.1            # Dropout rate
batch_size = 32          # Training batch size

# Generation
temperature = 0.7        # Sampling temperature
top_k = 40               # Top-k sampling
top_p = 0.9              # Top-p (nucleus) sampling
max_new_tokens = 100     # Maximum generation length
```

---

## Installation

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **PyTorch 1.9+** (2.0+ recommended)
- **pip** or **conda** for package management

### Step-by-Step Installation

#### Option 1: Using pip (Recommended)

```bash
# 1. Navigate to project directory
cd Mini-GPT

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.__version__)"
```

#### Option 2: Using Conda

```bash
# Create conda environment
conda create -n minigpt python=3.10
conda activate minigpt

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

Run a simple test:

```python
# test_installation.py
from gpt_lib.model import MiniGPT
from gpt_lib.config import Config

# Create model
config = Config()
model = MiniGPT(config)
print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
```

---

## Configuration

### Configuration File Structure

The project uses a centralized configuration system in `gpt_lib/config.py`:

```python
from gpt_lib.config import Config

# Get default configuration
config = Config()

# Modify parameters
config.embed_dim = 512
config.num_heads = 8
config.learning_rate = 5e-4
```

### Common Configuration Scenarios

#### Scenario 1: Small Model for Quick Experimentation

```python
config = Config()
config.embed_dim = 128
config.num_heads = 2
config.num_blocks = 2
config.batch_size = 16
config.block_size = 64
```

#### Scenario 2: Larger Model for Better Quality

```python
config = Config()
config.embed_dim = 768
config.num_heads = 12
config.num_blocks = 12
config.batch_size = 64
config.block_size = 256
config.learning_rate = 5e-4
```

#### Scenario 3: Instruction Following (Fine-tuning)

```python
config = Config()
config.embed_dim = 512
config.num_heads = 8
config.num_blocks = 8
config.learning_rate = 1e-4  # Lower LR for fine-tuning
config.dropout = 0.15
```

### Key Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `embed_dim` | 256 | 64-1024 | Embedding dimension |
| `num_heads` | 4 | 1-16 | Number of attention heads |
| `num_blocks` | 4 | 1-24 | Transformer blocks |
| `vocab_size` | 4096 | 1024-50000 | Vocabulary size |
| `block_size` | 128 | 32-512 | Context window |
| `dropout` | 0.1 | 0.0-0.5 | Dropout rate |
| `learning_rate` | 1e-3 | 1e-5 to 1e-2 | Adam learning rate |
| `batch_size` | 32 | 8-256 | Training batch size |
| `weight_decay` | 1e-4 | 0.0-1e-2 | L2 regularization |

---

## Training Guide

### Basic Training

```python
from gpt_lib.pipeline import Pipeline
from gpt_lib.dataset import TextDataset
from torch.utils.data import DataLoader

# Step 1: Initialize pipeline
pipeline = Pipeline()

# Step 2: Build model and trainer
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# Step 3: Prepare data
train_dataset = TextDataset("data/train.txt", block_size=128)
val_dataset = TextDataset("data/val.txt", block_size=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 4: Train model
train_losses, val_losses = trainer.train(
    train_loader,
    val_loader,
    epochs=20,
    checkpoint_path="output/my_model.pth"
)

# Step 5: Save final model
pipeline.save_model(model, "output/my_model_final.pth")
```

### Instruction Tuning

```python
from gpt_lib.data import create_instruction_following_dataset

# Prepare instruction-following data
# Format: [INST] question [/INST] answer
dataset, tokenizer = create_instruction_following_dataset(
    csv_path="data/instructions.csv"
)

# Train with lower learning rate for fine-tuning
trainer.learning_rate = 1e-4
train_losses, val_losses = trainer.train(
    train_loader,
    val_loader,
    epochs=10,
    checkpoint_path="output/instruction_model.pth"
)
```

### Advanced Training Features

#### 1. **Early Stopping**

```python
trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    early_stopping_patience=5,  # Stop if no improvement for 5 epochs
    early_stopping_min_delta=0.001,
    checkpoint_path="output/model.pth"
)
```

#### 2. **Learning Rate Scheduling** (Custom)

```python
import torch.optim as optim

# Create optimizer with schedule
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# In training loop
for epoch in range(epochs):
    train_loss = trainer.train_epoch(train_loader)
    scheduler.step()
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

#### 3. **Gradient Accumulation** (For larger effective batch size)

```python
accumulation_steps = 4
optimizer.zero_grad()

for batch_idx, (X, Y) in enumerate(train_loader):
    logits = model(X)
    loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 4. **Mixed Precision Training** (For faster training on GPUs)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X, Y in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        logits = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Training Best Practices

1. **Start small**: Begin with MiniGPT configuration to verify pipeline works
2. **Monitor metrics**: Track training/validation loss, perplexity, accuracy
3. **Use checkpoints**: Save best model based on validation loss
4. **Validate frequently**: Check validation loss every epoch
5. **Adjust learning rate**: Lower LR for fine-tuning, higher for pre-training
6. **Use regularization**: Apply dropout, weight decay to prevent overfitting
7. **Data quality**: Clean data significantly improves results

---

## Inference & Generation

### Basic Inference

```python
from gpt_lib.inference import LoadedModel

# Load model
model = LoadedModel("output/MiniGPT.pth")

# Single prediction
prompt = "The future of AI is"
result = model.predict(prompt, max_new_tokens=50)
print(result)
```

### Advanced Generation with Sampling

```python
# Temperature sampling (higher = more random)
result = model.predict(
    prompt="The future of AI is",
    max_new_tokens=50,
    temperature=0.7,  # 0.1-2.0 range
)

# Top-k sampling (keep only top k likely tokens)
result = model.predict(
    prompt="The future of AI is",
    max_new_tokens=50,
    top_k=40,
)

# Top-p (nucleus) sampling (keep tokens until probability reaches p)
result = model.predict(
    prompt="The future of AI is",
    max_new_tokens=50,
    top_p=0.9,
)

# With repetition penalty (prevent repeating tokens)
result = model.predict(
    prompt="The future of AI is",
    max_new_tokens=50,
    repetition_penalty=1.2,
)
```

### Batch Inference

```python
# Multiple prompts
prompts = [
    "The future of AI is",
    "Machine learning helps us",
    "Python is a great language for"
]

results = model.predict_batch(
    prompts,
    max_new_tokens=50,
    temperature=0.7
)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result}\n")
```

### Generation Parameters Explained

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `temperature` | 0.7 | 0.1-2.0 | Lower = more deterministic, Higher = more creative |
| `top_k` | 40 | 1-vocab_size | Keep only top k tokens |
| `top_p` | 0.9 | 0.0-1.0 | Keep tokens until probability >= p |
| `repetition_penalty` | 1.0 | 1.0-2.0 | Penalize already-generated tokens |
| `max_new_tokens` | 100 | 1-block_size | Maximum tokens to generate |

---

## API Reference

### LoadedModel Class

```python
class LoadedModel:
    def __init__(self, checkpoint_path):
        """Load a trained model from checkpoint"""
        
    def predict(self, prompt, max_new_tokens=100, **kwargs):
        """Generate text from prompt
        
        Args:
            prompt (str): Starting text
            max_new_tokens (int): Max tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p (nucleus) sampling
            
        Returns:
            str: Generated text (including prompt)
        """
        
    def predict_batch(self, prompts, max_new_tokens=100, **kwargs):
        """Generate text for multiple prompts
        
        Args:
            prompts (List[str]): List of prompts
            max_new_tokens (int): Max tokens to generate
            
        Returns:
            List[str]: Generated texts
        """
```

### Pipeline Class

```python
class Pipeline:
    def build_model_and_trainer(self, model_type="MiniGPT"):
        """Build model and trainer
        
        Args:
            model_type (str): "MiniGPT", "MediumGPT", or "MediumGPT-T"
            
        Returns:
            Tuple[model, trainer]
        """
        
    def save_model(self, model, path):
        """Save model checkpoint with metadata"""
        
    def load_model(self, path):
        """Load model from checkpoint"""
```

### Trainer Class

```python
class Trainer:
    def train(self, train_loader, val_loader, epochs=10, 
              checkpoint_path=None, early_stopping_patience=None):
        """Train model
        
        Args:
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
            epochs (int): Number of epochs
            checkpoint_path (str): Where to save checkpoints
            early_stopping_patience (int): Patience for early stopping
            
        Returns:
            Tuple[train_losses, val_losses]
        """
```

### Config Class

```python
class Config:
    embed_dim = 256              # Embedding dimension
    num_heads = 4                # Attention heads
    num_blocks = 4               # Transformer blocks
    vocab_size = 4096            # Vocabulary size
    block_size = 128             # Context length
    dropout = 0.1                # Dropout rate
    learning_rate = 1e-3         # Learning rate
    batch_size = 32              # Batch size
    weight_decay = 1e-4          # L2 regularization
```

---

## Web Interface

### Starting the Web Server

```bash
python app.py
```

Server runs on `http://localhost:5000`

### Web API Endpoints

#### 1. **GET / (Main Interface)**
Returns HTML interface for text generation

#### 2. **POST /generate**
Generate text from a prompt

**Request:**
```json
{
    "prompt": "The future of AI is",
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
}
```

**Response:**
```json
{
    "generated_text": "The future of AI is a topic that is being discussed by many...",
    "status": "success"
}
```

#### 3. **POST /api/predict**
Alternative endpoint for predictions

**Request:**
```json
{
    "prompt": "Machine learning is",
    "max_new_tokens": 50
}
```

**Response:**
```json
{
    "prediction": "Machine learning is a subset of artificial intelligence...",
    "status": "success"
}
```

### Example Web Requests

#### Using curl

```bash
# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Python is great for",
    "max_new_tokens": 30,
    "temperature": 0.7
  }'
```

#### Using Python requests

```python
import requests

url = "http://localhost:5000/generate"
data = {
    "prompt": "Artificial intelligence",
    "max_new_tokens": 50,
    "temperature": 0.8
}

response = requests.post(url, json=data)
print(response.json())
```

#### Using JavaScript

```javascript
const prompt = "The future of technology is";
const params = {
    prompt: prompt,
    max_new_tokens: 50,
    temperature: 0.7
};

fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
})
.then(res => res.json())
.then(data => console.log(data.generated_text));
```

---

## Examples

### Example 1: Basic Text Generation

File: [examples/example1_generation.py](../examples/example1_generation.py)

```python
from gpt_lib.inference import LoadedModel

# Load pre-trained model
model = LoadedModel("output/MiniGPT.pth")

# Generate multiple texts
prompts = [
    "The future of artificial intelligence",
    "Python programming language",
    "Deep learning models are"
]

for prompt in prompts:
    generated = model.predict(prompt, max_new_tokens=50, temperature=0.7)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
```

### Example 2: Custom Training

File: [examples/example2_custom_training.py](../examples/example2_custom_training.py)

```python
from gpt_lib.pipeline import Pipeline
from gpt_lib.dataset import TextDataset
from torch.utils.data import DataLoader

# Initialize pipeline
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

# Prepare dataset
dataset = TextDataset("data/data.txt", block_size=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Split for validation
val_size = len(dataset) // 10
train_size = len(dataset) - val_size

# Train model
train_losses, val_losses = trainer.train(
    loader, loader,
    epochs=20,
    checkpoint_path="output/custom_model.pth"
)
```

### Example 3: Instruction Following

File: [examples/example3_instruction_training.py](../examples/example3_instruction_training.py)

```python
from gpt_lib.pipeline import Pipeline
from gpt_lib.data import create_instruction_following_dataset
from torch.utils.data import DataLoader

# Load instruction data
dataset, tokenizer = create_instruction_following_dataset(
    "data/instructions.csv"
)

# Build model
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MediumGPT-T")

# Fine-tune on instructions
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=16)

trainer.learning_rate = 1e-4
trainer.train(
    train_loader,
    val_loader,
    epochs=5,
    checkpoint_path="output/instruction_model.pth"
)
```

### Example 4: Advanced Usage

File: [examples/example4_advanced.py](../examples/example4_advanced.py)

- Model ensemble
- Custom sampling strategies
- Attention visualization
- Token analysis

### Example 5: Advanced Training

File: [examples/example5_advanced_training.py](../examples/example5_advanced_training.py)

- Multi-GPU training
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling

---

## Data Preparation

### Supported Data Formats

#### 1. **Plain Text Files**
```
This is a simple text file.
Each line is a document.
The tokenizer will handle splitting.
```

Loading:
```python
from gpt_lib.dataset import TextDataset

dataset = TextDataset("data/text.txt", block_size=128)
```

#### 2. **CSV Files (Question-Answer)**
```csv
question,answer
What is AI?,Artificial Intelligence is a field of computer science
What is ML?,Machine Learning is a subset of AI
```

Loading:
```python
from gpt_lib.data import load_csv_dataset

dataset = load_csv_dataset("data/qa.csv")
```

#### 3. **JSON Files**
```json
[
  {"text": "First document text..."},
  {"text": "Second document text..."}
]
```

Loading:
```python
import json
from gpt_lib.dataset import TextDataset

with open("data.json") as f:
    data = json.load(f)
    texts = [item["text"] for item in data]
```

#### 4. **Markdown Files** (Extracted in dataset/ directory)

Pre-processed markdown files on various topics:
- Programming languages (Python, JavaScript, Java, etc.)
- Cloud platforms (AWS, Azure, GCP)
- Databases (MongoDB, MySQL, PostgreSQL)

### Data Cleaning Pipeline

```python
from gpt_lib.data import clean_text

# Clean individual texts
text = "  Hello world!   \n\n"
cleaned = clean_text(text)

# Batch cleaning
texts = [...]
cleaned_texts = [clean_text(t) for t in texts]
```

### Creating Train/Val Splits

```python
from gpt_lib.data import train_val_split
from gpt_lib.dataset import TextDataset

# Load data
dataset = TextDataset("data/all_text.txt", block_size=128)

# Create 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

from torch.utils.data import random_split
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)
```

---

## Evaluation & Diagnostics

### Perplexity

Measure how well the model predicts sequences (lower is better):

```python
from evaluation.diagnostic import calculate_perplexity

perplexity = calculate_perplexity(model, val_loader)
print(f"Validation Perplexity: {perplexity:.2f}")
```

### Token Accuracy

Percentage of correctly predicted next tokens:

```python
from evaluation.diagnostic import calculate_accuracy

accuracy = calculate_accuracy(model, val_loader)
print(f"Token Accuracy: {accuracy:.2%}")
```

### Concept Understanding

Evaluate if model understands key concepts:

```python
from evaluation.diagnostic_simple import test_concept

# Test if model knows about Python
score = test_concept(model, "Python", num_samples=100)
print(f"Python concept score: {score:.2%}")
```

### Long-Context Evaluation

Test model's ability with long sequences:

```python
from evaluation.diagnostic import evaluate_long_context

scores = evaluate_long_context(
    model,
    context_lengths=[64, 128, 256, 512]
)

for length, score in scores.items():
    print(f"Length {length}: {score:.2%}")
```

### Running Full Diagnostic Suite

```python
from evaluation.diagnostic import full_diagnostic

results = full_diagnostic(model, val_loader)
print(results)
```

---

## FAQ

### Q: What hardware do I need?

**A:** Minimum: 4GB RAM, CPU. Recommended: GPU (NVIDIA with CUDA) for faster training. 
- CPU training: ~1 hour for MiniGPT on small dataset
- GPU training: ~10 minutes on NVIDIA RTX series

### Q: How much data do I need?

**A:** 
- Minimum: 10MB of text for reasonable results
- Recommended: 100MB+ for diverse, high-quality models
- Start with smaller datasets to test the pipeline

### Q: Can I use this on production?

**A:** Not recommended. This is an educational project prioritizing clarity over performance. For production, use:
- Llama 2
- GPT-3.5-Turbo / GPT-4
- Claude
- Mistral

### Q: How do I fine-tune for specific domains?

**A:** 
1. Prepare domain-specific text data
2. Use lower learning rate (1e-4 to 5e-5)
3. Fewer epochs (3-10)
4. Optional: Start from pre-trained checkpoint

### Q: What's the difference between temperature and top-k sampling?

**A:** 
- **Temperature**: Controls randomness (0.1=deterministic, 2.0=very random)
- **Top-k**: Only considers k most likely next tokens
- Use together for best results

### Q: How do I increase model quality?

**A:**
1. Increase model size (more embed_dim, blocks, heads)
2. Provide more training data
3. Train longer (more epochs)
4. Fine-tune on domain-specific data
5. Use lower learning rate during training

### Q: Can I export the model?

**A:** Yes, models are saved as PyTorch `.pth` files which include:
- Model weights
- Configuration
- Tokenizer
- Training metadata

### Q: How do I debug training issues?

**A:** 
1. Check training/validation loss trends
2. Verify data is being loaded correctly
3. Use gradient clipping if loss explodes
4. Reduce learning rate if loss is erratic
5. Use diagnostic tools to check model predictions

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install torch sentencepiece flask numpy
```

### Issue: Out of Memory (OOM) error

**Solution:**
1. Reduce batch size: `config.batch_size = 16` (or lower)
2. Reduce model size: Use MiniGPT instead of MediumGPT
3. Reduce sequence length: `config.block_size = 64`
4. Use gradient accumulation to simulate larger batches

### Issue: Training loss not decreasing

**Possible causes and solutions:**
1. Learning rate too high → reduce to 5e-4
2. Data quality issues → check your data
3. Model too small → increase model size
4. Check for NaN values: `torch.isnan(loss).any()`

### Issue: Generated text is gibberish

**Solutions:**
1. Model not sufficiently trained → train more epochs
2. Temperature too high → reduce to 0.5
3. Model overfitted → add more training data or increase dropout

### Issue: Model loading fails

**Solution:**
```python
# Verify checkpoint exists
import os
assert os.path.exists("output/model.pth"), "Checkpoint not found"

# Check checkpoint format
import torch
checkpoint = torch.load("output/model.pth")
print(checkpoint.keys())  # Should contain 'model', 'config', etc.
```

### Issue: Tokenizer errors

**Solution:**
```python
# Verify tokenizer exists
import os
assert os.path.exists("data/tokenizer.json"), "Tokenizer not found"

# Retrain tokenizer if needed
from gpt_lib.tokenizer import train_tokenizer
train_tokenizer("data/all_text.txt", vocab_size=4096)
```

---

## Project Structure

```
Mini-GPT/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── dataset.json                    # Dataset configuration
├── README.md                       # Original README
│
├── config/                         # Configuration files
│   ├── n_training.py              # Training configuration
│   └── tuining.py                 # Alternative config
│
├── data/                           # Training data
│   ├── all_cleaned_text.md        # Cleaned combined text
│   ├── tokenizer.json             # SentencePiece tokenizer
│   ├── alpaca.csv                 # Instruction data
│   ├── qa.csv                     # Q&A data
│   └── cleaner/                   # Data cleaning utilities
│
├── dataset/                        # Knowledge base markdown files
│   ├── Python (programming language).md
│   ├── JavaScript.md
│   └── ... (20+ more files)
│
├── gpt_lib/                        # Main library
│   ├── __init__.py
│   ├── model.py                   # Core model architecture
│   ├── trainer.py                 # Training loop
│   ├── generator.py               # Text generation
│   ├── inference.py               # Inference interface
│   ├── tokenizer.py               # Tokenization
│   ├── dataset.py                 # Data loading
│   ├── config.py                  # Configuration
│   ├── pipeline.py                # Training pipeline
│   ├── diagnostics.py             # Evaluation tools
│   ├── data.py                    # Data utilities
│   ├── regularization.py          # Regularization techniques
│   ├── utils.py                   # Helper functions
│   └── logics/                    # Logical reasoning module
│
├── examples/                       # Usage examples
│   ├── example1_generation.py
│   ├── example2_custom_training.py
│   ├── example3_instruction_training.py
│   ├── example4_advanced.py
│   └── example5_advanced_training.py
│
├── evaluation/                     # Evaluation tools
│   ├── diagnostic.py              # Main diagnostic tools
│   ├── diagnostic_simple.py       # Simple diagnostics
│   ├── instruction_tuning.py      # Instruction tuning eval
│   ├── optimize.py                # Optimization tools
│   └── README.md
│
├── output/                         # Pre-trained models
│   ├── MiniGPT.pth
│   ├── MediumGPT.pth
│   ├── mini_gpt.pth
│   └── ... (10+ checkpoints)
│
├── notebooks/                      # Jupyter notebooks
│   ├── loading-data.ipynb
│   ├── loading-model.ipynb
│   ├── prepare_ins_data.ipynb
│   └── train.ipynb
│
├── docs/                           # Documentation
│   ├── README.md
│   ├── PROJECT_MAP.md
│   ├── last-raport.md
│   ├── FULL_DOCUMENTATION.md      # (This file)
│   └── development/
│
├── templates/                      # Web UI templates
│   └── index.html
│
└── tests/                          # Unit tests
    ├── test_config.py
    └── test_core.py
```

---

## Contributing

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo_url>
cd Mini-GPT

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest black flake8
```

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black gpt_lib/ examples/ tests/

# Check style
flake8 gpt_lib/ --max-line-length=100
```

---

## Performance Benchmarks

### Training Speed (on NVIDIA RTX 3090)

| Model | Dataset Size | Batch Size | Epoch Time |
|-------|--------------|-----------|-----------|
| MiniGPT | 10MB | 32 | ~30s |
| MiniGPT | 100MB | 32 | ~5m |
| MediumGPT | 10MB | 32 | ~1m |
| MediumGPT | 100MB | 32 | ~15m |

### Generation Speed

| Model | Device | Tokens/sec | Quality |
|-------|--------|-----------|---------|
| MiniGPT | CPU | ~20 | Basic |
| MiniGPT | GPU | ~500 | Basic |
| MediumGPT | CPU | ~10 | Good |
| MediumGPT | GPU | ~300 | Good |

### Model Sizes

| Model | Parameters | Checkpoint Size |
|-------|-----------|-----------------|
| MiniGPT | ~2M | 8MB |
| MediumGPT | ~11M | 45MB |
| MediumGPT-T | ~11M | 45MB |

---

## License

This project is provided for educational purposes. See LICENSE file for details.

---

## Resources

### Recommended Reading

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer paper

2. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
   - GPT-2 paper introducing large-scale language modeling

3. **"Language Models are Few-Shot Learners"** (Brown et al., 2020)
   - GPT-3 paper

### Online Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Course](https://huggingface.co/course/)
- [Neural Networks from Scratch](https://github.com/karpathy/makemore)

### Related Projects

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Llama](https://github.com/facebookresearch/llama)
- [GPT-2](https://github.com/openai/gpt-2)

---

## Support & Contact

For issues, questions, or suggestions:

1. Check the [FAQ](#faq) section
2. Review [Troubleshooting](#troubleshooting)
3. Check existing GitHub issues
4. Create a new issue with detailed description

---

**Last Updated:** June 20, 2026  
**Documentation Version:** 1.0  
**Project Version:** 1.0
