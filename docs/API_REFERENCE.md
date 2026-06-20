# API Reference Guide

Complete API documentation for MiniGPT library modules.

## Table of Contents

1. [Model Classes](#model-classes)
2. [Training Classes](#training-classes)
3. [Inference Classes](#inference-classes)
4. [Data Loading](#data-loading)
5. [Tokenization](#tokenization)
6. [Utilities](#utilities)
7. [Configuration](#configuration)

---

## Model Classes

### MiniGPT

**File:** `gpt_lib/model.py`

Main transformer-based language model.

#### Constructor

```python
MiniGPT(config, device='cpu')
```

**Parameters:**
- `config` (Config): Configuration object with model hyperparameters
- `device` (str): 'cpu' or 'cuda'

**Attributes:**
```python
model.embedding_layer          # Token embedding layer
model.positional_embedding     # Positional embedding layer
model.blocks                   # Stack of transformer blocks
model.output_layer             # Final output projection
```

**Methods:**

```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
    """Forward pass of the model
    
    Args:
        idx: Tensor of shape (batch_size, seq_length)
             containing token indices
    
    Returns:
        logits: Tensor of shape (batch_size, seq_length, vocab_size)
                containing logits for each position
    """

def generate(self, idx: torch.Tensor, max_new_tokens: int, 
             temperature: float = 1.0, top_k: int = None,
             top_p: float = None) -> torch.Tensor:
    """Generate text autoregressively
    
    Args:
        idx: Initial token indices
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top k tokens
        top_p: Keep tokens until cumulative probability >= top_p
    
    Returns:
        Generated token indices
    """

def get_num_parameters(self) -> int:
    """Return total number of trainable parameters"""
```

#### Example

```python
from gpt_lib.model import MiniGPT
from gpt_lib.config import Config

# Create model
config = Config()
model = MiniGPT(config, device='cuda')

# Forward pass
logits = model(token_ids)  # (batch_size, seq_len, vocab_size)

# Generate
generated = model.generate(
    idx=start_tokens,
    max_new_tokens=50,
    temperature=0.7,
    top_k=40
)
```

---

### Self-Attention Layer

**File:** `gpt_lib/model.py`

Multi-head self-attention mechanism.

```python
class SelfAttention(nn.Module):
    """Multi-head self-attention with causal masking"""
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 block_size: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            block_size: Maximum sequence length
            dropout: Dropout rate
        """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, embed_dim)
            
        Returns:
            (batch_size, seq_length, embed_dim)
        """
```

---

### Transformer Block

**File:** `gpt_lib/model.py`

Single transformer block combining attention and FFN.

```python
class GPTBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, embed_dim: int, num_heads: int,
                 block_size: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            block_size: Context length
            dropout: Dropout rate
        """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block
        
        Args:
            x: (batch_size, seq_length, embed_dim)
            
        Returns:
            (batch_size, seq_length, embed_dim)
        """
```

---

## Training Classes

### Trainer

**File:** `gpt_lib/trainer.py`

Handles model training and validation.

#### Constructor

```python
Trainer(model, learning_rate=1e-3, weight_decay=1e-4, 
        device='cpu')
```

**Parameters:**
- `model`: MiniGPT model instance
- `learning_rate` (float): Adam learning rate
- `weight_decay` (float): L2 regularization
- `device` (str): 'cpu' or 'cuda'

**Attributes:**
```python
trainer.model                  # The model being trained
trainer.optimizer              # Adam optimizer instance
trainer.learning_rate          # Current learning rate
trainer.epoch_count            # Current epoch number
trainer.best_val_loss          # Best validation loss seen
```

#### Methods

```python
def train(self, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int = 10, checkpoint_path: str = None,
          early_stopping_patience: int = None,
          early_stopping_min_delta: float = 0.001) -> tuple:
    """Train the model
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        checkpoint_path: Where to save best model
        early_stopping_patience: Stop if no improvement for N epochs
        early_stopping_min_delta: Minimum improvement threshold
    
    Returns:
        (train_losses, val_losses): Lists of loss values
    """

def train_epoch(self, train_loader: DataLoader) -> float:
    """Train for one epoch
    
    Returns:
        Average training loss for the epoch
    """

def validate(self, val_loader: DataLoader) -> float:
    """Run validation
    
    Returns:
        Average validation loss
    """

def save_checkpoint(self, path: str):
    """Save model checkpoint"""

def load_checkpoint(self, path: str):
    """Load model from checkpoint"""
```

#### Example

```python
from gpt_lib.trainer import Trainer
from torch.utils.data import DataLoader

# Create trainer
trainer = Trainer(
    model=model,
    learning_rate=1e-3,
    device='cuda'
)

# Train model
train_losses, val_losses = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    checkpoint_path="model.pth",
    early_stopping_patience=5
)
```

---

## Inference Classes

### LoadedModel

**File:** `gpt_lib/inference.py`

Wrapper for loading and using trained models.

#### Constructor

```python
LoadedModel(checkpoint_path: str, device: str = 'cpu')
```

**Parameters:**
- `checkpoint_path` (str): Path to model checkpoint
- `device` (str): 'cpu' or 'cuda'

#### Methods

```python
def predict(self, prompt: str, max_new_tokens: int = 100,
            temperature: float = 0.7, top_k: int = 40,
            top_p: float = 0.9, 
            repetition_penalty: float = 1.0) -> str:
    """Generate text from a prompt
    
    Args:
        prompt: Starting text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.1-2.0)
        top_k: Keep only top k tokens
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Generated text (including prompt)
    """

def predict_batch(self, prompts: List[str], 
                 max_new_tokens: int = 100, **kwargs) -> List[str]:
    """Generate text for multiple prompts
    
    Args:
        prompts: List of prompts
        max_new_tokens: Max tokens per prompt
        **kwargs: Other generation parameters
    
    Returns:
        List of generated texts
    """

def encode(self, text: str) -> torch.Tensor:
    """Tokenize text to token IDs"""

def decode(self, token_ids: torch.Tensor) -> str:
    """Convert token IDs back to text"""
```

#### Example

```python
from gpt_lib.inference import LoadedModel

# Load model
model = LoadedModel("output/MiniGPT.pth", device='cuda')

# Single prediction
result = model.predict(
    "The future of AI is",
    max_new_tokens=50,
    temperature=0.7
)
print(result)

# Batch prediction
prompts = ["AI is", "Python is", "Deep learning is"]
results = model.predict_batch(prompts, max_new_tokens=50)
```

---

## Data Loading

### TextDataset

**File:** `gpt_lib/dataset.py`

Dataset for loading text files.

#### Constructor

```python
TextDataset(file_path: str, block_size: int = 128)
```

**Parameters:**
- `file_path` (str): Path to text file
- `block_size` (int): Sequence length (context window)

**Methods:**

```python
def __len__(self) -> int:
    """Number of sequences in dataset"""

def __getitem__(self, idx: int) -> tuple:
    """Get a single sequence
    
    Returns:
        (input_tokens, target_tokens)
    """
```

#### Example

```python
from gpt_lib.dataset import TextDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = TextDataset("data/text.txt", block_size=128)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for X, Y in loader:
    print(X.shape)  # (batch_size, block_size)
    print(Y.shape)  # (batch_size, block_size)
```

### InstructionDataset

**File:** `gpt_lib/dataset.py`

Dataset for instruction-following training.

```python
class InstructionDataset(Dataset):
    """Dataset with instruction-following format"""
    
    def __init__(self, data: List[Dict[str, str]], 
                 tokenizer, block_size: int = 128):
        """
        Args:
            data: List of dicts with 'instruction' and 'output' keys
            tokenizer: Tokenizer instance
            block_size: Sequence length
        """
```

---

## Tokenization

### Tokenizer

**File:** `gpt_lib/tokenizer.py`

BPE tokenizer using SentencePiece.

#### Constructor

```python
Tokenizer(model_path: str = "data/tokenizer.json")
```

**Parameters:**
- `model_path` (str): Path to SentencePiece model file

#### Methods

```python
def encode(self, text: str) -> List[int]:
    """Convert text to token IDs
    
    Args:
        text: Input text
        
    Returns:
        List of token IDs
    """

def decode(self, tokens: List[int]) -> str:
    """Convert token IDs back to text
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Decoded text
    """

def get_vocab_size(self) -> int:
    """Get vocabulary size"""

@staticmethod
def train(text_file: str, vocab_size: int = 4096, 
          output_path: str = "tokenizer.json"):
    """Train tokenizer on text
    
    Args:
        text_file: Path to training text
        vocab_size: Number of tokens in vocabulary
        output_path: Where to save model
    """
```

#### Example

```python
from gpt_lib.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer("data/tokenizer.json")

# Encode
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [123, 456, 789, ...]

# Decode
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Train new tokenizer
Tokenizer.train(
    "data/large_text.txt",
    vocab_size=8192,
    output_path="new_tokenizer.json"
)
```

---

## Utilities

### Pipeline

**File:** `gpt_lib/pipeline.py`

Convenience class for model building and management.

#### Constructor

```python
Pipeline()
```

#### Methods

```python
def build_model_and_trainer(self, model_type: str = "MiniGPT",
                           device: str = 'cuda') -> tuple:
    """Build model and trainer
    
    Args:
        model_type: "MiniGPT", "MediumGPT", or "MediumGPT-T"
        device: 'cpu' or 'cuda'
    
    Returns:
        (model, trainer)
    """

def save_model(self, model, path: str):
    """Save model with metadata"""

def load_model(self, path: str) -> MiniGPT:
    """Load saved model"""

def check_compatibility(self, checkpoint_path: str) -> bool:
    """Check if checkpoint is compatible"""
```

#### Example

```python
from gpt_lib.pipeline import Pipeline

pipeline = Pipeline()

# Build model
model, trainer = pipeline.build_model_and_trainer("MediumGPT")

# Train and save
train_losses, val_losses = trainer.train(
    train_loader, val_loader, epochs=10,
    checkpoint_path="model.pth"
)

# Load model
loaded_model = pipeline.load_model("model.pth")
```

---

## Configuration

### Config Class

**File:** `gpt_lib/config.py`

Central configuration object.

#### Attributes

```python
# Model architecture
embed_dim: int = 256                  # Embedding dimension
num_heads: int = 4                    # Attention heads
num_blocks: int = 4                   # Transformer blocks
vocab_size: int = 4096                # Vocabulary size
block_size: int = 128                 # Context length

# Training
learning_rate: float = 1e-3           # Adam LR
weight_decay: float = 1e-4            # L2 regularization
dropout: float = 0.1                  # Dropout rate
batch_size: int = 32                  # Batch size
num_epochs: int = 20                  # Training epochs

# Generation
temperature: float = 0.7              # Sampling temperature
top_k: int = 40                       # Top-k sampling
top_p: float = 0.9                    # Top-p (nucleus) sampling
max_new_tokens: int = 100             # Max generation length

# Device
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### Methods

```python
def __init__(self, **kwargs):
    """Initialize with optional parameter overrides"""
    
def to_dict(self) -> dict:
    """Convert config to dictionary"""

@classmethod
def from_dict(cls, config_dict: dict) -> 'Config':
    """Create config from dictionary"""
```

#### Example

```python
from gpt_lib.config import Config

# Use defaults
config = Config()

# Custom config
config = Config(
    embed_dim=512,
    num_heads=8,
    num_blocks=8,
    learning_rate=5e-4,
    batch_size=64
)

# Save/load
config_dict = config.to_dict()
loaded_config = Config.from_dict(config_dict)
```

---

## Evaluation & Diagnostics

### Diagnostics

**File:** `evaluation/diagnostic.py`

Tools for model evaluation.

#### Functions

```python
def calculate_perplexity(model, data_loader) -> float:
    """Calculate perplexity on data
    
    Lower is better. Perplexity = exp(average_loss)
    """

def calculate_accuracy(model, data_loader) -> float:
    """Calculate token-level accuracy
    
    Returns percentage of correctly predicted next tokens
    """

def evaluate_long_context(model, test_sequences: List[str],
                         context_lengths: List[int]) -> dict:
    """Evaluate performance on different sequence lengths
    
    Returns:
        Dict mapping context_length to accuracy score
    """

def full_diagnostic(model, val_loader) -> dict:
    """Run complete diagnostic suite
    
    Returns:
        Dict with perplexity, accuracy, and other metrics
    """
```

#### Example

```python
from evaluation.diagnostic import calculate_perplexity, calculate_accuracy

# Evaluate
perplexity = calculate_perplexity(model, val_loader)
accuracy = calculate_accuracy(model, val_loader)

print(f"Perplexity: {perplexity:.2f}")
print(f"Accuracy: {accuracy:.2%}")
```

---

## Function Index

### Data Utilities (`gpt_lib/data.py`)

```python
create_instruction_following_dataset(csv_path) -> (dataset, tokenizer)
load_csv_dataset(csv_path) -> dataset
train_val_split(dataset, train_ratio=0.8) -> (train, val)
clean_text(text) -> str
create_data_loaders(train_dataset, val_dataset, batch_size) -> (train_loader, val_loader)
```

### Generator (`gpt_lib/generator.py`)

```python
generate_text(model, prompt, max_tokens, temperature, top_k, top_p) -> str
generate_batch(model, prompts, max_tokens, **kwargs) -> List[str]
```

### Regularization (`gpt_lib/regularization.py`)

```python
apply_dropout(x, dropout_rate) -> Tensor
apply_weight_decay(optimizer, weight_decay)
apply_gradient_clipping(model, max_norm)
```

---

## Error Handling

### Common Exceptions

```python
# ImportError: Module not found
# → Solution: pip install -r requirements.txt

# RuntimeError: CUDA out of memory
# → Solution: Reduce batch_size or model size

# FileNotFoundError: Checkpoint not found
# → Solution: Verify path exists and is correct

# ValueError: Incompatible checkpoint
# → Solution: Use compatible config or retrain
```

---

## Performance Tips

1. **Use GPU**: 10-50x speedup for training/generation
2. **Batch processing**: Larger batches = faster training
3. **Gradient checkpointing**: Reduces memory for large models
4. **Mixed precision**: Use `torch.cuda.amp` for faster training
5. **Model caching**: Load model once, reuse for multiple inferences

---

**Last Updated:** June 20, 2026  
**Version:** 1.0
