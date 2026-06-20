# Config Directory

Configuration files for MiniGPT models and training.

## ⚙️ Configuration System

MiniGPT uses a centralized configuration system.

### Primary Config (in src/config.py)
```python
from src.config import Config

# Get default config
config = Config()

# Customize
config.embed_dim = 512
config.learning_rate = 5e-4
```

### Configuration Files

- **model_configs.py** - Pre-defined model configurations
- **training_configs.py** - Pre-defined training setups
- **README.md** - This file

## 🎯 Model Configurations

### MiniGPT (Fast, Learning)
```python
from config.model_configs import MINIGPT_CONFIG

config = MINIGPT_CONFIG()
# embed_dim=256, num_heads=4, num_blocks=4, vocab_size=4096
```

### MediumGPT (Better Quality)
```python
from config.model_configs import MEDIUMGPT_CONFIG

config = MEDIUMGPT_CONFIG()
# embed_dim=512, num_heads=8, num_blocks=8, vocab_size=4096
```

### MediumGPT-T (Instruction Tuning)
```python
from config.model_configs import MEDIUMGPT_T_CONFIG

config = MEDIUMGPT_T_CONFIG()
# Same as MediumGPT but optimized for instruction following
```

## 🎓 Training Configurations

### Quick Test
```python
from config.training_configs import QUICK_TEST_CONFIG

config = QUICK_TEST_CONFIG()
# epochs=5, learning_rate=1e-3, batch_size=32
```

### Standard Training
```python
from config.training_configs import STANDARD_CONFIG

config = STANDARD_CONFIG()
# epochs=20, learning_rate=1e-3, batch_size=32
```

### Fine-tuning
```python
from config.training_configs import FINETUNING_CONFIG

config = FINETUNING_CONFIG()
# epochs=5, learning_rate=1e-4, batch_size=16
```

### Large-scale
```python
from config.training_configs import LARGE_SCALE_CONFIG

config = LARGE_SCALE_CONFIG()
# epochs=100, learning_rate=5e-4, batch_size=64
```

## 📊 Configuration Parameters

### Model Architecture
```python
config.embed_dim = 256              # Embedding dimension
config.num_heads = 4                # Attention heads
config.num_blocks = 4               # Transformer blocks
config.vocab_size = 4096            # Vocabulary size
config.block_size = 128             # Context length
config.dropout = 0.1                # Dropout rate
```

### Training
```python
config.learning_rate = 1e-3         # Adam LR
config.weight_decay = 1e-4          # L2 regularization
config.batch_size = 32              # Batch size
config.num_epochs = 20              # Training epochs
config.early_stopping_patience = 5  # Early stopping
config.early_stopping_min_delta = 0.001
```

### Generation
```python
config.temperature = 0.7            # Sampling temperature
config.top_k = 40                   # Top-k sampling
config.top_p = 0.9                  # Top-p sampling
config.max_new_tokens = 100         # Max generation length
```

### Device
```python
config.device = 'cuda'              # 'cuda' or 'cpu'
```

## 🚀 Using Configurations

### Use pre-defined config
```python
from config.model_configs import MINIGPT_CONFIG
from src.pipeline import Pipeline

config = MINIGPT_CONFIG()
pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer(config=config)
```

### Customize config
```python
from config.model_configs import MINIGPT_CONFIG

config = MINIGPT_CONFIG()
config.embed_dim = 512              # Override
config.batch_size = 64              # Override
config.learning_rate = 5e-4         # Override

# Use custom config
pipeline.build_model_and_trainer(config=config)
```

### Create from scratch
```python
from src.config import Config

config = Config(
    embed_dim=512,
    num_heads=8,
    num_blocks=8,
    learning_rate=5e-4,
    batch_size=64
)
```

## 📝 Common Scenarios

### Scenario 1: Fast Testing
```python
from config.model_configs import MINIGPT_CONFIG
from config.training_configs import QUICK_TEST_CONFIG

model_cfg = MINIGPT_CONFIG()
training_cfg = QUICK_TEST_CONFIG()

trainer.learning_rate = training_cfg.learning_rate
trainer.train(..., epochs=training_cfg.num_epochs)
```

### Scenario 2: Production Quality
```python
from config.model_configs import MEDIUMGPT_CONFIG
from config.training_configs import LARGE_SCALE_CONFIG

model_cfg = MEDIUMGPT_CONFIG()
training_cfg = LARGE_SCALE_CONFIG()
```

### Scenario 3: Fine-tuning
```python
from config.training_configs import FINETUNING_CONFIG

cfg = FINETUNING_CONFIG()
# Lower learning rate, fewer epochs
```

## 💾 Saving Configurations

```python
# Save config
config.save("my_config.json")

# Load config
loaded_config = Config.load("my_config.json")
```

## 🔗 Related

- Main Config: `../src/config.py`
- Model Architectures: `../src/model.py`
- Training: `../docs/02_FULL_DOCUMENTATION.md#training-guide`
- API Reference: `../docs/03_API_REFERENCE.md#configuration`

## 📚 Full Configuration Reference

See: `../docs/03_API_REFERENCE.md#config-class`

---

**Start with pre-defined configs, then customize as needed!**
