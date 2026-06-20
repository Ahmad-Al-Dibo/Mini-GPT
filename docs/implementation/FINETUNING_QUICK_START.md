# 🚀 FINETUNING QUICK START GUIDE

**Ready to finetune your MiniGPT model? Start here!**

---

## ⚡ 30-Second Quick Start

```python
from src.miniGPT import load_model
from src.miniGPT.config import get_finetuning_config
from src.miniGPT.pipeline import build_trainer

# Load pretrained model
model = load_model("models/MiniGPT.pth")

# Create finetuning config (backbone frozen, discriminative LR)
config = get_finetuning_config()

# Build trainer
trainer = build_trainer(model, config)

# Finetune on your data
trainer.train(train_loader, val_loader=val_loader)

# Save finetuned model
trainer.save("models/MiniGPT_finetuned.pth")
```

That's it! Your model is finetuned.

---

## 📖 Common Use Cases

### 1️⃣ Quick Finetuning (Minimal Code)
```python
from src.miniGPT import load_model, prepare_data
from src.miniGPT.config import get_finetuning_config
from src.miniGPT.pipeline import build_trainer

# Finetuning with defaults
config = get_finetuning_config()  # 3 epochs, 2e-5 LR, frozen backbone
config.data_path = "data/my_domain_data.txt"

model = load_model("models/MiniGPT.pth")
trainer = build_trainer(model, config)

data = prepare_data(config)
trainer.train(data.train_loader, epochs=3, val_loader=data.val_loader)
```

### 2️⃣ Custom Parameters
```python
# Custom finetuning settings
config = get_finetuning_config(
    num_epochs=5,           # More training
    batch_size=16,          # Smaller batch for limited data
    learning_rate=5e-5      # Slightly higher LR
)

trainer = build_trainer(model, config)
trainer.train(train_loader, val_loader=val_loader)
```

### 3️⃣ Manual Layer Freezing
```python
trainer = build_trainer(model, config)

# Freeze specific layers
trainer.freeze_layers("embedding")  # Freeze token embeddings
trainer.freeze_layers("blocks.0")   # Freeze first block
trainer.freeze_layers("blocks.1")   # Freeze second block

# Check what's frozen
info = trainer.get_frozen_layers_info()
print(f"Trainable: {info['trainable_pct']:.1f}%")

# Train
trainer.train(train_loader, val_loader=val_loader)
```

### 4️⃣ Instruction Following Model
```python
# Specific config for instruction tuning
config = get_instruction_tuning_config()  # 5 epochs, 5e-5 LR
config.data_path = "data/instructions.jsonl"

trainer = build_trainer(model, config)
trainer.train(train_loader, val_loader=val_loader)
```

### 5️⃣ Different LR for Different Layers
```python
config = get_finetuning_config(use_discriminative_lr=True)

# Custom LR multipliers (default shown)
config.lr_multiplier = {
    'embeddings': 0.01,    # Very slow learning for embeddings
    'blocks': 0.1,         # Slow learning for blocks
    'head': 1.0            # Fast learning for head
}

trainer = build_trainer(model, config)  # Prints LR for each group
trainer.train(train_loader, val_loader=val_loader)
```

---

## 🎯 Layer Freezing Strategies

### Strategy 1: Freeze Everything, Train Head Only
**Use when**: Very small new dataset (< 1000 examples)
```python
trainer = build_trainer(model, config)
trainer.freeze_layers("embedding")
trainer.freeze_layers("blocks")
# Only head is trainable now
trainer.train(train_loader)
```

### Strategy 2: Freeze Embeddings & Early Blocks
**Use when**: Small dataset, want some adaptation
```python
trainer.freeze_layers("embedding")
trainer.freeze_layers("blocks.0")
trainer.freeze_layers("blocks.1")
# Only later blocks and head trainable
trainer.train(train_loader)
```

### Strategy 3: Freeze Everything, Lower LR for All
**Use when**: Medium dataset, want careful adaptation
```python
config = get_finetuning_config(use_discriminative_lr=True)
# All layers trainable but with low LR
trainer = build_trainer(model, config)
trainer.train(train_loader)
```

### Strategy 4: Fine-Grained Control
**Use when**: You know exactly what you want to finetune
```python
trainer = build_trainer(model, config)

# Check what's frozen
info = trainer.get_frozen_layers_info()
print(info)  # {'frozen': 28, 'trainable': 6, 'total': 34, 'trainable_pct': 17.6}

# Adjust based on results
if info['trainable_pct'] < 10:
    trainer.unfreeze_layers("blocks.2")  # Unfreeze specific layer
    trainer.unfreeze_layers("blocks.3")

trainer.train(train_loader)
```

---

## 🔧 Configuration Reference

### Finetuning Config Parameters
```python
config = get_finetuning_config(
    # Training
    num_epochs=3,                    # Number of epochs
    batch_size=32,                   # Batch size
    learning_rate=2e-5,              # Learning rate
    
    # Layer control
    freeze_backbone=True,            # Freeze transformer blocks
    use_discriminative_lr=True,      # Use different LR per layer type
    
    # Early stopping
    early_stopping_patience=3,       # Stop if no improvement for N epochs
    restore_best_model=True          # Restore best model after training
)
```

### Available Attributes
```python
# After creating config, you can access:
config.epochs                        # Number of epochs
config.batch_size                    # Batch size
config.learning_rate                 # Base learning rate
config.freeze_backbone               # Whether backbone frozen
config.use_discriminative_lr         # Whether using discriminative LR
config.lr_multiplier                 # Dict of LR multipliers
config.early_stopping_patience       # Patience for early stopping
config.weight_decay                  # L2 regularization
config.dropout                       # Dropout rate
config.device                        # CPU or CUDA
# ... and 40+ more parameters
```

---

## 📊 Check Layer Freezing Status

```python
info = trainer.get_frozen_layers_info()

print(f"Frozen layers: {info['frozen']}")           # Number frozen
print(f"Trainable layers: {info['trainable']}")     # Number trainable
print(f"Total layers: {info['total']}")             # Total parameters
print(f"Trainable %: {info['trainable_pct']:.1f}%") # Percentage trainable
```

---

## ⚙️ Advanced: Custom LR Multipliers

```python
from src.miniGPT.pipeline import build_optimizer_with_discriminative_lr

# Create custom LR multipliers
lr_multiplier = {
    'embeddings': 0.01,   # Embeddings: 1% of base LR
    'blocks': 0.05,       # Blocks: 5% of base LR
    'head': 1.0           # Head: 100% of base LR
}

# Create optimizer directly
optimizer = build_optimizer_with_discriminative_lr(
    model=model,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_multiplier=lr_multiplier
)

# Use with trainer
trainer = build_trainer(model, config)
trainer.optimizer = optimizer
trainer.train(train_loader)
```

---

## ❓ FAQ

**Q: How many epochs for finetuning?**
A: Usually 3-5 epochs. Less than pretraining since you're just adapting.

**Q: Should I freeze the backbone?**
A: Yes, unless you have a large new dataset (>100k examples).

**Q: What learning rate should I use?**
A: 2e-5 to 5e-5 for finetuning (much lower than pretraining).

**Q: What's discriminative learning rates?**
A: Different layers get different learning rates. Lower for features (embeddings), higher for new layers (head).

**Q: Can I change learning rate during training?**
A: Yes, we have a scheduler ready (Phase 1.5). Use `config.use_lr_scheduler = True`.

**Q: How do I save the finetuned model?**
A: `trainer.save("path/to/model.pth")`

**Q: Can I continue training a finetuned model?**
A: Yes, load it and finetune again on different data.

---

## 🎓 Learning Resources

### Example Files
- `examples/example_finetuning.py` - Complete finetuning example
- `test_phase1_implementation.py` - Test suite showing all features
- `examples/example1_generation.py` - Pretraining baseline (for comparison)

### Documentation Files
- `docs/PHASE_1_COMPLETE.md` - Detailed completion guide
- `docs/PHASE_1_IMPLEMENTATION_REPORT.md` - Full implementation report
- `IMPLEMENTATION_ROADMAP.md` - Technical details (Sections 1.1-1.4)
- `MINIGPT_TECHNICAL_ANALYSIS.md` - Code-level analysis

### Key Methods
- `trainer.freeze_layers(pattern)` - Freeze layers by name pattern
- `trainer.unfreeze_layers(pattern)` - Unfreeze layers
- `trainer.get_frozen_layers_info()` - Get layer statistics
- `get_finetuning_config()` - Create finetuning config
- `get_instruction_tuning_config()` - Create instruction tuning config
- `build_trainer(model, config)` - Create trainer with discriminative LR support

---

## 🚀 Next Steps

### Immediate
- [ ] Run `examples/example_finetuning.py` to see finetuning in action
- [ ] Try `get_finetuning_config()` on your own data
- [ ] Experiment with `freeze_layers()` to see the effect

### Short Term (Phase 2)
- [ ] Implement instruction tuning (Phase 2: 8 hours)
- [ ] Create custom instruction datasets
- [ ] Finetune on instruction-following tasks

### Medium Term (Phase 3)
- [ ] Integrate learning rate scheduler
- [ ] Add regularization techniques
- [ ] Optimize for production deployment

---

**Questions?** Check:
1. docs/PHASE_1_IMPLEMENTATION_REPORT.md (comprehensive)
2. examples/example_finetuning.py (working code)
3. test_phase1_implementation.py (test examples)
4. Docstrings in src/miniGPT/*.py files

**Ready to finetune?** Start with the 30-second quick start above! 🚀
