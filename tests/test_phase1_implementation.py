"""Test Phase 1 Finetuning Implementation"""
from src.miniGPT.config import get_finetuning_config, get_instruction_tuning_config
from src.miniGPT.trainer import Trainer
from src.miniGPT.model import MiniGPT
import torch

print("\n" + "="*70)
print("TESTING PHASE 1: FINETUNING ESSENTIALS IMPLEMENTATION")
print("="*70 + "\n")

# Test 1: Finetuning config
print("[TEST 1] Finetuning config creation...")
config = get_finetuning_config(num_epochs=3, batch_size=32)
assert config.epochs == 3
assert config.batch_size == 32
assert config.freeze_backbone == True
assert config.use_discriminative_lr == True
print("✓ Finetuning config created successfully")
print(f"  - Epochs: {config.epochs}")
print(f"  - Batch size: {config.batch_size}")
print(f"  - Freeze backbone: {config.freeze_backbone}")
print(f"  - Use discriminative LR: {config.use_discriminative_lr}\n")

# Test 2: Instruction tuning config
print("[TEST 2] Instruction tuning config creation...")
config2 = get_instruction_tuning_config()
assert config2.epochs == 5
assert config2.batch_size == 16
print("✓ Instruction tuning config created successfully")
print(f"  - Epochs: {config2.epochs}")
print(f"  - Batch size: {config2.batch_size}\n")

# Test 3: Trainer with layer freezing
print("[TEST 3] Trainer layer freezing...")
model = MiniGPT(vocab_size=100, embed_dim=64, block_size=8, num_blocks=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion, config)

# Freeze blocks
frozen_count = trainer.freeze_layers("blocks", verbose=False)
assert frozen_count > 0, "No layers were frozen"
print(f"✓ Froze {frozen_count} parameters")

# Check frozen layer info
info = trainer.get_frozen_layers_info()
assert info["frozen"] > 0
assert info["trainable"] > 0
print(f"✓ Layer info correct: {info['trainable']} trainable, {info['frozen']} frozen\n")

# Test 4: Unfreeze layers
print("[TEST 4] Unfreezing layers...")
unfrozen_count = trainer.unfreeze_layers(None, verbose=False)
assert unfrozen_count > 0
print(f"✓ Unfroze {unfrozen_count} parameters")

info = trainer.get_frozen_layers_info()
assert info["frozen"] == 0, "All parameters should be trainable now"
print(f"✓ All layers now trainable: {info['trainable']} trainable\n")

# Test 5: Discriminative learning rates
print("[TEST 5] Discriminative learning rates...")
from src.miniGPT.pipeline import build_optimizer_with_discriminative_lr

config3 = get_finetuning_config(use_discriminative_lr=True)
config3.lr_multiplier = {
    'embeddings': 0.1,
    'blocks': 0.1,
    'head': 1.0
}

optimizer = build_optimizer_with_discriminative_lr(
    model,
    config3.learning_rate,
    config3.weight_decay,
    config3.lr_multiplier
)

# Check param groups
assert len(optimizer.param_groups) > 1, "Should have multiple param groups"
print(f"✓ Created optimizer with {len(optimizer.param_groups)} param groups")
for i, group in enumerate(optimizer.param_groups):
    print(f"  Group {i+1}: LR = {group['lr']:.2e}, Params = {len(group['params'])}\n")

print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n🎉 Phase 1: Finetuning Essentials - SUCCESSFULLY IMPLEMENTED!\n")
print("Summary of implemented features:")
print("  ✓ Layer freezing/unfreezing")
print("  ✓ Get frozen layers info")
print("  ✓ Discriminative learning rates")
print("  ✓ Finetuning config presets")
print("  ✓ Instruction tuning config presets\n")
