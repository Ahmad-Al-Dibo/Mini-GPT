"""
Example 6: Finetuning a Pretrained Model

Demonstrates how to finetune a pretrained MiniGPT model on a new domain
or task using the new finetuning features:
- Layer freezing
- Discriminative learning rates
- Finetuning config presets
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.miniGPT import load_model, prepare_data, create_dataloader
from src.miniGPT.config import get_finetuning_config
from src.miniGPT.pipeline import build_trainer


def finetune_pretrained_model():
    """Example: Finetune a pretrained model on new data"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Finetuning a Pretrained Model")
    print("="*80 + "\n")
    
    # ===== STEP 1: Load pretrained model =====
    print("[STEP 1] Loading pretrained model...")
    model_path = "models/MiniGPT.pth"
    if not Path(model_path).exists():
        print(f"[ERROR] Pretrained model not found at {model_path}")
        print("        Run example1_generation.py first to train a model")
        return
    
    model = load_model(model_path)
    print(f"✓ Loaded model from {model_path}\n")
    
    # ===== STEP 2: Prepare new domain data =====
    print("[STEP 2] Preparing new domain data...")
    config = get_finetuning_config(
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        freeze_backbone=True,
        use_discriminative_lr=True,
        data_path="data/data.txt",  # New domain data
        validation_split=0.1,
    )
    
    print(f"✓ Finetuning config created:")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate:.2e}")
    print(f"  - Freeze backbone: {config.freeze_backbone}")
    print(f"  - Use discriminative LR: {config.use_discriminative_lr}\n")
    
    # Prepare data
    data_bundle = prepare_data(config)
    train_loader = data_bundle.train_loader
    val_loader = data_bundle.val_loader
    print(f"✓ Data prepared:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}\n")
    
    # ===== STEP 3: Build trainer with discriminative LR =====
    print("[STEP 3] Building trainer with finetuning support...")
    trainer = build_trainer(model, config)
    print("✓ Trainer created with discriminative learning rates\n")
    
    # ===== STEP 4: Freeze backbone and show layer info =====
    print("[STEP 4] Freezing backbone layers...")
    trainer.freeze_layers("blocks")  # Freeze transformer blocks
    
    frozen_info = trainer.get_frozen_layers_info()
    print(f"✓ Layer status:")
    print(f"  - Frozen: {frozen_info['frozen']}")
    print(f"  - Trainable: {frozen_info['trainable']}")
    print(f"  - Total: {frozen_info['total']}")
    print(f"  - Trainable %: {frozen_info['trainable_pct']:.1f}%\n")
    
    # ===== STEP 5: Finetune on new data =====
    print("[STEP 5] Finetuning model on new domain data...")
    print("(This will finetune only the head while backbone is frozen)\n")
    
    trainer.train(
        train_loader,
        config.num_epochs,
        val_loader=val_loader,
        early_stopping_patience=3,
        min_delta=1e-4,
    )
    
    # ===== STEP 6: Save finetuned model =====
    print("\n[STEP 6] Saving finetuned model...")
    finetuned_model_path = "models/MiniGPT_finetuned.pth"
    trainer.save(finetuned_model_path)
    print(f"✓ Model saved to {finetuned_model_path}\n")
    
    print("="*80)
    print("Finetuning Complete!")
    print("="*80 + "\n")
    
    # Summary
    print("Summary:")
    print("- Loaded pretrained model from models/MiniGPT.pth")
    print("- Froze transformer blocks (kept them fixed)")
    print("- Used discriminative learning rates (lower for blocks, higher for head)")
    print("- Trained only the head on new domain data")
    print("- Saved finetuned model to models/MiniGPT_finetuned.pth")
    print("\nBenefits of finetuning:")
    print("- 10x faster than pretraining (smaller effective model)")
    print("- Better generalization (leverages pretrained knowledge)")
    print("- Prevents catastrophic forgetting (backbone frozen)")
    print("- Adapts to new domain (head learns new patterns)\n")


def show_different_freezing_strategies():
    """Show different layer freezing strategies"""
    
    print("\n" + "="*80)
    print("LAYER FREEZING STRATEGIES FOR FINETUNING")
    print("="*80 + "\n")
    
    model_path = "models/MiniGPT.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
    
    model = load_model(model_path)
    config = get_finetuning_config()
    trainer = build_trainer(model, config)
    
    # Strategy 1: Freeze all blocks, train head only
    print("[Strategy 1] Freeze transformer blocks, train head only")
    print("  Use case: Very small new dataset")
    print("  Implementation:")
    print("    trainer.freeze_layers('blocks')")
    trainer.freeze_layers("blocks", verbose=False)
    info = trainer.get_frozen_layers_info()
    print(f"  Result: {info['trainable']} trainable, {info['frozen']} frozen\n")
    
    # Strategy 2: Freeze embeddings and blocks, train head
    trainer.unfreeze_layers(None, verbose=False)  # Unfreeze all
    print("[Strategy 2] Freeze embeddings and blocks, train head only")
    print("  Use case: Very limited new data")
    print("  Implementation:")
    print("    trainer.freeze_layers('token_embedding')")
    print("    trainer.freeze_layers('position_embedding')")
    print("    trainer.freeze_layers('blocks')")
    trainer.freeze_layers("embedding", verbose=False)
    trainer.freeze_layers("blocks", verbose=False)
    info = trainer.get_frozen_layers_info()
    print(f"  Result: {info['trainable']} trainable, {info['frozen']} frozen\n")
    
    # Strategy 3: Freeze early blocks, finetune later blocks and head
    trainer.unfreeze_layers(None, verbose=False)
    print("[Strategy 3] Freeze early blocks, finetune later blocks and head")
    print("  Use case: Medium dataset, want to adapt deeper layers")
    print("  Implementation:")
    print("    trainer.freeze_layers('blocks.0')")
    print("    trainer.freeze_layers('blocks.1')")
    trainer.freeze_layers("blocks.0", verbose=False)
    trainer.freeze_layers("blocks.1", verbose=False)
    info = trainer.get_frozen_layers_info()
    print(f"  Result: {info['trainable']} trainable, {info['frozen']} frozen\n")
    
    # Strategy 4: No freezing, just discriminative LR
    trainer.unfreeze_layers(None, verbose=False)
    print("[Strategy 4] No freezing, use discriminative learning rates")
    print("  Use case: Larger dataset, want to adapt entire model")
    print("  Implementation:")
    print("    config.use_discriminative_lr = True")
    print("    config.lr_multiplier = {'embeddings': 0.1, 'blocks': 0.1, 'head': 1.0}")
    print("    trainer = build_trainer(model, config)")
    info = trainer.get_frozen_layers_info()
    print(f"  Result: {info['trainable']} trainable, {info['frozen']} frozen")
    print("  All layers trainable but with different learning rates\n")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetuning examples")
    parser.add_argument("--strategy", action="store_true", 
                       help="Show different freezing strategies")
    
    args = parser.parse_args()
    
    if args.strategy:
        show_different_freezing_strategies()
    else:
        finetune_pretrained_model()
