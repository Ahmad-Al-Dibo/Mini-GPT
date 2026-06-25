"""
MiniGPT Training Script

Train or finetune the MiniGPT model with configurable parameters.
Updated for Phase 1: Supports layer freezing and discriminative learning rates.

Usage:
    python train.py
"""

import time
import torch
from collections import Counter

from src.miniGPT import (
    Config,
    Generator,
    build_model,
    build_trainer,
    prepare_data,
    format_duration,
)


def create_config() -> Config:
    """Create training configuration."""
    
    return Config(
        # ========== Model Architecture ==========
        embed_dim=182,
        num_blocks=3,
        num_heads=4,
        dropout=0.1,
        block_size=128,
        
        # ========== Training Hyperparameters ==========
        batch_size=32,
        epochs=10,
        learning_rate=2.5e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        
        # ========== Data ==========
        data_path="data/data.txt",
        max_data_size=1_000_000,
        validation_split=0.1,
        
        # ========== Tokenizer ==========
        tokenizer_type="sentencepiece",
        sentencepiece_model_type="bpe",
        sentencepiece_character_coverage=1.0,
        max_vocab=10000,
        tokenizer_rare_threshold=2,
        
        # ========== Training Callbacks ==========
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
        training_log_interval=50,
        
        # ========== Device & Misc ==========
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path="models/MiniGPT.pth",
    )


def main():
    """Train the MiniGPT model."""
    main_start = time.time()
    config = create_config()
    torch.manual_seed(config.seed)
    
    # ========== HEADER ==========
    print("\n" + "=" * 75)
    print("🚀 MiniGPT Training Script")
    print("=" * 75)
    print("\n📊 Configuration:")
    print(config)
    
    # ========== DATA PREPARATION ==========
    print("\n📁 Preparing data...")
    data = prepare_data(config)
    print(f"✓ Data tokens: {len(data.tokens):,}")
    print(f"✓ Train tokens: {len(data.train_tokens):,}")
    print(f"✓ Validation tokens: {len(data.val_tokens):,}")
    print(f"✓ Vocab size: {data.vocab_size}")
    print(f"✓ Train batches: {len(data.train_loader)}")
    if data.val_loader is not None:
        print(f"✓ Validation batches: {len(data.val_loader)}")
    
    # Data analysis
    print(f"\n Data sample analysis:")
    print(f"  Raw tokens (first 20): {data.train_encoded[:20]}")
    print(f"  Decoded: {data.tokenizer.decode(data.train_encoded[:20])}")
    counter = Counter(data.tokenizer.decode(data.train_encoded))
    print(f"  Unique tokens: {len(counter)}")
    print(f"  Top 5 tokens: {counter.most_common(5)}")
    
    # ========== MODEL CREATION ==========
    print("\n Building model...")
    model = build_model(config, data.vocab_size)
    params = model.get_num_parameters()
    print(f"✓ Model parameters: {params:,}")
    
    # ========== USER CONFIRMATION ==========
    print("\n" + "=" * 75)
    choice = input("Continue with training? (press Enter to continue, 'q' to exit): ").lower()
    if choice == "q":
        print("Exiting...")
        return
    
    # ========== TRAINER SETUP ==========
    print("\n Setting up trainer...")
    trainer = build_trainer(model, config)
    print("✓ Trainer created")
    print(f"  - Optimizer: AdamW")
    print(f"  - Learning rate: {config.learning_rate:.2e}")
    print(f"  - Weight decay: {config.weight_decay}")
    print(f"  - Device: {config.device}")
    
    # ========== TRAINING ==========
    print("\n" + "=" * 75)
    print("🎓 Starting training...")
    print("=" * 75 + "\n")
    
    trainer.train(
        data.train_loader,
        config.epochs,
        val_loader=data.val_loader,
        early_stopping_patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    
    # ========== SAVE MODEL ==========
    print("\n💾 Saving trained model...")
    trainer.save(config.model_path)
    print(f"✓ Model saved to: {config.model_path}")
    
    # ========== GENERATE SAMPLE ==========
    print("\n🎲 Generating sample text...")
    gen_start = time.time()
    
    generator = Generator(
        model,
        data.tokenizer.stoi if hasattr(data.tokenizer, 'stoi') else None,
        data.tokenizer.itos if hasattr(data.tokenizer, 'itos') else None,
        config.block_size,
        torch.device(config.device),
    )
    
    generated_text = generator.generate_string(
        "Programming is",
        max_new_tokens=100,
        temperature=0.9,
        repetition_penalty=1.2,
        top_p=0.9,
    )
    gen_time = time.time() - gen_start
    
    print(f"\n✨ Generated text ({gen_time:.2f}s):")
    print("-" * 75)
    print(generated_text)
    print("-" * 75)
    
    # ========== SUMMARY ==========
    total_time = time.time() - main_start
    
    print("\n" + "=" * 75)
    print("📈 Training Summary")
    print("=" * 75)
    print(f"Total execution time: {format_duration(total_time)}")
    print(f"Model parameters: {params:,}")
    print(f"Final model saved to: {config.model_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
