"""
Quick MiniGPT Trainer - Fast Introduction

A small, fast trainer for quick demonstrations and introduction to the project.
Trains on data.txt only with minimal configuration.

Usage:
    python train_quick.py

Time to train: ~2-5 minutes 
"""

import time
import torch

from src.miniGPT import (
    Config,
    Generator,
    build_model,
    build_trainer,
    prepare_data,
    format_duration,
)


def create_quick_config() -> Config:
    
    return Config(
        # ========== SMALL MODEL ==========
        embed_dim=64,          
        num_blocks=2,          
        num_heads=2,           
        dropout=0.0,           
        block_size=64,         
        
        # ========== FAST TRAINING ==========
        batch_size=64,          
        epochs=3,               
        learning_rate=1e-3,     
        weight_decay=0.0,       
        grad_clip=1.0,
        
        # ========== DATA ==========
        data_path="data/data.txt",
        max_data_size=100_000,  
        validation_split=0.1,
        
        # ========== TOKENIZER ==========
        tokenizer_type="word",  # or "sentencepiece"
        max_vocab=5000,
        
        # ========== LOGGING ==========
        early_stopping_patience=None,  
        restore_best_model=False,
        training_log_interval=10,      
        
        # ========== DEVICE ==========
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path="models/MiniGPT_quick.pth",
    )


def main():
    start = time.time()
    config = create_quick_config()
    torch.manual_seed(config.seed)
    
    print("\n" + "=" * 70)
    print(" MiniGPT Quick Trainer")
    print("=" * 70)
    
    print(f"\n Device: {config.device}")
    
    print("\n Loading data.txt...")
    data = prepare_data(config)
    print(f"  ✓ Tokens loaded: {len(data.tokens):,}")
    print(f"  ✓ Train set: {len(data.train_tokens):,} tokens")
    print(f"  ✓ Vocab size: {data.vocab_size}")
    print(f"  ✓ Batches: {len(data.train_loader)}")

    print("\n Building tiny model...")
    model = build_model(config, data.vocab_size)
    params = model.get_num_parameters()
    print(f"  ✓ Parameters: {params:,}")
    print(f"  ✓ Model size: ~{params * 4 / 1024 / 1024:.1f}MB (fp32)")
    
    print("\n Creating trainer...")
    trainer = build_trainer(model, config)
    print(f"  ✓ Optimizer: AdamW (LR={config.learning_rate:.2e})")
    
    print("\n" + "=" * 70)
    print(" Training (this should be fast!)...")
    print("=" * 70 + "\n")
    
    trainer.train(
        data.train_loader,
        config.epochs,
        val_loader=data.val_loader,
    )
    
    print("\n Saving model...")
    trainer.save(config.model_path)
    print(f"  ✓ Saved to: {config.model_path}")
    
    print("\n Generating sample...")
    gen_start = time.time()
    
    generator = Generator(
        model,
        data.tokenizer.stoi if hasattr(data.tokenizer, 'stoi') else None,
        data.tokenizer.itos if hasattr(data.tokenizer, 'itos') else None,
        config.block_size,
        torch.device(config.device),
    )
    
    text = generator.generate_string(
        "Programming",
        max_new_tokens=50,
        temperature=0.8,
    )
    gen_time = time.time() - gen_start
    
    print(f"\n✨ Generated ({gen_time:.2f}s):")
    print("-" * 70)
    print(text)
    print("-" * 70)
    
    # ========== STATS ==========
    total_time = time.time() - start
    
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total time: {format_duration(total_time)}")
    print(f"Model params: {params:,}")
    print(f"Saved to: {config.model_path}")


if __name__ == "__main__":
    main()
