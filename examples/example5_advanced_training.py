"""
Example 5: Advanced Training with Generalization
Volledig training voorbeeld met validatie, early stopping, en regularisatie
"""

import time
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_lib import (
    Config, Tokenizer, create_dataloader, 
    MiniGPT, Trainer, L2Regularization, 
    EarlyStopping, GeneralizationMonitor, LearningRateScheduler
)


def train_with_generalization(data_path):
    """Train model met generalisatie monitoring"""
    
    print("="*60)
    print("Advanced Training with Generalization Features")
    print("="*60)
    
    config = Config(
        embed_dim=64,
        block_size=8,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        num_blocks=2,
        model_path="output/model_advanced.pth",
        data_path=data_path,
        max_vocab=50000,
        max_data_size=1000000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nConfig:")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.embed_dim}D, {config.num_blocks} blocks")
    
    # ====================================================
    # DATA LOADING
    # ====================================================
    
    print("\n[1/6] Loading data...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    tokens = text.split()
    sample_size = min(config.max_data_size, len(tokens))
    tokens = tokens[:sample_size]
    text = " ".join(tokens)
    
    # Split into train/val (80/20)
    train_tokens = tokens[:int(len(tokens) * 0.8)]
    val_tokens = tokens[int(len(tokens) * 0.8):]
    
    print(f"  Train samples: {len(train_tokens)}")
    print(f"  Val samples: {len(val_tokens)}")
    
    # ====================================================
    # TOKENIZER
    # ====================================================
    
    print("\n[2/6] Building tokenizer...")
    tokenizer = Tokenizer(max_vocab=config.max_vocab)
    tokenizer.build(text)
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size}")
    
    encoded_train = tokenizer.encode(train_tokens)
    encoded_val = tokenizer.encode(val_tokens)
    
    # ====================================================
    # DATALOADERS
    # ====================================================
    
    print("\n[3/6] Creating dataloaders...")
    train_loader = create_dataloader(
        encoded_train,
        config.block_size,
        config.batch_size
    )
    val_loader = create_dataloader(
        encoded_val,
        config.block_size,
        config.batch_size,
        shuffle=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ====================================================
    # MODEL
    # ====================================================
    
    print("\n[4/6] Creating model...")
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        block_size=config.block_size,
        num_blocks=config.num_blocks
    )
    model = model.to(torch.device(config.device))
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # ====================================================
    # REGULARIZATION
    # ====================================================
    
    print("\n[5/6] Setting up regularization...")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5  # L2 regularization via weight decay
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # L2 Regularization
    l2_reg = L2Regularization(lambda_l2=1e-4)
    
    # Early stopping
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
    
    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        optimizer,
        strategy="cosine",
        total_epochs=config.epochs
    )
    
    # Generalization monitor
    gen_monitor = GeneralizationMonitor()
    
    print("  [OK] L2 Regularization (lambda=1e-4)")
    print("  [OK] Early Stopping (patience=15)")
    print("  [OK] Learning Rate Scheduler (cosine)")
    print("  [OK] Generalization Monitor")
    
    # ====================================================
    # TRAINER
    # ====================================================
    
    print("\n[6/6] Training...")
    trainer = Trainer(model, optimizer, criterion, config)
    
    model.train()
    train_start = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        total_train_loss = 0
        
        # Training loop
        for x, y in train_loader:
            x = x.to(torch.device(config.device))
            y = y.to(torch.device(config.device))
            
            logits = model(x)
            B, T, V = logits.shape
            
            # Cross-entropy loss
            ce_loss = criterion(
                logits.view(B * T, V),
                y.view(B * T)
            )
            
            # Add regularization
            reg_loss = l2_reg.compute_loss(model)
            total_loss = ce_loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += ce_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(torch.device(config.device))
                y = y.to(torch.device(config.device))
                
                logits = model(x)
                B, T, V = logits.shape
                
                loss = criterion(
                    logits.view(B * T, V),
                    y.view(B * T)
                )
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update monitors
        gen_monitor.update(avg_train_loss, avg_val_loss)
        
        # Update learning rate
        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print stats
        gen_gap = gen_monitor.get_generalization_gap()
        epoch_time = time.time() - epoch_start
        
        status = "[WARNING]  OVERFITTING" if gen_monitor.is_overfitting(threshold=1.0) else "[OK] OK"
        
        print(
            f"Epoch {epoch+1:3d}/{config.epochs}: "
            f"Train={avg_train_loss:.4f} Val={avg_val_loss:.4f} "
            f"Gap={gen_gap:.4f} LR={current_lr:.2e} {status} [{epoch_time:.1f}s]"
        )
        
        # Check early stopping
        model.train()
        if early_stopper.check(avg_val_loss):
            print(f"\n[WARNING]  Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {early_stopper.best_loss:.4f}")
            break
    
    total_train_time = time.time() - train_start
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    
    print(f"\n[OK] Training completed in {hours}h {minutes}m {seconds}s")
    
    # ====================================================
    # SAVE & SUMMARY
    # ====================================================
    
    print("\nSaving model...")
    trainer.save(
        config.model_path,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        block_size=config.block_size,
        vocab_size=vocab_size
    )
    
    # Final report
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(gen_monitor.get_report())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example5_advanced_training.py <data_path>")
        print("Example: python example5_advanced_training.py data/data.txt")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if not Path(data_path).exists():
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)
    
    train_with_generalization(data_path)
