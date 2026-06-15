"""
Optimization Pipeline - Improve Generalization
Step 1: Kernel optimization (bigger/better model)
Step 2: Advanced regularization
Step 3: Data augmentation
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from gpt_lib import (
    Config, Tokenizer, create_dataloader,
    MiniGPT, Trainer, Generator,
    L1Regularization, L2Regularization,
    EarlyStopping, LearningRateScheduler,
    GeneralizationMonitor, MixupAugmentation,
    LabelSmoothing
)


def optimize_kernel_size(data_path):
    """Test different model sizes to find sweet spot"""
    print("\n" + "="*60)
    print(" KERNEL SIZE OPTIMIZATION")
    print("="*60)
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split = int(len(text) * 0.8)
    train_text, val_text = text[:split], text[split:]
    
    tokenizer = Tokenizer()
    tokenizer.build(text)
    
    train_tokens = train_text.split()
    val_tokens = val_text.split()
    train_encoded = tokenizer.encode(train_tokens)[:500000]
    val_encoded = tokenizer.encode(val_tokens)[:100000]
    
    train_loader = create_dataloader(train_encoded, 8, 64)
    val_loader = create_dataloader(val_encoded, 8, 64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different architectures
    architectures = [
        {"name": "Small", "embed_dim": 32, "num_blocks": 1},
        {"name": "Medium", "embed_dim": 64, "num_blocks": 2},
        {"name": "Large", "embed_dim": 128, "num_blocks": 3},
        {"name": "XLarge", "embed_dim": 256, "num_blocks": 4},
    ]
    
    print("\nTesting different model sizes (5 epochs each):\n")
    print(f"{'Model':<15} {'Params':<12} {'Final Train':<12} {'Final Val':<12} {'Gap':<10}")
    print("-"*60)
    
    results = []
    
    for arch in architectures:
        model = MiniGPT(tokenizer.get_vocab_size(), arch["embed_dim"], 8, arch["num_blocks"])
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        config = Config(epochs=5, embed_dim=arch["embed_dim"])
        
        trainer = Trainer(model, optimizer, criterion, config)
        trainer.train(train_loader, 5, val_loader)
        
        history = trainer.get_train_history()
        final_train = history['train_losses'][-1]
        final_val = history['val_losses'][-1]
        gap = final_val - final_train
        
        print(f"{arch['name']:<15} {num_params:<12,d} {final_train:<12.4f} {final_val:<12.4f} {gap:<10.4f}")
        
        results.append({
            "name": arch["name"],
            "params": num_params,
            "train_loss": final_train,
            "val_loss": final_val,
            "gap": gap
        })
    
    # Recommend
    best = min(results, key=lambda x: x["gap"])
    print(f"\n[OK] RECOMMENDATION: {best['name']} model (Gap={best['gap']:.4f})")
    
    return best


def advanced_training_strategy(data_path, embed_dim=128, num_blocks=3):
    """Implement advanced training with multiple regularization techniques"""
    print("\n" + "="*60)
    print(" ADVANCED TRAINING STRATEGY")
    print("="*60)
    
    print(f"\nUsing: {embed_dim}D embedding, {num_blocks} blocks")
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split = int(len(text) * 0.8)
    train_text, val_text = text[:split], text[split:]
    
    tokenizer = Tokenizer()
    tokenizer.build(text)
    
    train_tokens = train_text.split()
    val_tokens = val_text.split()
    train_encoded = tokenizer.encode(train_tokens)[:500000]
    val_encoded = tokenizer.encode(val_tokens)[:100000]
    
    train_loader = create_dataloader(train_encoded, 8, 64)
    val_loader = create_dataloader(val_encoded, 8, 64)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(tokenizer.get_vocab_size(), embed_dim, 8, num_blocks)
    model = model.to(device)
    
    # Optimizers & regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Different loss functions to try
    criterion_configs = [
        {"name": "Standard CE", "criterion": nn.CrossEntropyLoss(), "l2_lambda": 0},
        {"name": "Label Smoothing", "criterion": LabelSmoothing(tokenizer.get_vocab_size(), smoothing=0.1), "l2_lambda": 1e-4},
    ]
    
    print("\nComparing different regularization strategies (10 epochs each):\n")
    
    for config_item in criterion_configs:
        model_copy = MiniGPT(tokenizer.get_vocab_size(), embed_dim, 8, num_blocks)
        model_copy = model_copy.to(device)
        
        optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = config_item["criterion"]
        
        config = Config(epochs=10)
        scheduler = LearningRateScheduler(optimizer_copy, "cosine", 10)
        early_stopper = EarlyStopping(patience=10)
        monitor = GeneralizationMonitor()
        
        print(f"\n{config_item['name']}:")
        print(f"{'Epoch':<8} {'Train':<12} {'Val':<12} {'Gap':<10} {'LR':<12}")
        print("-"*54)
        
        for epoch in range(10):
            # Train
            model_copy.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                logits = model_copy(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                optimizer_copy.zero_grad()
                loss.backward()
                optimizer_copy.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Val
            model_copy.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model_copy(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            monitor.update(train_loss, val_loss)
            gap = monitor.get_generalization_gap()
            scheduler.step(epoch)
            
            current_lr = optimizer_copy.param_groups[0]['lr']
            print(f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {gap:<10.4f} {current_lr:<12.2e}")
            
            if early_stopper.check(val_loss):
                print(f"Early stop at epoch {epoch+1}")
                break
        
        print(f"\nFinal Gap: {gap:.4f}")


def compare_sampling_strategies(model_path, tokenizer_path=None):
    """Compare different sampling methods for text generation"""
    print("\n" + "="*60)
    print(" SAMPLING STRATEGY COMPARISON")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = Config(**checkpoint.get("config", {}))
    model = MiniGPT(
        checkpoint["vocab_size"],
        config.embed_dim,
        config.block_size,
        config.num_blocks
    )
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[state_key])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.stoi = checkpoint["stoi"]
    tokenizer.itos = checkpoint["itos"]
    tokenizer.vocab = checkpoint.get("vocab")
    tokenizer.vocab_size = checkpoint["vocab_size"]
    
    generator = Generator(model, tokenizer.stoi, tokenizer.itos, config.block_size, device)
    
    prompt = "machine learning is"
    
    strategies = [
        {"name": "Standard (T=1.0)", "temp": 1.0, "penalty": 1.0, "top_k": None, "top_p": None},
        {"name": "Lower Temp (T=0.7)", "temp": 0.7, "penalty": 1.0, "top_k": None, "top_p": None},
        {"name": "Higher Temp (T=1.5)", "temp": 1.5, "penalty": 1.0, "top_k": None, "top_p": None},
        {"name": "Top-K (k=50)", "temp": 1.0, "penalty": 1.0, "top_k": 50, "top_p": None},
        {"name": "Top-P (p=0.9)", "temp": 1.0, "penalty": 1.0, "top_k": None, "top_p": 0.9},
        {"name": "Repetition Penalty", "temp": 1.0, "penalty": 1.5, "top_k": None, "top_p": None},
    ]
    
    print(f"\nPrompt: '{prompt}'\n")
    
    for strategy in strategies:
        text = generator.generate_string(
            prompt,
            max_new_tokens=15,
            temperature=strategy["temp"],
            repetition_penalty=strategy["penalty"],
            top_k=strategy["top_k"],
            top_p=strategy["top_p"]
        )
        print(f"{strategy['name']:<25} -> {text}")


def run_optimization_pipeline(data_path):
    """Run complete optimization pipeline"""
    print("\n" + "#"*60)
    print("# GENERALIZATION IMPROVEMENT PIPELINE")
    print("#"*60)
    
    # Step 1: Find optimal kernel
    best_arch = optimize_kernel_size(data_path)
    
    # Step 2: Advanced training
    advanced_training_strategy(
        data_path,
        embed_dim=next(
            arch["embed_dim"] for arch in [
                {"name": "Small", "embed_dim": 32},
                {"name": "Medium", "embed_dim": 64},
                {"name": "Large", "embed_dim": 128},
                {"name": "XLarge", "embed_dim": 256},
            ]
            if arch["name"] == best_arch["name"]
        ),
        num_blocks=next(
            arch["num_blocks"] for arch in [
                {"name": "Small", "num_blocks": 1},
                {"name": "Medium", "num_blocks": 2},
                {"name": "Large", "num_blocks": 3},
                {"name": "XLarge", "num_blocks": 4},
            ]
            if arch["name"] == best_arch["name"]
        )
    )
    
    print("\n" + "="*60)
    print("[OK] OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    data_path = "data/data.txt"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    run_optimization_pipeline(data_path)
