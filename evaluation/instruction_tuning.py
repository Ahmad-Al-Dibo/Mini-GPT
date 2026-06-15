"""
Instruction-Following Training - Teach model to follow instructions
Uses [INST]...[/INST] format for explicit instruction marking
"""

import sys
sys.path.insert(0, '.')

import os
import torch
import torch.nn as nn
from gpt_lib import (
    Config, Tokenizer, create_dataloader,
    MiniGPT, Trainer, Generator,
    EarlyStopping, LearningRateScheduler,
    GeneralizationMonitor
)


def create_instruction_dataset(data_path, output_path="data/instruction_data.txt"):
    """Convert raw data to instruction format
    
    Format:
    [INST]What is Python?[/INST] Python is a programming language...
    [INST]Explain machine learning[/INST] Machine learning is the study of...
    """
    
    print("\n" + "="*60)
    print(" CREATING INSTRUCTION DATASET")
    print("="*60)
    
    # Read original data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    sentences = text.split('.')
    
    # Create instruction pairs
    instruction_pairs = []
    
    # Question templates
    templates = [
        ("What is", "A"),
        ("Explain", "This is"),
        ("Define", ""),
        ("How does", "It works by"),
        ("Tell me about", "Here's what I know:"),
    ]
    
    print(f"\nProcessing {len(sentences)} sentences...")
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        sentence = sentence.strip()
        
        # Extract key noun phrases
        words = sentence.split()
        if len(words) < 3:
            continue
        
        # Create instruction variations
        if any(word in sentence.lower() for word in ["python", "machine", "data", "learning", "algorithm"]):
            key_phrase = words[0] + " " + words[1] if len(words) > 1 else words[0]
            
            # Simple instruction from sentence
            instruction = f"What is {key_phrase}?"
            response = sentence + "."
            
            instruction_pairs.append(f"[INST]{instruction}[/INST] {response}\n")
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1} sentences -> {len(instruction_pairs)} instruction pairs")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(instruction_pairs)
    
    print(f"\n[OK] Created {len(instruction_pairs)} instruction pairs")
    print(f"[OK] Saved to {output_path}")
    
    return output_path


def train_instruction_model(data_path, instruction_data_path=None):
    """Train model specifically for instruction following"""
    
    print("\n" + "="*60)
    print(" INSTRUCTION-FOLLOWING TRAINING")
    print("="*60)
    
    # Create instruction dataset if needed
    if instruction_data_path is None:
        instruction_data_path = create_instruction_dataset(data_path)
    
    # Load instruction data
    with open(instruction_data_path, 'r', encoding='utf-8') as f:
        instruction_text = f.read()
    
    # Split for validation
    split = int(len(instruction_text) * 0.8)
    train_text = instruction_text[:split]
    val_text = instruction_text[split:]
    
    # Build tokenizer
    tokenizer = Tokenizer()
    tokenizer.build(instruction_text)
    
    train_tokens = train_text.split()
    val_tokens = val_text.split()
    train_encoded = tokenizer.encode(train_tokens)[:500000]
    val_encoded = tokenizer.encode(val_tokens)[:100000]
    
    train_loader = create_dataloader(train_encoded, block_size=16, batch_size=32)
    val_loader = create_dataloader(val_encoded, block_size=16, batch_size=32)
    
    print(f"\nDataset:")
    print(f"  Instruction pairs: {instruction_text.count('[INST]')}")
    print(f"  Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print(f"  Train sequences: {len(train_encoded)}")
    print(f"  Val sequences: {len(val_encoded)}")
    
    # Create model (larger for instruction following)
    config = Config(
        embed_dim=128,
        num_blocks=3,
        epochs=50,
        block_size=16
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(tokenizer.get_vocab_size(), config.embed_dim, config.block_size, config.num_blocks)
    model = model.to(device)
    
    print(f"\nModel:")
    print(f"  Architecture: {config.embed_dim}D, {config.num_blocks} blocks")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = LearningRateScheduler(optimizer, "cosine", config.epochs)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    monitor = GeneralizationMonitor()
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: 1e-3 (cosine decay)")
    print(f"  Regularization: L2 (weight_decay=1e-4)")
    print(f"  Early stopping: patience=20")
    
    # Training loop
    print(f"\n{'Epoch':<8} {'Train':<12} {'Val':<12} {'Gap':<10} {'LR':<12}")
    print("-"*54)
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        monitor.update(train_loss, val_loss)
        gap = monitor.get_generalization_gap()
        scheduler.step(epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        status = "[OK]" if gap < 1.0 else "[WARNING]"
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {gap:<10.4f} {current_lr:<12.2e} {status}")
        
        if early_stopper.check(val_loss):
            print(f"\n[OK] Early stopping at epoch {epoch+1} (no improvement for 20 epochs)")
            break
    
    # Save model
    import os
    os.makedirs("output", exist_ok=True)
    
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'embed_dim': config.embed_dim,
            'num_blocks': config.num_blocks,
            'block_size': 16
        },
        'vocab_size': tokenizer.get_vocab_size(),
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
    }, "output/model_instruction_following.pth")
    
    print(f"\n[OK] Model saved to output/model_instruction_following.pth")
    print("\n" + monitor.get_report())
    
    return model, tokenizer


def test_instruction_following(model, tokenizer):
    """Test instruction following capability"""
    
    print("\n" + "="*60)
    print("[OK] INSTRUCTION FOLLOWING TEST")
    print("="*60)
    
    device = next(model.parameters()).device
    generator = Generator(model, tokenizer.stoi, tokenizer.itos, 16, device)
    
    test_instructions = [
        "[INST]What is Python?[/INST]",
        "[INST]Explain machine learning[/INST]",
        "[INST]Define data science[/INST]",
        "[INST]How does neural network[/INST]",
    ]
    
    print("\nTesting instruction-following capability:\n")
    
    for instruction in test_instructions:
        response = generator.generate_string(instruction, max_new_tokens=20, top_p=0.9)
        print(f"Q: {instruction.replace('[INST]', '').replace('[/INST]', '')}")
        print(f"A: {response}\n")


def run_instruction_training(data_path):
    """Complete instruction-following training pipeline"""
    
    print("\n" + "#"*60)
    print("# INSTRUCTION-FOLLOWING TRAINING PIPELINE")
    print("#"*60)
    
    # Step 1: Create instruction dataset
    instruction_data_path = create_instruction_dataset(data_path)
    
    # Step 2: Train model
    model, tokenizer = train_instruction_model(data_path, instruction_data_path)
    
    # Step 3: Test
    test_instruction_following(model, tokenizer)
    
    print("\n" + "="*60)
    print("[OK] INSTRUCTION TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    data_path = "data/data.txt"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    run_instruction_training(data_path)
