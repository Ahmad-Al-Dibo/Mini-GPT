"""
Diagnostic Suite - Evaluate Model Properties (ASCII version for compatibility)
Tests: Knowledge representation, Generalization, Memorization, Instruction-following
"""

import sys
sys.path.insert(0, '.')

import os
import torch
import torch.nn as nn
from collections import Counter
from gpt_lib import (
    Config, Tokenizer, create_dataloader,
    MiniGPT, Trainer, Generator,
    GeneralizationMonitor
)


def load_text_sample(data_path, max_tokens=100000):
    """Load a bounded token sample from a text file."""
    tokens = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens.extend(line.split())
            if len(tokens) >= max_tokens:
                break
    return " ".join(tokens[:max_tokens])


def evaluate_knowledge_representation(model, tokenizer, test_prompts, config, device):
    """Test if model can answer basic questions about patterns"""
    print("\n" + "="*60)
    print("KNOWLEDGE REPRESENTATION TEST")
    print("="*60)
    
    generator = Generator(model, tokenizer.stoi, tokenizer.itos, config.block_size, device)
    
    test_cases = [
        ("python is", ["programming", "language", "code"]),
        ("machine learning is", ["ai", "algorithm", "data", "model"]),
        ("data science is", ["analysis", "statistics", "data"]),
    ]
    
    print("\nTesting conceptual understanding:\n")
    matches = 0
    for prompt, expected_keywords in test_cases:
        generated = generator.generate_string(prompt, max_new_tokens=5)
        generated_lower = generated.lower()
        matched = any(keyword in generated_lower for keyword in expected_keywords)
        matches += int(matched)
        print(f"Prompt: '{prompt}'")
        print(f"Expected keywords: {', '.join(expected_keywords)}")
        print(f"Generated: {generated}\n")

    score = matches / len(test_cases)
    print(f"Knowledge keyword score: {score*100:.1f}%")
    return score


def evaluate_generalization_gap(model, tokenizer, data_path):
    """Measure train/validation loss gap without changing model weights."""
    print("\n" + "="*60)
    print("GENERALIZATION GAP TEST")
    print("="*60)
    
    # Load and split a bounded data sample.
    text = load_text_sample(data_path)
    
    # 80/20 split
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Tokenize (use already-built tokenizer)
    train_tokens = train_text.split()
    val_tokens = val_text.split()
    train_encoded = tokenizer.encode(train_tokens)[:100000]
    val_encoded = tokenizer.encode(val_tokens)[:100000]
    
    # Create dataloaders
    block_size = getattr(model, "block_size", 8)
    train_loader = create_dataloader(train_encoded, block_size=block_size, batch_size=64, shuffle=False)
    val_loader = create_dataloader(val_encoded, block_size=block_size, batch_size=64, shuffle=False)
    
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    def evaluate_loader(loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == y).sum().item()
                total_tokens += y.numel()
        return total_loss / len(loader), total_correct / total_tokens if total_tokens else 0.0

    train_loss, train_acc = evaluate_loader(train_loader)
    val_loss, val_acc = evaluate_loader(val_loader)
    gap = val_loss - train_loss

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print(f"Gap:        {gap:.4f}")

    return gap


def evaluate_memorization_vs_learning(model, tokenizer, data_path):
    """Detect if model memorizes training data vs learns patterns"""
    print("\n" + "="*60)
    print("MEMORIZATION vs LEARNING TEST")
    print("="*60)
    
    # Load a bounded data sample.
    text = load_text_sample(data_path)
    
    # Use already-built tokenizer
    tokens = text.split()
    encoded = tokenizer.encode(tokens)[:100000]  # Sample
    
    # Get predictions on training data
    model.eval()
    accuracies = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(encoded)-8, 100):
            x = torch.tensor(encoded[i:i+8]).unsqueeze(0).to(device)
            y = torch.tensor(encoded[i+1:i+9]).unsqueeze(0).to(device)
            
            logits = model(x)
            predictions = logits.argmax(-1)
            
            acc = (predictions == y).float().mean().item()
            accuracies.append(acc)
    
    avg_acc = sum(accuracies) / len(accuracies)
    
    print(f"\nAccuracy on training data: {avg_acc*100:.2f}%")
    print(f"Token vocab size: {tokenizer.get_vocab_size()}")
    print(f"Random chance: {100/tokenizer.get_vocab_size():.2f}%")
    
    if avg_acc > 0.5:
        print("\n[WARNING] MEMORIZATION: Model shows high accuracy -> may be memorizing")
    elif avg_acc > 0.1:
        print("\n[OK] LEARNING: Model shows moderate accuracy -> learning patterns")
    else:
        print("\n[WARNING] POOR LEARNING: Model accuracy near random -> not learning")
    
    return avg_acc


def evaluate_instruction_following(model, tokenizer, generator):
    """Test if model can follow instructions"""
    print("\n" + "="*60)
    print("INSTRUCTION FOLLOWING TEST")
    print("="*60)
    
    instructions = [
        "explain what is",
        "write code for",
        "list advantages of",
        "describe how to",
        "summarize the",
    ]
    
    print("\nTesting instruction compliance:\n")
    
    for inst in instructions:
        generated = generator.generate_string(inst, max_new_tokens=5)
        print(f"Instruction: '{inst}'")
        print(f"Response:    '{generated}'")
        print()
    
    print("[WARNING] If responses are generic or not instruction-aware -> model needs instruction tuning")


def run_full_diagnostic(data_path):
    """Run complete diagnostic suite"""
    print("\n" + "#"*60)
    print("# COMPREHENSIVE MODEL DIAGNOSTIC SUITE")
    print("#"*60)
    
    # Create output directory if needed
    os.makedirs("output", exist_ok=True)
    
    config = Config()
    tokenizer = Tokenizer()

    text = load_text_sample(data_path)
    tokenizer.build(text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "output/mini_gpt_generalized.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "output/mini_gpt.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint.get("config", config.to_dict())
        config = Config(**config_dict)

        if checkpoint.get("vocab") is not None:
            tokenizer.vocab = checkpoint["vocab"]
            tokenizer.stoi = checkpoint["stoi"]
            tokenizer.itos = checkpoint["itos"]
            tokenizer.vocab_size = checkpoint["vocab_size"]

        model = MiniGPT(
            tokenizer.get_vocab_size(),
            config.embed_dim,
            config.block_size,
            config.num_blocks,
            getattr(config, "dropout", 0.0)
        ).to(device)
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
        model.load_state_dict(checkpoint[state_key])
        print(f"\n[OK] Loaded existing model from {checkpoint_path}")
    else:
        print("\n[WARNING] No model found - training fresh model...")
        model = MiniGPT(
            tokenizer.get_vocab_size(),
            config.embed_dim,
            config.block_size,
            config.num_blocks,
            getattr(config, "dropout", 0.0)
        ).to(device)
        tokens = text.split()
        train_encoded = tokenizer.encode(tokens)[:100000]
        train_loader = create_dataloader(train_encoded, config.block_size, config.batch_size)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        config_train = Config(epochs=10)
        trainer = Trainer(model, optimizer, criterion, config_train)
        trainer.train(train_loader, 10)
    
    # Run diagnostics
    generator = Generator(model, tokenizer.stoi, tokenizer.itos, config.block_size, device)
    
    test_prompts = [
        "python is",
        "machine learning is",
        "data science is",
    ]
    
    knowledge_score = evaluate_knowledge_representation(model, tokenizer, test_prompts, config, device)
    
    gap = evaluate_generalization_gap(model, tokenizer, data_path)
    
    mem_acc = evaluate_memorization_vs_learning(model, tokenizer, data_path)
    
    evaluate_instruction_following(model, tokenizer, generator)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    knowledge_rep = "[OK] Good" if knowledge_score >= 0.67 else "[WARNING] Weak"
    gap_status = "[OK] Small" if gap and gap < 1.0 else "[WARNING] Large"
    mem_status = "[WARNING] High" if mem_acc > 0.5 else "[OK] Moderate"
    gap_value = f"{gap:.4f}" if gap else "N/A"
    
    print(f"""
Knowledge Representation: {knowledge_rep}
Generalization Gap: {gap_value} {gap_status}
Memorization Level: {mem_acc*100:.1f}% {mem_status}
Instruction Following: [WARNING] Weak

Recommendations:
1. Increase model capacity (embed_dim, num_blocks)
2. More training data
3. Better regularization
4. Instruction-tuning phase
    """)


if __name__ == "__main__":
    data_path = "data/data.txt"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    run_full_diagnostic(data_path)
