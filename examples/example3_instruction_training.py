"""
Example 3: Instruction-Following Training
Hoe je het model trainen voor instructies
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.miniGPT import Config, Tokenizer, create_dataloader, MiniGPT, Trainer


def prepare_instruction_data(input_path, output_path):
    """Convert plain text to instruction-response format
    
    Expected input format (one per line):
        Question: What is Python? Answer: Python is...
    
    Output format for training:
        [INST] what is python? [/INST] python is a ...
    """
    
    formatted_data = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            
            # Parse instruction-answer format
            if "?" in line and ":" in line:
                parts = line.split("?")
                if len(parts) >= 2:
                    question = parts[0].replace("question:", "").strip()
                    answer = "?".join(parts[1:]).replace("answer:", "").strip()
                    
                    # Format as instruction-response
                    formatted = f"[INST] {question}? [/INST] {answer}"
                    formatted_data.append(formatted)
    
    # Write formatted data
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_data))
    
    print(f"[OK] Converted {len(formatted_data)} instruction-response pairs")
    print(f"[OK] Saved to {output_path}")


def train_instruction_model(data_path):
    """Train model specifically for instruction following"""
    
    # Create config for instruction-following
    config = Config(
        embed_dim=128,
        block_size=16,  # Longer context for instructions
        batch_size=32,
        epochs=50,
        learning_rate=1e-3,
        num_blocks=3,  # More blocks for complex instructions
        model_path="output/model_instruction.pth",
        data_path=data_path,
    )
    
    print(f"\n{'='*50}")
    print("Training Instruction-Following Model")
    print(f"{'='*50}")
    print(config)
    
    # Load data
    print("\nLoading instruction data...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    tokens = text.split()[:config.max_data_size]
    
    # Build tokenizer (include special tokens)
    print("Building tokenizer...")
    tokenizer = Tokenizer(max_vocab=config.max_vocab)
    tokenizer.build(" ".join(tokens))
    vocab_size = tokenizer.get_vocab_size()
    
    encoded = tokenizer.encode(tokens)
    
    # Create dataloader
    print("Creating dataloader...")
    loader = create_dataloader(encoded, config.block_size, config.batch_size)
    
    # Create model
    print("Creating model...")
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        block_size=config.block_size,
        num_blocks=config.num_blocks
    )
    model = model.to(torch.device(config.device))
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, config)
    
    print("\nTraining...")
    trainer.train(loader, config.epochs)
    
    # Save
    print("\nSaving...")
    trainer.save(
        config.model_path,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        block_size=config.block_size,
        vocab_size=vocab_size
    )


if __name__ == "__main__":
    print("Instruction-Following Model Training")
    print("="*50)
    
    # Check if we need to format data first
    if len(sys.argv) > 1 and sys.argv[1] == "format":
        if len(sys.argv) < 4:
            print("Usage: python example3_instruction.py format <input> <output>")
            sys.exit(1)
        prepare_instruction_data(sys.argv[2], sys.argv[3])
    else:
        data_path = "data/instruction_data.txt"
        if not Path(data_path).exists():
            print(f"[ERROR] Instruction data not found at {data_path}")
            print("   Format your data first:")
            print("   python example3_instruction.py format input.txt data/instruction_data.txt")
            sys.exit(1)
        
        train_instruction_model(data_path)
