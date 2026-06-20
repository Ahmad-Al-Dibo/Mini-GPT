"""
Example 2: Custom Model Configuration
Hoe je andere modelmaten kan trainen
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.miniGPT import Config, Tokenizer, create_dataloader, MiniGPT, Trainer


def train_custom_model(data_path, model_size="small"):
    """Train model with custom configuration
    
    Args:
        data_path: Path to training data
        model_size: "small", "medium", or "large"
    """
    
    # Define configurations
    configs = {
        "small": {
            "embed_dim": 32,
            "num_blocks": 1,
            "batch_size": 128,
            "epochs": 50,
            "learning_rate": 5e-4,
        },
        "medium": {
            "embed_dim": 128,
            "num_blocks": 4,
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 1e-3,
        },
        "large": {
            "embed_dim": 256,
            "num_blocks": 8,
            "batch_size": 32,
            "epochs": 200,
            "learning_rate": 5e-4,
        },
    }
    
    model_config = configs[model_size]
    
    # Create config
    config = Config(
        **model_config,
        model_path=f"output/model_{model_size}.pth",
        data_path=data_path,
    )
    
    print(f"\n{'='*50}")
    print(f"Training {model_size.upper()} Model")
    print(f"{'='*50}")
    print(config)
    
    # Load and tokenize data
    print("\nLoading data...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    tokens = text.split()[:config.max_data_size]
    
    # Tokenize
    print("Building tokenizer...")
    tokenizer = Tokenizer(max_vocab=config.max_vocab)
    tokenizer.build(" ".join(tokens))
    vocab_size = tokenizer.get_vocab_size()
    
    encoded = tokenizer.encode(tokens)
    
    # Create dataloader
    print("Creating dataloader...")
    loader = create_dataloader(
        encoded,
        config.block_size,
        config.batch_size
    )
    
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
    
    # Create trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, config)
    
    # Train
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
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example2_custom.py <data_path> [small|medium|large]")
        print("Example: python example2_custom.py data/data.txt medium")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    
    train_custom_model(data_path, model_size)
