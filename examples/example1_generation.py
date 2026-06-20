"""
Example 1: Basic Text Generation
Eenvoudige tekstgeneratie met trained model
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.miniGPT import Config, Tokenizer, MiniGPT, Generator


def main():
    # Load model checkpoint
    model_path = "models/mini_gpt.pth"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("   Run train.py first to train a model")
        return
    
    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Recreate model
    config_dict = checkpoint["config"]
    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=config_dict["embed_dim"],
        block_size=config_dict["block_size"],
        num_blocks=config_dict["num_blocks"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create generator
    generator = Generator(
        model,
        checkpoint["stoi"],
        checkpoint["itos"],
        checkpoint["block_size"],
        "cpu"
    )
    
    # Generate text
    print("\n" + "="*50)
    print("Text Generation Examples")
    print("="*50)
    
    prompts = [
        "english is",
        "machine learning",
        "artificial intelligence",
        "python is",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 50)
        
        for temp in [0.5, 1.0, 1.5]:
            text = generator.generate_string(
                prompt,
                max_new_tokens=15,
                temperature=temp
            )
            print(f"Temp {temp}: {text}")


if __name__ == "__main__":
    main()
