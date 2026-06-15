"""
Example 4: Repetition Penalty & Generalization Features
Demonstratie van repetition penalty en generalisatie eigenschappen
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_lib import Config, Tokenizer, MiniGPT, Generator, GeneralizationMonitor


def demo_repetition_penalty():
    """Toon verschil tussen met en zonder repetition penalty"""
    
    model_path = "output/mini_gpt.pth"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        return
    
    print("\n" + "="*60)
    print("REPETITION PENALTY DEMO")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    
    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["config"]["embed_dim"],
        block_size=checkpoint["config"]["block_size"],
        num_blocks=checkpoint["config"]["num_blocks"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    generator = Generator(
        model,
        checkpoint["stoi"],
        checkpoint["itos"],
        checkpoint["block_size"],
        "cpu"
    )
    
    # Test prompts
    prompts = [
        "machine learning is",
        "artificial intelligence",
        "the future of technology",
        "what is python"
    ]
    
    print("\nVergelijking van generatie resultaten:\n")
    
    for prompt in prompts:
        print(f"\n{'-'*60}")
        print(f"Prompt: '{prompt}'")
        print(f"{'-'*60}")
        
        # Zonder penalty (baseline)
        print("\n[ERROR] ZONDER Repetition Penalty (baseline):")
        text_no_penalty = generator.generate_string(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            repetition_penalty=1.0  # No penalty
        )
        print(f"   {text_no_penalty}")
        
        # Met lage penalty
        print("\n[WARNING]  MET Low Repetition Penalty (1.2):")
        text_low_penalty = generator.generate_string(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            repetition_penalty=1.2
        )
        print(f"   {text_low_penalty}")
        
        # Met hoge penalty
        print("\n[OK] MET High Repetition Penalty (2.0):")
        text_high_penalty = generator.generate_string(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            repetition_penalty=2.0
        )
        print(f"   {text_high_penalty}")
        
        # Analyse repetities
        def count_repeated_words(text):
            words = text.split()
            return sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        
        print(f"\n   Repeated words (no penalty): {count_repeated_words(text_no_penalty)}")
        print(f"   Repeated words (low penalty): {count_repeated_words(text_low_penalty)}")
        print(f"   Repeated words (high penalty): {count_repeated_words(text_high_penalty)}")


def demo_sampling_strategies():
    """Toon verschillende sampling strategieen"""
    
    model_path = "output/mini_gpt.pth"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found")
        return
    
    print("\n" + "="*60)
    print("SAMPLING STRATEGIES DEMO")
    print("="*60)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    
    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["config"]["embed_dim"],
        block_size=checkpoint["config"]["block_size"],
        num_blocks=checkpoint["config"]["num_blocks"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    generator = Generator(
        model,
        checkpoint["stoi"],
        checkpoint["itos"],
        checkpoint["block_size"],
        "cpu"
    )
    
    prompt = "python programming"
    
    print(f"\nPrompt: '{prompt}'\n")
    
    # Standard sampling
    print("1. STANDARD SAMPLING (temperature=1.0):")
    text1 = generator.generate_string(prompt, max_new_tokens=15, temperature=1.0)
    print(f"   {text1}\n")
    
    # Top-K sampling
    print("2. TOP-K SAMPLING (keep top 50 tokens):")
    text2 = generator.generate_string(prompt, max_new_tokens=15, top_k=50)
    print(f"   {text2}\n")
    
    # Top-P (nucleus) sampling
    print("3. TOP-P SAMPLING (keep tokens with cum_prob >= 0.9):")
    text3 = generator.generate_string(prompt, max_new_tokens=15, top_p=0.9)
    print(f"   {text3}\n")
    
    # Combined
    print("4. COMBINED (temp=1.2, repetition_penalty=1.5, top_p=0.9):")
    text4 = generator.generate_string(
        prompt,
        max_new_tokens=15,
        temperature=1.2,
        repetition_penalty=1.5,
        top_p=0.9
    )
    print(f"   {text4}\n")


def demo_generalization():
    """Toon generalisatie monitoring"""
    
    model_path = "output/mini_gpt.pth"
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found")
        return
    
    print("\n" + "="*60)
    print("GENERALIZATION MONITORING DEMO")
    print("="*60)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Haal training history op
    if "train_history" in checkpoint:
        history = checkpoint["train_history"]
        
        monitor = GeneralizationMonitor()
        
        # Populate with history
        if history.get("train_losses") and history.get("val_losses"):
            for tl, vl in zip(history["train_losses"], history["val_losses"]):
                monitor.update(tl, vl)
        
        print(monitor.get_report())
    else:
        print("[ERROR] No training history in checkpoint")


if __name__ == "__main__":
    import sys
    
    print("\nGPT Library - Advanced Features Demo")
    print("="*60)
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
        
        if demo_type == "penalty":
            demo_repetition_penalty()
        elif demo_type == "sampling":
            demo_sampling_strategies()
        elif demo_type == "generalization":
            demo_generalization()
        else:
            print("[ERROR] Unknown demo type")
            print("Usage: python example4_advanced.py [penalty|sampling|generalization|all]")
    else:
        print("\nAvailable demos:")
        print("  1. Repetition Penalty Demo")
        demo_repetition_penalty()
        
        print("\n\n2. Sampling Strategies Demo")
        demo_sampling_strategies()
        
        print("\n\n3. Generalization Monitoring")
        demo_generalization()
