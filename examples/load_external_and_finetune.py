"""Load an external checkpoint source and fine-tune it through ArcLM."""

import argparse

import torch
import os
import sys

# add ../
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arclm import load_external_model, train_model


def main():
    parser = argparse.ArgumentParser(description="Inspect and fine-tune an external model source.")
    parser.add_argument("source", help="ArcLM checkpoint, raw state dict, safetensors file, or Hugging Face ID/folder.")
    parser.add_argument("--data", default="data/finetune.txt")
    parser.add_argument("--output", default="models/arclm_external_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    loaded = load_external_model(args.source, map_location="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded source type: {loaded.source_type}")
    print(f"Detected vocab size: {loaded.vocab_size}")

    result = train_model(
        mode="finetune",
        checkpoint=args.source,
        data=args.data,
        output=args.output,
        num_epochs=args.epochs,
        batch_size=8,
        learning_rate=2e-5,
        freeze_backbone=True,
        validation_split=0.1,
        training_log_interval=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Saved adapted fine-tuned checkpoint: {result.model_path}")

# example usage: python examples/load_external_and_finetune.py "path/to/external_model.pth" --data "data/finetune.txt" --output "models/arclm_external_finetuned.pth" --epochs 2

if __name__ == "__main__":
    main()
