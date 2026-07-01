"""Fine-tune an existing ArcLM checkpoint."""

import argparse

import torch

from arclm import train_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an ArcLM checkpoint.")
    parser.add_argument("--checkpoint", default="models/arclm_pretrained.pth")
    parser.add_argument("--data", default="data/finetune.txt")
    parser.add_argument("--output", default="models/arclm_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = train_model(
        mode="finetune",
        checkpoint=args.checkpoint,
        data=args.data,
        output=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=2e-5,
        freeze_backbone=True,
        use_discriminative_lr=True,
        validation_split=0.1,
        training_log_interval=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Saved fine-tuned checkpoint: {result.model_path}")


if __name__ == "__main__":
    main()
