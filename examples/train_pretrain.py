"""Pretrain a compact ArcLM checkpoint from local text."""

import argparse

import torch

from arclm import train_model


def main():
    parser = argparse.ArgumentParser(description="Pretrain ArcLM from a text file.")
    parser.add_argument("--data", default="data/pretrain.txt")
    parser.add_argument("--output", default="models/arclm_pretrained.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=2000)
    args = parser.parse_args()

    result = train_model(
        mode="pretrain",
        data=args.data,
        output=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        max_vocab=args.max_vocab,
        validation_split=0.1,
        learning_rate=3e-4,
        grad_clip=1.0,
        training_log_interval=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Saved pretrained checkpoint: {result.model_path}")


if __name__ == "__main__":
    main()
