import argparse
import json
import os
import sys

import torch

project_root = os.path.abspath(".....")
sys.path.insert(0, project_root)

from src.miniGPT import (
    Generator,
    SentencePieceTokenizer,
    build_model,
    build_trainer,
    build_training_diagnostics_report,
    create_config,
    create_epoch_checkpoint_callback,
    load_compatible_checkpoint,
    load_model_checkpoint,
    prepare_data,
    run_long_context_evaluation,
    save_training_checkpoint,
    format_long_context_results,
)


def print_data_summary(data):
    print("\n=== Dataset Summary ===")
    print(f"Data tokens      : {len(data.tokens)}")
    print(f"Train tokens     : {len(data.train_tokens)}")
    print(f"Validation tokens: {len(data.val_tokens)}")
    print(f"Vocab size       : {data.vocab_size}")
    print(f"Train batches    : {len(data.train_loader)}")

    if data.val_loader is not None:
        print(f"Validation batches: {len(data.val_loader)}")

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training an existing GPT model."
    )

    parser.add_argument(
        "model_path",
        help="Path to model checkpoint (.pth)"
    )

    parser.add_argument(
        "data_path",
        help="Path to training dataset"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cpu/cuda)"
    )

    parser.add_argument(
        "--tokenizer",
        default="data/tokenizer.json",
        help="Tokenizer JSON file"
    )

    return parser.parse_args()


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    return SentencePieceTokenizer.from_json(tokenizer_data)


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model_path}"
        )

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}"
        )

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}"
        )

    checkpoint = load_model_checkpoint(
        args.model_path,
        "cpu"
    )

    old_model_settings = checkpoint.get("config", {})

    if not old_model_settings:
        raise ValueError(
            "Failed to load configuration from checkpoint."
        )

    old_model_settings.update(
        {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "device": args.device,
            "use_checkpoint_tokenizer": True,
            "diagnostic_prompts": [
                "python is a ",
                "javascript is a ",
            ],
            # "no_data_limate": True
        }
    )

    cfg = create_config(**old_model_settings)

    tokenizer = load_tokenizer(args.tokenizer)

    data = prepare_data(
        cfg,
        old_tokenizer=tokenizer
    )

    print_data_summary(data)

    model = build_model(
        cfg,
        data.vocab_size
    )

    print(
        f"Parameters: {model.get_num_parameters():,}"
    )

    trainer = build_trainer(
        model,
        cfg
    )

    checkpoint_loaded = checkpoint_is_compatible_for_tuining(
        trainer,
        cfg,
        data.vocab_size
    )

    if not checkpoint_loaded:
        print(
            "\nNo compatible training checkpoint found. "
            "Starting from model checkpoint."
        )

    checkpoint_callback = (
        create_epoch_checkpoint_callback(
            cfg,
            data.tokenizer,
            data.vocab_size,
        )
    )

    trainer.train(
        data.train_loader,
        cfg.epochs,
        val_loader=data.val_loader,
        early_stopping_patience=cfg.early_stopping_patience,
        checkpoint_callback=checkpoint_callback,
    )

    save_training_checkpoint(
        trainer,
        cfg,
        data.tokenizer,
        data.vocab_size,
    )

    generator = Generator(
        model,
        data.tokenizer.stoi,
        data.tokenizer.itos,
        cfg.block_size,
        cfg.get_device(),
    )

    generated_text = generator.generate_string(
        "python is",
        max_new_tokens=cfg.diagnostic_sample_tokens,
        temperature=0.9,
        repetition_penalty=1.2,
        top_p=0.9,
    )

    print("\n=== Generated Sample ===")
    print(generated_text)

    print("\n=== Diagnostics ===")
    print(
        build_training_diagnostics_report(
            model,
            data,
            cfg,
        )
    )

    if cfg.run_long_context_evaluation:
        results = run_long_context_evaluation(
            cfg,
            block_sizes=cfg.long_context_block_sizes,
            sample_prompt="Programming is",
        )

        print("\n=== Long Context Evaluation ===")
        print(
            format_long_context_results(
                results
            )
        )


if __name__ == "__main__":
    main()