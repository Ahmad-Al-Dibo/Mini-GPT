import time
import torch
from gpt_lib import (
    Config,
    create_config,
    Generator,
    build_training_diagnostics_report,
    build_model,
    build_trainer,
    create_epoch_checkpoint_callback,
    format_duration,
    format_long_context_results,
    load_compatible_checkpoint,
    prepare_data,
    run_long_context_evaluation,
    save_training_checkpoint,
)

from collections import Counter


def print_data_summary(data):
    print(f"\nData tokens: {len(data.tokens)}")
    print(f"Train tokens: {len(data.train_tokens)}")
    print(f"Validation tokens: {len(data.val_tokens)}")
    print(f"Vocab size: {data.vocab_size}")
    print(f"Train batches: {len(data.train_loader)}")
    if data.val_loader is not None:
        print(f"Validation batches: {len(data.val_loader)}")


def main():
    """Train or load the improved MiniGPT model."""
    config = create_config(
        # -------------------
        # MODEL CAPACITY
        # -------------------
        embed_dim=82,
        num_blocks=4,
        dropout=0.1,

        block_size=128,

        # -------------------
        # TRAINING
        # -------------------
        batch_size=32,
        epochs=10,

        learning_rate=3e-5,
        weight_decay=0.1,

        grad_clip=1.0,

        # -------------------
        # DATA
        # -------------------
        data_path="data/data.txt",

        max_data_size=3_700_000,
        validation_split=0.1,

        # -------------------
        # TOKENIZER
        # -------------------
        tokenizer_type="sentencepiece",
        sentencepiece_model_type="bpe",
        sentencepiece_character_coverage=1.0,

        max_vocab=20_000,
        tokenizer_rare_threshold=2,

        # -------------------
        # REGULARIZATION / STABILITY
        # -------------------
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,

        # -------------------
        # SYSTEM
        # -------------------
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",

        model_path="output/Meduim-T-f.pth",

        training_log_interval=50,

        domain_data_path = "data/data.txt",
        domain_data_repeats = 2

    )
    
    data = prepare_data(config)

    print_data_summary(data)

    counter = data.count()

    top = counter.most_common(50)

    print(top)

    model = build_model(config, data.vocab_size)
    print(f"Parameters: {model.get_num_parameters():,}")
    
    choise=input("\nDo you want to contune? press enter to contune else type 'q' to exit: ")
    if choise.lower() == "q":
      exit()

    trainer = build_trainer(model, config)
    checkpoint_loaded = load_compatible_checkpoint(trainer, config, data.vocab_size)

    if not checkpoint_loaded:
        print("\nNo improved checkpoint found; starting training.")
    checkpoint_callback = create_epoch_checkpoint_callback(
        config,
        data.tokenizer,
        data.vocab_size,
    )
    trainer.train(
        data.train_loader,
        config.epochs,
        val_loader=data.val_loader,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_callback=checkpoint_callback,
    )

    save_training_checkpoint(trainer, config, data.tokenizer, data.vocab_size)

    generator = Generator(
        model,
        data.tokenizer.stoi,
        data.tokenizer.itos,
        config.block_size,
        config.get_device(),
    )

    generated_text = generator.generate_string(
        "Programming is",
        max_new_tokens=config.diagnostic_sample_tokens,
        temperature=0.9,
        repetition_penalty=1.2,
        top_p=0.9,
    )

    print("\nGenerated sample:")
    print(generated_text)

    print()
    print(build_training_diagnostics_report(model, data, config))

    if config.run_long_context_evaluation:
        results = run_long_context_evaluation(
            config,
            block_sizes=config.long_context_block_sizes,
            sample_prompt="Programming is",
        )
        print()
        print(format_long_context_results(results))




if __name__ == "__main__":
  main()
