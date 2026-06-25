import time
from collections import Counter

import torch

from arclm import Config, Generator, build_model, build_trainer, format_duration, prepare_data


def create_config() -> Config:
    return Config(
        embed_dim=64,
        num_blocks=2,
        dropout=0.1,
        block_size=32,
        batch_size=8,
        num_epochs=3,
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        data_path="data/data.txt",
        max_data_size=20_000,
        validation_split=0.2,
        tokenizer_type="word",
        max_vocab=2_000,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
        training_log_interval=10,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path="models/arclm.pth",
    )


def main():
    start = time.time()
    config = create_config()
    torch.manual_seed(config.seed)

    print("\nArcLM training")
    print(config)

    data = prepare_data(config)
    config.vocab_size = data.vocab_size
    print(f"\nData tokens: {len(data.tokens):,}")
    print(f"Train tokens: {len(data.train_tokens):,}")
    print(f"Validation tokens: {len(data.val_tokens):,}")
    print(f"Vocab size: {data.vocab_size:,}")
    print(f"Train batches: {len(data.train_loader):,}")
    if data.val_loader is not None:
        print(f"Validation batches: {len(data.val_loader):,}")

    decoded_sample = data.tokenizer.decode(data.train_encoded[:20])
    counter = Counter(data.tokenizer.decode(data.train_encoded))
    print(f"Decoded sample: {decoded_sample}")
    print(f"Top tokens: {counter.most_common(5)}")

    model = build_model(config, data.vocab_size)
    print(f"Parameters: {model.get_num_parameters():,}")

    trainer = build_trainer(model, config)
    trainer.train(
        data.train_loader,
        config.num_epochs,
        val_loader=data.val_loader,
        early_stopping_patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    trainer.save(
        config,
        vocab=data.tokenizer.vocab,
        stoi=data.tokenizer.stoi,
        itos=data.tokenizer.itos,
        tokenizer_metadata=data.tokenizer.to_checkpoint(),
    )

    generator = Generator(
        model=model,
        stoi=data.tokenizer.stoi,
        itos=data.tokenizer.itos,
        block_size=config.block_size,
        device=torch.device(config.device),
        tokenizer=data.tokenizer,
    )
    print("\nGenerated sample:")
    print(generator.generate_string("machine learning", max_new_tokens=30, top_p=0.9))

    print(f"\nSaved model: {config.model_path}")
    print(f"Total time: {format_duration(time.time() - start)}")


if __name__ == "__main__":
    main()
