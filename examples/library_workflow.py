from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arclm import Config, Generator, build_model, build_trainer, prepare_data


def main():
    config = Config(
        data_path="data/data.txt",
        model_path="models/example_model.pth",
        tokenizer_type="word",
        max_vocab=1000,
        embed_dim=32,
        num_blocks=1,
        block_size=16,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,
        validation_split=0.0,
        training_log_interval=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    data = prepare_data(config)
    config.vocab_size = data.vocab_size

    model = build_model(config, data.vocab_size)
    trainer = build_trainer(model, config)
    trainer.train(data.train_loader, config.num_epochs)

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

    print(generator.generate_string("machine learning", max_new_tokens=12, top_p=0.9))
    print(f"Saved checkpoint to {config.model_path}")


if __name__ == "__main__":
    main()
