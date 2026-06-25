from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arclm import Config, Tokenizer, build_model, build_trainer, create_dataloader, prepare_data


def test_tokenizer_round_trip():
    tokenizer = Tokenizer(max_vocab=10)
    tokenizer.build("arc lm arc model")

    encoded = tokenizer.encode(["arc", "missing", "model"])
    decoded = tokenizer.decode(encoded)

    assert decoded[0] == "arc"
    assert decoded[1] == "<UNK>"
    assert decoded[2] == "model"


def test_model_forward_shape():
    config = Config(embed_dim=16, block_size=4, num_blocks=1, dropout=0.0, device="cpu")
    model = build_model(config, vocab_size=8)

    logits = model(torch.randint(0, 8, (2, 4)))

    assert tuple(logits.shape) == (2, 4, 8)


def test_prepare_train_and_save(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text(
        "arc lm learns from text. arc lm predicts tokens. text data trains models. " * 8,
        encoding="utf-8",
    )

    config = Config(
        data_path=str(data_path),
        model_path=str(tmp_path / "model.pth"),
        tokenizer_type="word",
        max_vocab=100,
        embed_dim=16,
        num_blocks=1,
        block_size=4,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        validation_split=0.0,
        training_log_interval=0,
        device="cpu",
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

    assert Path(config.model_path).exists()


def test_create_dataloader_shapes():
    loader = create_dataloader(list(range(20)), block_size=4, batch_size=2, shuffle=False)
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 4)
    assert tuple(y.shape) == (2, 4)
