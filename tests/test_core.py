import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_lib import (
    ConceptBenchmarkCase,
    Config,
    MiniGPT,
    SentencePieceTokenizer,
    Tokenizer,
    Trainer,
    build_model,
    build_trainer,
    checkpoint_is_compatible,
    create_dataloader,
    load_model,
    predict_top_k,
    prepare_data,
    score_concept_relationships,
)


class CorePipelineTests(unittest.TestCase):
    def _build_resume_trainer(self):
        config = Config(
            embed_dim=16,
            block_size=4,
            batch_size=4,
            epochs=2,
            learning_rate=1e-3,
            dropout=0.0,
            grad_clip=1.0,
            training_log_interval=0,
            restore_best_model=False,
            device="cpu",
        )
        data = [i % 8 for i in range(40)]
        loader = create_dataloader(data, config.block_size, config.batch_size)
        model = MiniGPT(8, config.embed_dim, config.block_size, 1, config.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(), config)
        return config, loader, trainer

    def test_tokenizer_round_trip(self):
        tokenizer = Tokenizer(max_vocab=10)
        tokenizer.build("python is useful python")

        encoded = tokenizer.encode(["python", "missing", "useful"])
        decoded = tokenizer.decode(encoded)

        self.assertEqual(encoded[0], tokenizer.stoi["python"])
        self.assertEqual(encoded[1], tokenizer.stoi["<UNK>"])
        self.assertEqual(decoded[0], "python")
        self.assertEqual(decoded[1], "<UNK>")

    def test_tokenizer_reports_coverage(self):
        tokenizer = Tokenizer(max_vocab=4)
        tokenizer.build("python python data models")

        coverage = tokenizer.analyze_coverage(["python", "unknown", "models"])

        self.assertEqual(coverage["total_tokens"], 3)
        self.assertEqual(coverage["unknown_tokens"], 1)
        self.assertGreater(coverage["vocab_coverage"], 0)
        self.assertIn(("unknown", 1), coverage["top_unknown_tokens"])

    def test_sentencepiece_tokenizer_round_trips_checkpoint_metadata(self):
        tokenizer = SentencePieceTokenizer(max_vocab=40)
        tokenizer.build(
            "python code builds software. javascript code runs in browsers. "
            "machine learning models use tokens."
        )
        metadata = tokenizer.to_checkpoint()
        restored = SentencePieceTokenizer.from_checkpoint(metadata)

        text = "python code models"

        self.assertEqual(
            restored.encode_text(text),
            tokenizer.encode_text(text),
        )
        self.assertEqual(
            restored.decode_string(tokenizer.encode_text(text)),
            tokenizer.decode_string(tokenizer.encode_text(text)),
        )

    def test_dataloader_shapes(self):
        loader = create_dataloader(list(range(20)), block_size=4, batch_size=2, shuffle=False)
        x, y = next(iter(loader))

        self.assertEqual(tuple(x.shape), (2, 4))
        self.assertEqual(tuple(y.shape), (2, 4))
        self.assertTrue(torch.equal(y[0], x[0] + 1))

    def test_model_forward_shape(self):
        model = MiniGPT(vocab_size=12, embed_dim=16, block_size=4, num_blocks=1, dropout=0.1)
        idx = torch.randint(0, 12, (3, 4))

        logits = model(idx)

        self.assertEqual(tuple(logits.shape), (3, 4, 12))

    def test_trainer_tracks_validation_metrics(self):
        config = Config(
            embed_dim=16,
            block_size=4,
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            dropout=0.1,
            grad_clip=1.0,
            device="cpu",
        )
        data = [i % 8 for i in range(40)]
        train_loader = create_dataloader(data[:30], config.block_size, config.batch_size)
        val_loader = create_dataloader(data[20:], config.block_size, config.batch_size, shuffle=False)
        model = MiniGPT(8, config.embed_dim, config.block_size, 1, config.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(), config)

        trainer.train(train_loader, epochs=1, val_loader=val_loader, early_stopping_patience=3)

        history = trainer.get_train_history()
        self.assertEqual(len(history["train_losses"]), 1)
        self.assertEqual(len(history["val_losses"]), 1)
        self.assertEqual(len(history["val_perplexities"]), 1)
        self.assertEqual(len(history["val_token_accuracies"]), 1)
        self.assertEqual(history["current_epoch"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            trainer.save(model_path, vocab=["<UNK>"], stoi={"<UNK>": 0}, itos={0: "<UNK>"}, block_size=4, vocab_size=8)
            self.assertTrue(model_path.exists())

    def test_trainer_saves_and_loads_current_epoch(self):
        _, loader, trainer = self._build_resume_trainer()

        trainer.train(loader, epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            trainer.save(model_path, vocab=["<UNK>"], stoi={"<UNK>": 0}, itos={0: "<UNK>"}, block_size=4, vocab_size=8)

            _, _, loaded_trainer = self._build_resume_trainer()
            loaded_trainer.load(model_path)

        self.assertEqual(loaded_trainer.current_epoch, 1)
        self.assertEqual(len(loaded_trainer.train_losses), 1)

    def test_trainer_resumes_from_next_epoch(self):
        _, loader, trainer = self._build_resume_trainer()
        trainer.train(loader, epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            trainer.save(model_path, vocab=["<UNK>"], stoi={"<UNK>": 0}, itos={0: "<UNK>"}, block_size=4, vocab_size=8)

            _, resume_loader, resumed_trainer = self._build_resume_trainer()
            resumed_trainer.load(model_path)
            resumed_trainer.train(resume_loader, epochs=2)

        self.assertEqual(resumed_trainer.current_epoch, 2)
        self.assertEqual(len(resumed_trainer.train_losses), 2)

    def test_trainer_skips_when_target_epochs_complete(self):
        _, loader, trainer = self._build_resume_trainer()
        trainer.train(loader, epochs=1)
        callback_calls = []

        trainer.train(loader, epochs=1, checkpoint_callback=lambda _: callback_calls.append(True))

        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(len(trainer.train_losses), 1)
        self.assertEqual(callback_calls, [])

    def test_old_checkpoint_load_infers_current_epoch_from_history(self):
        _, _, trainer = self._build_resume_trainer()

        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_history": {
                "train_losses": [1.0, 0.9],
                "val_losses": [],
                "val_perplexities": [],
                "val_token_accuracies": [],
                "best_val_loss": float("inf"),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "old_model.pth"
            torch.save(checkpoint, model_path)
            trainer.load(model_path)

        self.assertEqual(trainer.current_epoch, 2)
        self.assertEqual(trainer.train_losses, [1.0, 0.9])

    def test_prepare_data_and_pipeline_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.txt"
            data_path.write_text(
                "python is a programming language. "
                "machine learning uses data models. "
                "database systems store query data. "
                "python software is useful. "
                "language models use tokens.",
                encoding="utf-8",
            )
            config = Config(
                data_path=str(data_path),
                domain_data_path=None,
                max_data_size=60,
                max_vocab=20,
                block_size=4,
                batch_size=2,
                validation_split=0.2,
                embed_dim=16,
                num_blocks=1,
                dropout=0.0,
                learning_rate=1e-3,
                device="cpu",
            )

            data = prepare_data(config)
            model = build_model(config, data.vocab_size)
            trainer = build_trainer(model, config)

            self.assertGreater(data.vocab_size, 1)
            self.assertEqual(model.block_size, config.block_size)
            self.assertIsInstance(trainer, Trainer)

    def test_checkpoint_compatibility(self):
        config = Config(
            embed_dim=16,
            block_size=4,
            num_blocks=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            dropout=0.0,
            validation_split=0.2,
        )
        checkpoint = {
            "vocab_size": 8,
            "config": {
                "embed_dim": 16,
                "block_size": 4,
                "num_blocks": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "validation_split": 0.2,
            },
        }

        self.assertTrue(checkpoint_is_compatible(checkpoint, config, vocab_size=8))
        self.assertFalse(checkpoint_is_compatible(checkpoint, config, vocab_size=9))

    def test_checkpoint_compatibility_checks_tokenizer_mapping(self):
        config = Config(
            embed_dim=16,
            block_size=4,
            num_blocks=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            dropout=0.0,
            validation_split=0.2,
        )
        tokenizer = Tokenizer(max_vocab=10)
        tokenizer.build("python code data")
        checkpoint = {
            "vocab_size": tokenizer.get_vocab_size(),
            "stoi": dict(tokenizer.stoi),
            "config": {
                "embed_dim": 16,
                "block_size": 4,
                "num_blocks": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "validation_split": 0.2,
            },
        }

        self.assertTrue(
            checkpoint_is_compatible(
                checkpoint,
                config,
                vocab_size=tokenizer.get_vocab_size(),
                tokenizer=tokenizer,
            )
        )

        checkpoint["stoi"] = {"<UNK>": 0, "different": 1}
        self.assertFalse(
            checkpoint_is_compatible(
                checkpoint,
                config,
                vocab_size=tokenizer.get_vocab_size(),
                tokenizer=tokenizer,
            )
        )

    def test_top_k_and_concept_diagnostics(self):
        tokenizer = Tokenizer(max_vocab=10)
        tokenizer.build("python programming language software data")
        model = MiniGPT(
            tokenizer.get_vocab_size(),
            embed_dim=16,
            block_size=4,
            num_blocks=1,
            dropout=0.0,
        )

        predictions = predict_top_k(
            model,
            tokenizer.stoi,
            tokenizer.itos,
            block_size=4,
            device="cpu",
            prompt="python",
            k=3,
        )
        report = score_concept_relationships(
            model,
            tokenizer.stoi,
            tokenizer.itos,
            block_size=4,
            device="cpu",
            benchmark_cases=[
                ConceptBenchmarkCase("python", ["programming", "language"])
            ],
            k=3,
        )

        self.assertEqual(len(predictions), 3)
        self.assertIn("overall_score", report)
        self.assertEqual(len(report["results"]), 1)

    def test_inference_loader_loads_checkpoint(self):
        config = Config(
            embed_dim=16,
            block_size=4,
            num_blocks=1,
            dropout=0.0,
            device="cpu",
        )
        stoi = {"<UNK>": 0, "python": 1, "code": 2}
        itos = {index: token for token, index in stoi.items()}
        model = MiniGPT(
            vocab_size=len(stoi),
            embed_dim=config.embed_dim,
            block_size=config.block_size,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "vocab": list(stoi),
                    "vocab_size": len(stoi),
                    "block_size": config.block_size,
                },
                model_path,
            )

            loaded_model = load_model(model_path=model_path, device="cpu")
            prediction = loaded_model.predict("python", max_new_tokens=1)

        self.assertEqual(loaded_model.config.block_size, config.block_size)
        self.assertEqual(loaded_model.model_path, model_path)
        self.assertIsInstance(prediction, str)

    def test_inference_loader_uses_saved_sentencepiece_tokenizer(self):
        tokenizer = SentencePieceTokenizer(max_vocab=40)
        tokenizer.build(
            "python code builds software. javascript code runs in browsers. "
            "machine learning models use tokens."
        )
        config = Config(
            embed_dim=16,
            block_size=4,
            num_blocks=1,
            dropout=0.0,
            device="cpu",
            tokenizer_type="sentencepiece",
            max_vocab=tokenizer.get_vocab_size(),
        )
        model = MiniGPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=config.embed_dim,
            block_size=config.block_size,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                    "vocab": tokenizer.vocab,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "block_size": config.block_size,
                    "tokenizer_metadata": tokenizer.to_checkpoint(),
                },
                model_path,
            )

            loaded_model = load_model(model_path=model_path, device="cpu")

        text = "python code models"
        self.assertEqual(
            loaded_model.generator.tokenizer.encode_text(text),
            tokenizer.encode_text(text),
        )


if __name__ == "__main__":
    unittest.main()
