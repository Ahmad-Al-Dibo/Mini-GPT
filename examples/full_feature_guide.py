"""End-to-end ArcLM feature guide.

This example is intentionally small and local. It creates tiny demo data under
``runs/full_feature_guide/`` and walks through the main library surfaces without
requiring external model downloads.
"""

from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arclm import (
    Config,
    DataProcessor,
    SmartLoader,
    build_model,
    build_trainer,
    build_training_diagnostics_report,
    create_checkpoint_callback,
    load_external_model,
    load_model,
    predict_top_k,
    prepare_data,
    train_model,
)
from arclm.config_loader import save_config_json, load_config_json
from arclm.tracking import create_experiment


SPECIAL_TOKENS = [
    "<|qa_start|>",
    "<|res_start|>",
    "<|end|>",
    "<|pad|>",
]


def write_demo_files(root):
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    pretrain_path = data_dir / "pretrain.txt"
    pretrain_path.write_text(
        (
            "arc language models learn from compact local text. "
            "arc language models predict the next token. "
            "<|qa_start|> what is arclm? <|res_start|> arclm is a compact training library. <|end|> "
        )
        * 20,
        encoding="utf-8",
    )

    qa_jsonl = data_dir / "qa.jsonl"
    records = [
        {
            "question": "What is ArcLM?",
            "context": "local model training",
            "answer": "ArcLM is a compact PyTorch language-model library.",
        },
        {
            "question": "What does SmartLoader inspect?",
            "context": "model folders and checkpoints",
            "answer": "It inspects metadata, tokenizer files, weights, and resume state.",
        },
    ]
    qa_jsonl.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    return pretrain_path, qa_jsonl


def prepare_instruction_text(qa_jsonl, output_path):
    dataset = (
        DataProcessor.load(qa_jsonl)
        .clean()
        .transform(
            format="instruction",
            mapping={
                "instruction": "question",
                "input": "context",
                "output": "answer",
            },
            template="<|qa_start|> {instruction}\n{input}\n<|res_start|> {output} <|end|>",
        )
    )
    output_path.write_text(
        "\n".join(sample["text"] for sample in dataset.samples),
        encoding="utf-8",
    )
    return dataset


def high_level_training(root, pretrain_path, finetune_path):
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)

    pretrained = train_model(
        mode="pretrain",
        data=str(pretrain_path),
        output=str(model_dir / "arclm_pretrained.pth"),
        tokenizer_type="word",
        user_defined_symbols=SPECIAL_TOKENS,
        max_vocab=128,
        embed_dim=16,
        num_blocks=1,
        block_size=12,
        batch_size=2,
        num_epochs=1,
        validation_split=0.1,
        early_stopping_patience=2,
        training_log_interval=0,
        checkpoint_batch_interval=100,
        device="cpu",
    )

    finetuned = train_model(
        mode="finetune",
        checkpoint=pretrained.model_path,
        data=str(finetune_path),
        output=str(model_dir / "arclm_finetuned.pth"),
        user_defined_symbols=SPECIAL_TOKENS,
        batch_size=2,
        num_epochs=1,
        learning_rate=2e-5,
        freeze_backbone=True,
        validation_split=0.1,
        early_stopping_patience=2,
        training_log_interval=0,
        checkpoint_batch_interval=50,
        device="cpu",
    )

    return pretrained, finetuned


def inference_and_diagnostics(model_path):
    loaded = load_model(model_path, device="cpu")
    prompt = "<|qa_start|> what is arclm? <|res_start|>"
    print("Generated:")
    print(loaded.predict(prompt, max_new_tokens=12, top_p=0.9))

    predictions = predict_top_k(
        loaded.model,
        loaded.generator.stoi,
        loaded.generator.itos,
        loaded.config.block_size,
        loaded.device,
        prompt,
        k=3,
        tokenizer=loaded.generator.tokenizer,
    )
    print("Top-k:")
    for item in predictions:
        print(f"  {item.rank}. {item.token}: {item.probability:.3f}")


def lower_level_training(root, pretrain_path):
    config = Config(
        data_path=str(pretrain_path),
        model_path=str(root / "models" / "arclm_low_level.pth"),
        tokenizer_type="word",
        user_defined_symbols=SPECIAL_TOKENS,
        max_vocab=128,
        embed_dim=16,
        num_blocks=1,
        block_size=12,
        batch_size=2,
        num_epochs=1,
        validation_split=0.1,
        early_stopping_patience=2,
        training_log_interval=0,
        checkpoint_batch_interval=100,
        device="cpu",
    )

    data = prepare_data(config)
    config.vocab_size = data.vocab_size
    model = build_model(config, data.vocab_size)
    trainer = build_trainer(model, config)
    trainer.train(
        data.train_loader,
        config.num_epochs,
        val_loader=data.val_loader,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_callback=create_checkpoint_callback(config, data.tokenizer, data.vocab_size),
        checkpoint_batch_interval=config.checkpoint_batch_interval,
    )
    trainer.save(
        config,
        vocab=data.tokenizer.vocab,
        stoi=data.tokenizer.stoi,
        itos=data.tokenizer.itos,
        tokenizer_metadata=data.tokenizer.to_checkpoint(),
    )
    print(build_training_diagnostics_report(model, data, config))
    return config


def loader_examples(root, checkpoint_path):
    plan = SmartLoader.inspect(checkpoint_path, auto_detect=True, precision="fp32")
    print("SmartLoader report:")
    print(plan.report)

    loaded = load_external_model(checkpoint_path)
    print(f"External loader source type: {loaded.source_type}")

    hf_like_dir = root / "hf_like_model"
    hf_like_dir.mkdir(exist_ok=True)
    (hf_like_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (hf_like_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_like_dir / "model.safetensors").write_bytes(b"")
    print(SmartLoader.inspect(hf_like_dir).report)


def config_and_tracking_examples(root, config):
    config_path = root / "config.json"
    save_config_json(config, str(config_path))
    loaded_config = load_config_json(str(config_path))
    print(f"Reloaded config block_size: {loaded_config.block_size}")

    tracker = create_experiment(
        "full_feature_guide",
        experiment_dir=str(root / "experiments"),
        tags={"example": "docs"},
    )
    tracker.log_config(config)
    tracker.log_metrics({"demo_metric": 1.0}, step=1)
    tracker.log_text("ArcLM full feature guide completed.", "notes.txt")
    tracker.end()


def main():
    torch.manual_seed(42)
    root = Path("runs/full_feature_guide")
    pretrain_path, qa_jsonl = write_demo_files(root)
    finetune_path = root / "data" / "finetune.txt"
    prepare_instruction_text(qa_jsonl, finetune_path)

    pretrained, finetuned = high_level_training(root, pretrain_path, finetune_path)
    inference_and_diagnostics(finetuned.model_path)
    low_level_config = lower_level_training(root, pretrain_path)
    loader_examples(root, pretrained.model_path)
    config_and_tracking_examples(root, low_level_config)

    print("Done. Artifacts are under runs/full_feature_guide/")


if __name__ == "__main__":
    main()
