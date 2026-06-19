import torch
import os
import sys
import json


project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

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
    SentencePieceTokenizer
)
def print_data_summary(data):
    print(f"Data tokens: {len(data.tokens)}")
    print(f"Train tokens: {len(data.train_tokens)}")
    print(f"Validation tokens: {len(data.val_tokens)}")
    print(f"Vocab size: {data.vocab_size}")
    print(f"Train batches: {len(data.train_loader)}")
    if data.val_loader is not None:
        print(f"Validation batches: {len(data.val_loader)}")
old_model_settings = {
    'embed_dim': 64, 
    'block_size': 128, 
    'batch_size': 32, 
    'epochs': 10, 
    'learning_rate': 0.00025, 
    'weight_decay': 0.1, 
    'dropout': 0.1, 
    'grad_clip': 1.0, 
    'num_blocks': 3, 
    'domain_data_path': "../data/data.txt", 
    'domain_data_repeats': 3, 
    'tokenizer_type': 'sentencepiece', 
    'sentencepiece_model_type': 'bpe', 
    'sentencepiece_character_coverage': 1.0, 
    'max_vocab': 10000, 
    'max_data_size': 1000000, 
    'device': 'cpu', 
    'validation_split': 0.1, 
    'early_stopping_patience': 2, 
    'early_stopping_min_delta': 0.0001, 
    'restore_best_model': True, 
    'seed': 42, 
    'diagnostic_top_k': 5, 
    'concept_benchmark_top_k': 10, 
    'diagnostic_sample_tokens': 60, 
    'tokenizer_rare_threshold': 2, 
    'training_log_interval': 50, 
    'run_long_context_evaluation': False,
    'long_context_block_sizes': [32, 64, 128]
    }

cfg = create_config(
    **old_model_settings,
    model_path = "../output/mini_gpt-fine_2.pth",
    data_path = "../data/bookcorpus_diverse.txt",
    diagnostic_prompts = ["python is a ", "javascript is a"],
    use_checkpoint_tokenizer=True
)
cfg.to_dict()

with open("tokenizer.json", "r", encoding="utf-8") as f:
    text = f.read()

tkz_data = json.loads(text)
old_tkz = SentencePieceTokenizer.from_json(tkz_data)
data = prepare_data(cfg, old_tokenizer= old_tkz)
print_data_summary(data)
# tokenizer = data.tokenizer.to_json()
model = build_model(cfg, data.vocab_size)
print(f"Parameters: {model.get_num_parameters():,}")
trainer = build_trainer(model, cfg)
checkpoint_loaded = load_compatible_checkpoint(trainer, cfg, data.vocab_size)
checkpoint_loaded
if not checkpoint_loaded:
        print("\nNo improved checkpoint found; starting training.")
checkpoint_callback = create_epoch_checkpoint_callback(
    cfg,
    data.tokenizer,
    data.vocab_size,
)
trainer.train(
    data.train_loader,
    cfg.epochs,
    val_loader=data.val_loader,
    early_stopping_patience=cfg.early_stopping_patience,
    checkpoint_callback=checkpoint_callback,
)
save_training_checkpoint(trainer, cfg, data.tokenizer, data.vocab_size)
# tokenizer_jsn = json.dumps(tokenizer)
# with open("tokenizer.json", "w") as f:
#     f.write(tokenizer_jsn)
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
print("\nGenerated sample:")
print(generated_text)
print()
print(build_training_diagnostics_report(model, data, cfg))
if cfg.run_long_context_evaluation:
    results = run_long_context_evaluation(
        cfg,
        block_sizes=cfg.long_context_block_sizes,
        sample_prompt="Programming is",
    )
    print()
    print(format_long_context_results(results))