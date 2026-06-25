import sys
import os
import torch
import random
import time
from time import gmtime, strftime
from collections import Counter
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.miniGPT import (
    create_config,
    SentencePieceTokenizer,
    Generator,
    build_model,
    build_trainer,
    prepare_data,
    create_epoch_checkpoint_callback,
)

config = create_config(
     # ========== Model Architecture ==========
        embed_dim=84,
        num_blocks=4,
        block_size=94,
        dropout=0.1,

        # ========== Training Hyperparameters ==========
        batch_size=94,
        num_epochs=15,
        learning_rate=2.5e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        
        # ========== Data ==========
        data_path="alpaca-cleaned.txt",
        max_data_size=7_000_000,
        validation_split=0.1,
        
        # ========== Tokenizer ==========
        tokenizer_type="sentencepiece",
        sentencepiece_model_type="bpe",
        sentencepiece_character_coverage=1.0,
        max_vocab=24_000,
        tokenizer_rare_threshold=2,
        
        # ========== Training Callbacks ==========
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
        training_log_interval=50,
        
        # ========== Device & Misc ==========
        seed=42,
        device= "cuda" if torch.cuda.is_available() else "cpu",
        model_path="output/GPT-15k.pth",
        tokenizer_path="output/tokenizer.model",
)

with open(config.data_path, "r", encoding="utf-8") as f:
    data_text = f.read()

tokenizer = SentencePieceTokenizer(max_vocab=config.max_vocab)
if os.path.exists(config.tokenizer_path):
    print(f"Loading existing tokenizer from {config.tokenizer_path}...")
    tokenizer = tokenizer.load(config.tokenizer_path)
    if tokenizer.vocab_size != config.max_vocab:
        # exit and create a new tokenizer with the correct vocab size
        print(f"Existing tokenizer vocab size ({tokenizer.vocab_size}) does not match config max_vocab ({config.max_vocab}). Creating a new tokenizer.")
        tokenizer = SentencePieceTokenizer(max_vocab=config.max_vocab)
        tokenizer.build(data_text)
        tokenizer.save(config.tokenizer_path)
else:
    tokenizer.build(data_text)
    tokenizer.save(config.tokenizer_path)
config.vocab_size = tokenizer.vocab_size  # Set vocab size in config after tokenizer is built

print(config.to_dict())

data = prepare_data(config, existing_tokenizer=tokenizer)

print(f"✓ Data tokens: {len(data.tokens):,}")
print(f"✓ Train tokens: {len(data.train_tokens):,}")
print(f"✓ Validation tokens: {len(data.val_tokens):,}")
print(f"✓ Vocab size: {data.vocab_size}")
print(f"✓ Train batches: {len(data.train_loader)}")
if data.val_loader is not None:
    print(f"✓ Validation batches: {len(data.val_loader)}")

print(f"\n Data sample analysis:")
print(f"  Raw tokens (first 20): {data.train_encoded[:20]}")
print(f"  Decoded: {data.tokenizer.decode(data.train_encoded[:20])}")
counter = Counter(data.tokenizer.decode(data.train_encoded))
print(f"  Unique tokens: {len(counter)}")
print(f"  Top 5 tokens: {counter.most_common(5)}")

model = build_model(config)

params = model.get_num_parameters()
print(f"✓ Model parameters: {params:,}")

trainer = build_trainer(model, config)

main_start = time.time()
print(strftime("\n[%Y-%m-%d %H:%M:%S] Starting training...", gmtime()))

input("\nPress Enter to start training... (or Ctrl+C to cancel)")

checkpoint_callback = create_epoch_checkpoint_callback(
        config,
        data.tokenizer,
        data.vocab_size,
)

trainer.train(
    data.train_loader,
    config.num_epochs,
    val_loader= data.val_loader,
    early_stopping_patience=config.early_stopping_patience,
    min_delta=config.early_stopping_min_delta,
    checkpoint_callback=checkpoint_callback,
)

try:
    trainer.save(
        config=config,
        model_path=config.model_path,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        tokenizer_metadata=tokenizer.get_metadata(),
    )
except Exception as e:
    print(f"Error saving model: {e}")
    trainer.save(config=config)
    sys.exit(1) # Exit with a non-zero status code to indicate failure


total_time = time.time() - main_start
print(strftime(f"\n[%Y-%m-%d %H:%M:%S] Training completed in {total_time:.2f} seconds.", gmtime()))