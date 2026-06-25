"""
Model construction and training pipeline helpers.
"""

import torch
import torch.nn as nn

from .model import ArcLM
from .trainer import Trainer


def build_model(config, vocab_size=None) -> ArcLM:
    """Build an ArcLM model from config and move it to the configured device."""
    model = ArcLM(
        vocab_size=vocab_size if vocab_size is not None else config.vocab_size,
        embed_dim=config.embed_dim,
        block_size=config.block_size,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    )
    return model.to(torch.device(config.device))


def build_optimizer_with_discriminative_lr(
    model,
    learning_rate: float,
    weight_decay: float,
    lr_multiplier: dict = None
) -> torch.optim.Optimizer:
    """Build optimizer with different LR for different layer types.
    
    Args:
        model: ArcLM model
        learning_rate: Base learning rate
        weight_decay: Weight decay
        lr_multiplier: Dict like {'embeddings': 0.1, 'blocks': 0.1, 'head': 1.0}
                      Multiplies base LR for each group
    
    Returns:
        AdamW optimizer with param groups
    """
    if lr_multiplier is None:
        lr_multiplier = {
            'embeddings': 0.1,
            'blocks': 0.1,
            'head': 1.0
        }
    
    param_groups = []
    assigned_params_names = set()
    
    # Group 1: Embeddings
    embedding_params = []
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append(param)
            assigned_params_names.add(name)
    
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': learning_rate * lr_multiplier.get('embeddings', 1.0),
            'weight_decay': weight_decay,
        })
    
    # Group 2: Transformer blocks
    block_params = []
    for name, param in model.named_parameters():
        if 'blocks' in name:
            block_params.append(param)
            assigned_params_names.add(name)
    
    if block_params:
        param_groups.append({
            'params': block_params,
            'lr': learning_rate * lr_multiplier.get('blocks', 1.0),
            'weight_decay': weight_decay,
        })
    
    # Group 3: Output head (anything not in embeddings or blocks)
    head_params = []
    for name, param in model.named_parameters():
        if name not in assigned_params_names:
            head_params.append(param)
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': learning_rate * lr_multiplier.get('head', 1.0),
            'weight_decay': weight_decay,
        })
    
    return torch.optim.AdamW(param_groups, lr=learning_rate)


def build_trainer(model, config):
    """Create the optimizer, criterion, and Trainer for a model."""
    # Check if using discriminative learning rates for finetuning
    if hasattr(config, 'use_discriminative_lr') and config.use_discriminative_lr:
        lr_multiplier = getattr(config, 'lr_multiplier', None)
        if lr_multiplier is None:
            lr_multiplier = {
                'embeddings': 0.1,
                'blocks': 0.1,
                'head': 1.0
            }
        optimizer = build_optimizer_with_discriminative_lr(
            model,
            config.learning_rate,
            config.weight_decay,
            lr_multiplier
        )
        logger = __import__('logging').getLogger(__name__)
        logger.info("✓ Using discriminative learning rates:")
        logger.info(f"  Embeddings: {config.learning_rate * lr_multiplier.get('embeddings', 1.0):.2e}")
        logger.info(f"  Blocks: {config.learning_rate * lr_multiplier.get('blocks', 1.0):.2e}")
        logger.info(f"  Head: {config.learning_rate * lr_multiplier.get('head', 1.0):.2e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    criterion = nn.CrossEntropyLoss()
    return Trainer(model, optimizer, criterion, config)


def checkpoint_is_compatible_for_continue_training(checkpoint, config, vocab_size, tokenizer=None):
    """Check whether a checkpoint can be reused for the current run."""
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config == {}:
        print("[WARNING] in `checkpoint_is_compatible` checkpoint config return {}! ")
    same_vocab = checkpoint.get("vocab_size") == vocab_size
    same_shape = (
        checkpoint_config.get("embed_dim", config.embed_dim) == config.embed_dim
        and checkpoint_config.get("block_size", config.block_size) == config.block_size
        and checkpoint_config.get("num_blocks", config.num_blocks) == config.num_blocks
    )
    same_tokenizer = (
        checkpoint_config.get("tokenizer_type", "word")
        == getattr(config, "tokenizer_type", "word")
        and checkpoint_config.get("sentencepiece_model_type", "bpe")
        == getattr(config, "sentencepiece_model_type", "bpe")
    )
    if tokenizer is not None:
        checkpoint_stoi = checkpoint.get("stoi")
        checkpoint_vocab = checkpoint.get("vocab")
        if checkpoint_stoi is not None:
            same_tokenizer = same_tokenizer and checkpoint_stoi == tokenizer.stoi
        elif checkpoint_vocab is not None:
            same_tokenizer = same_tokenizer and checkpoint_vocab == tokenizer.vocab
        else:
            same_tokenizer = False

    same_training_strategy = (
        checkpoint_config.get("learning_rate", config.learning_rate) == config.learning_rate
        and checkpoint_config.get("weight_decay", config.weight_decay) == config.weight_decay
        and checkpoint_config.get("dropout", 0.0) == config.dropout
        and checkpoint_config.get("validation_split", config.validation_split)
        == config.validation_split
    )
    
    print("same_vocab, same_shape, same_tokenizer, same_training_strategy", same_vocab, same_shape, same_tokenizer, same_training_strategy)
    # return same_vocab and same_shape and same_tokenizer and same_training_strategy
    return same_vocab and same_shape and same_tokenizer and same_training_strategy


def checkpoint_is_compatible_for_tuining(checkpoint, config, vocab_size, tokenizer=None):
    """Check whether a checkpoint can be reused for the current run."""
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config == {}:
        print("[WARNING] in `checkpoint_is_compatible` checkpoint config return {}! ")
    same_vocab = checkpoint.get("vocab_size") == vocab_size
    same_shape = (
        checkpoint_config.get("embed_dim", config.embed_dim) == config.embed_dim
        and checkpoint_config.get("block_size", config.block_size) == config.block_size
        and checkpoint_config.get("num_blocks", config.num_blocks) == config.num_blocks
    )
    same_tokenizer = (
        checkpoint_config.get("tokenizer_type", "word")
        == getattr(config, "tokenizer_type", "word")
        and checkpoint_config.get("sentencepiece_model_type", "bpe")
        == getattr(config, "sentencepiece_model_type", "bpe")
    )
    if tokenizer is not None:
        checkpoint_stoi = checkpoint.get("stoi")
        checkpoint_vocab = checkpoint.get("vocab")
        if checkpoint_stoi is not None:
            same_tokenizer = same_tokenizer and checkpoint_stoi == tokenizer.stoi
        elif checkpoint_vocab is not None:
            same_tokenizer = same_tokenizer and checkpoint_vocab == tokenizer.vocab
        else:
            same_tokenizer = False

    # same_training_strategy = (
    #     checkpoint_config.get("learning_rate", config.learning_rate) == config.learning_rate
    #     and checkpoint_config.get("weight_decay", config.weight_decay) == config.weight_decay
    #     and checkpoint_config.get("dropout", 0.0) == config.dropout
    #     and checkpoint_config.get("validation_split", config.validation_split)
    #     == config.validation_split
    # )
    same_training_strategy = (
        checkpoint_config.get("weight_decay", config.weight_decay) == config.weight_decay
        and checkpoint_config.get("dropout", 0.0) == config.dropout
        and checkpoint_config.get("validation_split", config.validation_split)
        == config.validation_split
    )
    print("same_vocab, same_shape, same_tokenizer, same_training_strategy", same_vocab, same_shape, same_tokenizer, same_training_strategy)
    # return same_vocab and same_shape and same_tokenizer and same_training_strategy
    return same_shape and same_tokenizer and same_training_strategy





def load_compatible_checkpoint(trainer, config, vocab_size, tokenizer=None):
    """Load an existing checkpoint when it matches the current model setup."""
    if not trainer.exists(config.model_path):
        return False

    checkpoint = torch.load(config.model_path, map_location=torch.device(config.device))
    if checkpoint_is_compatible(checkpoint, config, vocab_size, tokenizer=tokenizer):
        trainer.load(config.model_path)
        return True

    print("[WARNING] Existing checkpoint is incompatible; retraining.")
    print(f"  Checkpoint vocab: {checkpoint.get('vocab_size')}, current vocab: {vocab_size}")
    print(f"  Checkpoint config: {checkpoint.get('config', {})}")
    if tokenizer is not None:
        print("  Tokenizer mapping differs or is missing; retraining is required.")
    return False


def create_epoch_checkpoint_callback(config, tokenizer, vocab_size):
    """Create a callback that saves the latest resumable checkpoint each epoch."""
    def checkpoint_callback(trainer):
        save_training_checkpoint(trainer, config, tokenizer, vocab_size)

    return checkpoint_callback


def save_training_checkpoint(trainer, config, tokenizer, vocab_size):
    """Save a trainer checkpoint with tokenizer metadata."""
    tokenizer_metadata = None
    if hasattr(tokenizer, "to_checkpoint"):
        tokenizer_metadata = tokenizer.to_checkpoint()

    trainer.save(
        config.model_path,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        block_size=config.block_size,
        vocab_size=vocab_size,
        tokenizer_metadata=tokenizer_metadata,
    )
