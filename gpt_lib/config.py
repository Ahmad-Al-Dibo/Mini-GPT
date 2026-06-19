"""
Configuration Module - Centralized settings
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

import torch

class Config:
    """Configuration class for model and training"""
    
    def __init__(self, **kwargs):
        # Default values
        self.embed_dim = kwargs.get("embed_dim", 64)
        self.block_size = kwargs.get("block_size", 8)
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.dropout = kwargs.get("dropout", 0.0)
        self.grad_clip = kwargs.get("grad_clip", None)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.model_path = kwargs.get("model_path", "output/model.pth")
        self.data_path = kwargs.get("data_path", "data/data.txt")
        self.domain_data_path = kwargs.get("domain_data_path", None)
        self.domain_data_repeats = kwargs.get("domain_data_repeats", 1)
        self.tokenizer_type = kwargs.get("tokenizer_type", "word")
        self.sentencepiece_model_type = kwargs.get("sentencepiece_model_type", "bpe")
        self.sentencepiece_character_coverage = kwargs.get(
            "sentencepiece_character_coverage",
            1.0,
        )
        self.tokenizer_max_line_length = kwargs.get("tokenizer_max_line_length", 4000)
        self.max_vocab = kwargs.get("max_vocab", 50000)
        self.max_data_size = kwargs.get("max_data_size", 1000000)
        self.device = kwargs.get("device", "cpu")
        self.validation_split = kwargs.get("validation_split", 0.0)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", None)
        self.early_stopping_min_delta = kwargs.get("early_stopping_min_delta", 0.0)
        self.restore_best_model = kwargs.get("restore_best_model", True)
        self.seed = kwargs.get("seed", 42)
        self.diagnostic_top_k = kwargs.get("diagnostic_top_k", 5)
        self.concept_benchmark_top_k = kwargs.get("concept_benchmark_top_k", 10)
        self.diagnostic_prompts = kwargs.get(
            "diagnostic_prompts",
            ["machine learning", "donald trump"],
        )
        self.diagnostic_sample_tokens = kwargs.get("diagnostic_sample_tokens", 60)
        self.tokenizer_rare_threshold = kwargs.get("tokenizer_rare_threshold", 2)
        self.training_log_interval = kwargs.get("training_log_interval", 50)
        self.run_long_context_evaluation = kwargs.get("run_long_context_evaluation", False)
        self.use_checkpoint_tokenizer = kwargs.get("use_checkpoint_tokenizer", False)
        self.long_context_block_sizes = kwargs.get(
            "long_context_block_sizes",
            [32, 64, 128],
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "embed_dim": self.embed_dim,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "grad_clip": self.grad_clip,
            "num_blocks": self.num_blocks,
            "model_path": self.model_path,
            "data_path": self.data_path,
            "domain_data_path": self.domain_data_path,
            "domain_data_repeats": self.domain_data_repeats,
            "tokenizer_type": self.tokenizer_type,
            "sentencepiece_model_type": self.sentencepiece_model_type,
            "sentencepiece_character_coverage": self.sentencepiece_character_coverage,
            "tokenizer_max_line_length": self.tokenizer_max_line_length,
            "max_vocab": self.max_vocab,
            "max_data_size": self.max_data_size,
            "device": self.device,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "restore_best_model": self.restore_best_model,
            "seed": self.seed,
            "diagnostic_top_k": self.diagnostic_top_k,
            "concept_benchmark_top_k": self.concept_benchmark_top_k,
            "diagnostic_prompts": self.diagnostic_prompts,
            "diagnostic_sample_tokens": self.diagnostic_sample_tokens,
            "tokenizer_rare_threshold": self.tokenizer_rare_threshold,
            "training_log_interval": self.training_log_interval,
            "run_long_context_evaluation": self.run_long_context_evaluation,
            "use_checkpoint_tokenizer": self.use_checkpoint_tokenizer,
            "long_context_block_sizes": self.long_context_block_sizes,
        }


    def load_config_from_model(self, model_path):
        pass

    def get_device(self):
        return torch.device(self.device)
    
    def __repr__(self):
        items = "\n".join(f"  {k}: {v}" for k, v in self.to_dict().items())
        return f"Config(\n{items}\n)"




def create_config(**kwargs) -> Config:
    """
    Create a Config object with sensible defaults that can be overridden.

    Parameters
    ----------
    **kwargs
        Any configuration field supported by the Config class.

    Common Parameters
    -----------------
    embed_dim : int
        Transformer embedding dimension.
    num_blocks : int
        Number of transformer blocks.
    block_size : int
        Maximum context length in tokens.
    batch_size : int
        Training batch size.
    learning_rate : float
        Optimizer learning rate.
    dropout : float
        Dropout probability.
    max_vocab : int
        Maximum tokenizer vocabulary size.
    epochs : int
        Number of training epochs.

    Returns
    -------
    Config
        Initialized configuration object.

    Examples
    --------
    Create a default configuration:

    >>> cfg = create_config()

    Create a larger model:

    >>> cfg = create_config(
    ...     embed_dim=256,
    ...     num_blocks=6,
    ...     block_size=256
    ... )

    Customize training settings:

    >>> cfg = create_config(
    ...     batch_size=64,
    ...     learning_rate=1e-4,
    ...     epochs=20
    ... )

    Configure tokenizer settings:

    >>> cfg = create_config(
    ...     tokenizer_type="sentencepiece",
    ...     max_vocab=16000
    ... )

    Notes
    -----
    - Any keyword argument matching a Config field will override
      the default value.
    - Unknown arguments are forwarded directly to Config.
    - The device is automatically selected unless explicitly specified.
    - For multi-head attention models, embed_dim should generally be
      divisible by the number of attention heads.
    """
    defaults = {
        "embed_dim": 64,
        "block_size": 8,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "grad_clip": None,
        "num_blocks": 2,

        "model_path": "output/model.pth",
        "data_path": "data/data.txt",

        "domain_data_path": None,
        "domain_data_repeats": 1,

        "tokenizer_type": "word",
        "sentencepiece_model_type": "bpe",
        "sentencepiece_character_coverage": 1.0,
        "tokenizer_max_line_length": 4000,
        "max_vocab": 50000,
        "max_data_size": 1_000_000,
        "tokenizer_rare_threshold": 2,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "validation_split": 0.0,
        "early_stopping_patience": None,
        "early_stopping_min_delta": 0.0,
        "restore_best_model": True,

        "seed": 42,

        "diagnostic_top_k": 5,
        "concept_benchmark_top_k": 10,
        "diagnostic_prompts": [
            "machine learning",
            "donald trump",
        ],
        "diagnostic_sample_tokens": 60,

        "training_log_interval": 50,

        "run_long_context_evaluation": False,
        "use_checkpoint_tokenizer": False,
        "long_context_block_sizes": [32, 64, 128],
    }

    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown configuration parameters: {', '.join(sorted(unknown))}"
        )

    defaults.update(kwargs)

    torch.manual_seed(defaults.get("seed"))

    return Config(**defaults)