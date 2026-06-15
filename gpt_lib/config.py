"""
Configuration Module - Centralized settings
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json


# @dataclass
# class Config:
#     # Model
#     embed_dim: int = 64
#     block_size: int = 8
#     batch_size: int = 64
#     epochs: int = 100
#     learning_rate: float = 1e-3
#     weight_decay: float = 0.0
#     dropout: float = 0.0
#     grad_clip: float | None = None
#     num_blocks: int = 2

#     # Paths
#     model_path: str = "output/model.pth"
#     data_path: str = "data/data.txt"
#     domain_data_path: str | None = None
#     domain_data_repeats: int = 1

#     # Tokenizer
#     tokenizer_type: str = "word"
#     sentencepiece_model_type: str = "bpe"
#     sentencepiece_character_coverage: float = 1.0
#     tokenizer_max_line_length: int = 4000
#     tokenizer_rare_threshold: int = 2

#     # Dataset
#     max_vocab: int = 50000
#     max_data_size: int = 1000000

#     # Training
#     device: str = "cpu"
#     validation_split: float = 0.0
#     early_stopping_patience: int | None = None
#     early_stopping_min_delta: float = 0.0
#     restore_best_model: bool = True
#     seed: int = 42
#     training_log_interval: int = 50

#     # Diagnostics
#     diagnostic_top_k: int = 5
#     concept_benchmark_top_k: int = 10
#     diagnostic_prompts: list[str] = field(
#         default_factory=lambda: [
#             "machine learning",
#             "donald trump siad",
#         ]
#     )
#     diagnostic_sample_tokens: int = 60

#     # Long context evaluation
#     run_long_context_evaluation: bool = False
#     long_context_block_sizes: list[int] = field(
#         default_factory=lambda: [32, 64, 128]
#     )

#     def to_dict(self) -> dict:
#         """Convert config to dictionary with very simple function asdict"""
#         return {"config": asdict(self)}

#     def save(self, filename: str) -> None:
#         """Save config to JSON file"""
#         filename = Path(filename)

#         if filename.parent:
#             filename.parent.mkdir(parents=True, exist_ok=True)

#         with open(filename, "w", encoding="utf-8") as f:
#             json.dump(
#                 self.to_dict(),
#                 f,
#                 indent=4,
#                 ensure_ascii=False,
#             )

#     @classmethod
#     def load(cls, filename: str) -> "Config":
#         """Load config from JSON file"""
#         filename = Path(filename)

#         if not filename.exists():
#             raise FileNotFoundError(
#                 f"Config file not found: {filename}"
#             )

#         with open(filename, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         config_data = data.get("config", {})

#         return cls(**config_data)

#     def update(self, **kwargs) -> None:
#         """Update configuration values"""
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#             else:
#                 raise AttributeError(
#                     f"Unknown configuration option: {key}"
#                 )

#     def __repr__(self) -> str:
#         items = "\n".join(
#             f"  {k}: {v}"
#             for k, v in asdict(self).items()
#         )
#         return f"Config(\n{items}\n)"


"""
Configuration Module - Centralized settings
"""

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
            "long_context_block_sizes": self.long_context_block_sizes,
        }


    def load_model_conig(self, model_path):
        pass
    
    def __repr__(self):
        items = "\n".join(f"  {k}: {v}" for k, v in self.to_dict().items())
        return f"Config(\n{items}\n)"
