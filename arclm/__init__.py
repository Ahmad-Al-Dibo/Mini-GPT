"""ArcLM - compact PyTorch language-model training and fine-tuning."""

import torch

from .config import Config, create_config
from .data import DataBundle, load_tokens, prepare_data, read_tokens, split_train_val
from .dataset import TextDataset, create_dataloader
from .diagnostics import (
    ConceptBenchmarkCase,
    ConceptBenchmarkResult,
    DEFAULT_CONCEPT_BENCHMARKS,
    LongContextResult,
    MetricsReport,
    TopKPrediction,
    build_training_diagnostics_report,
    calculate_metrics,
    calculate_perplexity,
    export_metrics_to_json,
    export_metrics_to_markdown,
    format_concept_benchmark_report,
    format_long_context_results,
    format_tokenizer_coverage_report,
    format_top_k_predictions,
    predict_top_k,
    run_long_context_evaluation,
    score_concept_relationships,
)
from .generator import Generator
from .inference import DEFAULT_MODEL_PATH, LoadedModel, load_model, predict
from .instruction_dataset import InstructionDataset, create_instruction_dataloader
from .logics import (
    And,
    Biconditional,
    Implication,
    Not,
    Or,
    Sentence,
    Symbol,
    model_check,
)
from .model import ArcLM, MiniGPT
from .pipeline import (
    build_model,
    build_trainer,
    checkpoint_is_compatible_for_continue_training,
    checkpoint_is_compatible_for_tuining,
    create_epoch_checkpoint_callback,
    load_compatible_checkpoint,
    save_training_checkpoint,
)
from .pipeline_v2 import ModelAdapter, PreTrainedModelLoader, StoppingCriteria, UnifiedPipeline
from .regularization import (
    EarlyStopping,
    GeneralizationMonitor,
    L1Regularization,
    L2Regularization,
    LabelSmoothing,
    LearningRateScheduler,
    MixupAugmentation,
)
from .tokenizer import SentencePieceTokenizer, Tokenizer, create_tokenizer, get_tokenizer_from_config
from .trainer import Trainer
from .utils import format_duration

__version__ = "0.1.0"
__author__ = "ArcLM Contributors"

__all__ = [
    "ArcLM",
    "MiniGPT",
    "Config",
    "create_config",
    "Tokenizer",
    "SentencePieceTokenizer",
    "create_tokenizer",
    "get_tokenizer_from_config",
    "TextDataset",
    "create_dataloader",
    "DataBundle",
    "load_tokens",
    "prepare_data",
    "read_tokens",
    "split_train_val",
    "Trainer",
    "Generator",
    "DEFAULT_MODEL_PATH",
    "LoadedModel",
    "load_model",
    "predict",
    "build_model",
    "build_trainer",
    "checkpoint_is_compatible_for_continue_training",
    "checkpoint_is_compatible_for_tuining",
    "create_epoch_checkpoint_callback",
    "load_compatible_checkpoint",
    "save_training_checkpoint",
    "UnifiedPipeline",
    "PreTrainedModelLoader",
    "ModelAdapter",
    "StoppingCriteria",
    "build_training_diagnostics_report",
    "calculate_metrics",
    "calculate_perplexity",
    "export_metrics_to_json",
    "export_metrics_to_markdown",
    "ConceptBenchmarkCase",
    "ConceptBenchmarkResult",
    "DEFAULT_CONCEPT_BENCHMARKS",
    "LongContextResult",
    "MetricsReport",
    "TopKPrediction",
    "format_concept_benchmark_report",
    "format_long_context_results",
    "format_tokenizer_coverage_report",
    "format_top_k_predictions",
    "predict_top_k",
    "run_long_context_evaluation",
    "score_concept_relationships",
    "format_duration",
    "L1Regularization",
    "L2Regularization",
    "EarlyStopping",
    "LearningRateScheduler",
    "GeneralizationMonitor",
    "MixupAugmentation",
    "LabelSmoothing",
    "create_instruction_dataloader",
    "InstructionDataset",
    "Sentence",
    "Symbol",
    "Not",
    "And",
    "Or",
    "Implication",
    "Biconditional",
    "model_check",
]


def get_version():
    return __version__


def list_available_models():
    from pathlib import Path

    models_dir = Path("models")
    if not models_dir.exists():
        return []
    return sorted(path.stem for path in models_dir.glob("*.pth") if path.is_file())


def load_model_checkpoint(path: str, device_type: str):
    return torch.load(path, map_location=torch.device(device_type))
