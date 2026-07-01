"""ArcLM - compact PyTorch language-model training and fine-tuning."""

import torch

from ._version import __version__
from .config import Config, create_config
from .data import DataBundle, load_tokens, prepare_data, read_tokens, split_train_val
from .data_processor import DataProcessor, ProcessedDataset
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
from .loaders import (
    AdaptedModelBundle,
    LoadPlan,
    LoadedCheckpoint,
    ModelInspector,
    SmartLoader,
    adapt_for_training,
    inspect_model_source,
    load_external_model,
    register_model_inspector,
    validate_tokenizer_compatibility,
)
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
    create_checkpoint_callback,
    create_epoch_checkpoint_callback,
    load_compatible_checkpoint,
    save_training_checkpoint,
    train_model,
    TrainingResult,
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

__author__ = "Ahmad_Al_Dibo"


"""
Add this to the education documentation:

| Task                  | Input                                     | Label (Target Output)           |
| --------------------- | ----------------------------------------- | ------------------------------- |
| Translation           | "Hello"                                   | "Bonjour"                       |
| Summarization         | Long article                              | Short summary                   |
| Question Answering    | "What is AI?"                             | "Artificial intelligence is..." |
| Chatbot               | User prompt                               | Assistant response              |
| Code Generation       | "Write a Python function to sort a list." | Correct Python code             |
| Instruction Following | "Explain photosynthesis."                 | High-quality explanation        |



"""


__types_of_supervised_fine_tuning = [
    "What is Supervised Fine-Tuning (SFT)?",
    "How Supervised Fine-Tuning Works",
    {
        "Core Concepts": [
            "Adapting to the Task at Hand",
            "The Role of Labeled Data",
            "Fine-Tuning the Model",
        ]
    },
    {
        "Types of Supervised Fine-Tuning": [
            "Full Fine-Tuning",
            "Parameter-Efficient Fine-Tuning (PEFT)",
            "Instruction Fine-Tuning",
        ]
    },
    {
        "Supervised Fine-Tuning Process": [
            "Step 1: Task Definition and Model Selection",
            "Step 2: Data Preparation",
            "Step 3: Dataset Tokenization",
            "Step 4: Fine-Tuning the Language Model",
            "Step 5: Hyperparameter Tuning",
            "Step 6: Evaluation and Deployment",
        ]
    },
    {
        "Common Fine-Tuning Techniques": [
            "Hyperparameter Adjustment",
            "Transfer Learning",
            "Multi-Task Training",
            "Few-Shot Adaptation",
            "Task-Specific Optimization",
            "Low-Rank Adaptation (LoRA)",
            "Other Parameter-Efficient Methods",
        ]
    },
    {
        "Advanced Alignment Methods": [
            "Reward-Based Learning",
            "Proximal Policy Optimization (PPO)",
            "Comparative Ranking",
        ]
    },
    {
        "Benefits of Supervised Fine-Tuning": [
            "Improved Performance",
            "Efficiency with Limited Data",
            "Versatility and Flexibility",
            "Cost Efficiency",
            "Customization",
            "Reduced Overfitting",
        ]
    },
    {
        "Challenges of Supervised Fine-Tuning": [
            "Overfitting",
            "Hyperparameter Tuning",
            "Data Quality Issues",
            "Catastrophic Forgetting",
            "Inconsistent Performance",
            "Time and Resource Intensive",
            "Requires Expertise",
            "Risk of Overwriting Pre-trained Knowledge",
            "Difficulty Selecting the Right Objective",
            "Difficulty Selecting the Right Training Data",
        ]
    },
]


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
    "DataProcessor",
    "ProcessedDataset",
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
    "train_model",
    "TrainingResult",
    "checkpoint_is_compatible_for_continue_training",
    "checkpoint_is_compatible_for_tuining",
    "create_checkpoint_callback",
    "create_epoch_checkpoint_callback",
    "load_compatible_checkpoint",
    "save_training_checkpoint",
    "UnifiedPipeline",
    "PreTrainedModelLoader",
    "ModelAdapter",
    "StoppingCriteria",
    "LoadedCheckpoint",
    "LoadPlan",
    "AdaptedModelBundle",
    "ModelInspector",
    "SmartLoader",
    "inspect_model_source",
    "register_model_inspector",
    "load_external_model",
    "adapt_for_training",
    "validate_tokenizer_compatibility",
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
