"""
ArcLM public API.

Example:
    >>> from arclm import Config, UnifiedPipeline, load_model
    >>> config = Config(embed_dim=128, learning_rate=1e-3)
    >>> pipeline = UnifiedPipeline(config, mode="pre_training")
    >>> pipeline.build(vocab_size=10000)
    >>> results = pipeline.train(train_loader, num_epochs=5)
"""

__version__ = "0.1.0"
__author__ = "ArcLM Contributors"
__all__ = [
    # Core Configuration
    "Config",
    
    # Training API (P4-1)
    "UnifiedPipeline",
    "StoppingCriteria",
    "PreTrainedModelLoader",
    "ModelAdapter",
    
    # Tokenizers (P4-2)
    "create_tokenizer",
    "get_tokenizer_from_config",
    "Tokenizer",
    "SentencePieceTokenizer",
    
    # Metrics & Evaluation (P4-3)
    "calculate_metrics",
    "calculate_perplexity",
    "export_metrics_to_json",
    "export_metrics_to_markdown",
    "MetricsReport",
    
    # Model Loading (Inference)
    "load_model",
    "LoadedModel",
    "predict",
    
    # Advanced: Config Files & Experiment Tracking (P4-4 Advanced)
    "load_config",
    "load_config_yaml",
    "load_config_json",
    "save_config",
    "save_config_yaml",
    "save_config_json",
    "ExperimentTracker",
    "create_experiment",
    "list_experiments",
    
    # Utilities
    "create_config",
]

# Lazy imports for faster module loading
def __getattr__(name):
    """Lazy load modules on first use"""
    if name == "Config":
        from .config import Config
        return Config
    elif name == "create_config":
        from .config import create_config
        return create_config
    elif name == "UnifiedPipeline":
        from .pipeline_v2 import UnifiedPipeline
        return UnifiedPipeline
    elif name == "StoppingCriteria":
        from .pipeline_v2 import StoppingCriteria
        return StoppingCriteria
    elif name == "PreTrainedModelLoader":
        from .pipeline_v2 import PreTrainedModelLoader
        return PreTrainedModelLoader
    elif name == "ModelAdapter":
        from .pipeline_v2 import ModelAdapter
        return ModelAdapter
    elif name == "create_tokenizer":
        from .tokenizer import create_tokenizer
        return create_tokenizer
    elif name == "get_tokenizer_from_config":
        from .tokenizer import get_tokenizer_from_config
        return get_tokenizer_from_config
    elif name == "Tokenizer":
        from .tokenizer import Tokenizer
        return Tokenizer
    elif name == "SentencePieceTokenizer":
        from .tokenizer import SentencePieceTokenizer
        return SentencePieceTokenizer
    elif name == "calculate_metrics":
        from .diagnostics import calculate_metrics
        return calculate_metrics
    elif name == "calculate_perplexity":
        from .diagnostics import calculate_perplexity
        return calculate_perplexity
    elif name == "export_metrics_to_json":
        from .diagnostics import export_metrics_to_json
        return export_metrics_to_json
    elif name == "export_metrics_to_markdown":
        from .diagnostics import export_metrics_to_markdown
        return export_metrics_to_markdown
    elif name == "MetricsReport":
        from .diagnostics import MetricsReport
        return MetricsReport
    elif name == "load_model":
        from .inference import load_model
        return load_model
    elif name == "LoadedModel":
        from .inference import LoadedModel
        return LoadedModel
    elif name == "predict":
        from .inference import predict
        return predict
    elif name == "load_config":
        from .config_loader import load_config
        return load_config
    elif name == "load_config_yaml":
        from .config_loader import load_config_yaml
        return load_config_yaml
    elif name == "load_config_json":
        from .config_loader import load_config_json
        return load_config_json
    elif name == "save_config":
        from .config_loader import save_config
        return save_config
    elif name == "save_config_yaml":
        from .config_loader import save_config_yaml
        return save_config_yaml
    elif name == "save_config_json":
        from .config_loader import save_config_json
        return save_config_json
    elif name == "ExperimentTracker":
        from .tracking import ExperimentTracker
        return ExperimentTracker
    elif name == "create_experiment":
        from .tracking import create_experiment
        return create_experiment
    elif name == "list_experiments":
        from .tracking import list_experiments
        return list_experiments
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_version():
    """Get library version"""
    return __version__


def list_available_models():
    """
    List available pre-trained model checkpoints.
    
    Returns:
        list: Available model names
    """
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        return []
    
    models = [
        f.stem for f in models_dir.glob("*.pth") 
        if f.is_file()
    ]
    return sorted(models)
