# ArcLM Project Map

[TECH_STACK]

- Language: Python.
- Package type: installable Python library.
- Runtime target in package metadata: Python `>=3.9`.
- Core runtime dependencies:
  - `torch>=2.1,<3`
  - `numpy>=1.24,<3`
  - `sentencepiece>=0.2,<0.3`
  - `transformers>=4.38,<6`
- Optional web dependency:
  - `flask>=3,<4`
- Development dependencies:
  - `pytest>=8`
  - `build>=1.2,<2`
  - `twine>=5,<7`
- Optional external source support:
  - `safetensors` is loaded only when installed and when a `.safetensors`
    source is requested.
- Packaging:
  - Build backend: `setuptools.build_meta`.
  - Version source: `arclm._version.__version__`.
  - License: Apache License 2.0.
- Dependency verification:
  - Date checked from shell: `2026-06`.
  - Latest PyPI versions observed in that check:
    - `torch==2.12.1`, requires Python `>=3.10`
    - `numpy==2.5.0`, requires Python `>=3.12`
    - `sentencepiece==0.2.1`, requires Python `>=3.9`
    - `transformers==5.12.1`, requires Python `>=3.10.0`
    - `pytest==9.1.1`, requires Python `>=3.10`
    - `build==1.5.0`, requires Python `>=3.10`
    - `twine==6.2.0`, requires Python `>=3.9`
    - `flask==3.1.3`, requires Python `>=3.9`
    - `safetensors==0.8.0`, requires Python `>=3.10`

[SYSTEM_FLOW]

Primary library dataflow:

1. A caller provides a local text file path and training configuration.
2. `arclm.data.prepare_data()` reads text, tokenizes it, builds or reuses the
   tokenizer, creates train/validation splits, and returns dataloaders in a
   `DataBundle`.
3. `arclm.pipeline.build_model()` creates an `ArcLM` causal language model from
   the active `Config`.
4. `arclm.pipeline.build_trainer()` creates the optimizer, loss function, and
   `Trainer`.
5. `Trainer.train()` runs pretraining, fine-tuning, or continued training.
6. `AsyncTrainingLogger` can write training events and metrics to JSONL without
   blocking the training loop on file I/O.
7. `Trainer.save()` writes a checkpoint containing model weights, optimizer
   state, config, history, vocabulary, token maps, and tokenizer metadata.
8. `arclm.inference.load_model()` restores a checkpoint for inference.
9. `LoadedModel.predict()` generates text from a prompt.

External checkpoint dataflow:

1. A caller passes a source to `load_external_model()`.
2. `LoaderRegistry` selects the first loader that supports the source.
3. The selected loader returns a normalized `LoadedCheckpoint`.
4. `adapt_for_training()` builds an ArcLM-compatible model bundle from the
   normalized checkpoint and target config.
5. Fine-tuning or continued training proceeds through the same `Trainer` flow.

Smart loading dataflow:

1. A caller passes a model folder, checkpoint file, or Hugging Face model ID to
   `SmartLoader.inspect()` or `inspect_model_source()`.
2. `ModelInspector` adapters inspect available files and metadata such as
   `config.json`, tokenizer files, weight files, adapter files, optimizer state,
   scheduler state, and trainer state.
3. Detected settings are merged with explicit user overrides, with user values
   taking priority.
4. The resulting `LoadPlan` can produce a load report, or `SmartLoader.load()`
   can continue through the existing external loader registry and attach the
   plan to checkpoint metadata.

Flexible preprocessing dataflow:

1. A caller loads JSON, JSONL, CSV, TXT, or custom data through
   `DataProcessor.load()`.
2. `ProcessedDataset` applies cleaning, filtering, prompt/template
   transformation, optional tokenization, splitting, and batch mapping.
3. The processed text or token samples can be passed into training workflows.

Optional Flask API flow:

1. `app.py` reads `MODEL_PATH`, defaulting to `models/arclm.pth`.
2. The first prediction request loads the checkpoint with `load_model()`.
3. `/predict` returns one generated token.
4. `/generate` repeatedly predicts one token until the configured limit.
5. `/health` reports process status and whether the model is loaded.

Verifiable goals:

- `python -m pytest tests` passes.
- Tokenizer round-trip preserves known tokens and maps unknown tokens to
  `<UNK>`.
- User-defined tokenizer symbols are reserved as real tokens for word
  tokenizers and passed through to SentencePiece training.
- Model forward pass returns logits shaped as `(batch, block_size, vocab_size)`.
- Training writes a checkpoint to the configured output path.
- Batch checkpointing writes the expected global step.
- External raw state dict loading adapts into a trainable ArcLM model.
- Native ArcLM checkpoint loading preserves tokenizer compatibility checks.
- Smart model inspection detects local Hugging Face-style metadata, adapter
  files, weight format, precision, resume state, and manual overrides.
- DataProcessor transforms JSONL/TXT data into prompt text, tokenizes, filters,
  cleans, and splits samples.
- High-level `train_model()` writes a checkpoint and JSONL metrics.
- `examples/full_feature_guide.py` runs end-to-end with generated tiny local
  data and covers preprocessing, training, fine-tuning, inference, diagnostics,
  loaders, config persistence, and local tracking.

[ARCHITECTURE]

- `arclm/__init__.py`: public exports and package metadata helpers.
- `arclm/_version.py`: package version.
- `arclm/config.py`: training and model configuration dataclass.
- `arclm/config_loader.py`: JSON/YAML config loading and saving helpers.
- `arclm/tokenizer.py`: word tokenizer and SentencePiece tokenizer, including
  `user_defined_symbols` support for fine-tuning and instruction prompts.
- `arclm/data.py`: data reading, tokenizer reuse, split creation, dataloader
  preparation.
- `arclm/data_processor.py`: flexible JSON/JSONL/CSV/TXT/custom dataset loading,
  prompt/template transformation, optional tokenization, cleaning, filtering,
  splitting, and batch preprocessing.
- `arclm/dataset.py`: sliding-window text dataset and dataloader factory.
- `arclm/model.py`: `ArcLM` transformer model; `MiniGPT` compatibility alias.
- `arclm/trainer.py`: training loop, validation, checkpoint persistence, layer
  freezing.
- `arclm/pipeline.py`: supported high-level and lower-level training pipeline.
- `arclm/pipeline_v2.py`: alternate unified pipeline surface retained in the
  package exports.
- `arclm/inference.py`: checkpoint loading and inference wrapper.
- `arclm/generator.py`: autoregressive token generation.
- `arclm/logging.py`: queue-backed asynchronous training event logger.
- `arclm/diagnostics.py`: metrics, perplexity, top-k prediction, tokenizer
  coverage, and benchmark helpers.
- `arclm/regularization.py`: optional regularization and training helpers.
- `arclm/instruction_dataset.py`: instruction/response dataset helpers.
- `arclm/tracking.py`: experiment metadata and local experiment tracking.
- `arclm/logics/`: propositional logic classes and model checking helpers.
- `arclm/loaders/`: external source loaders and adaptation layer.
  - `registry.py`: ordered loader selection.
  - `smart_loader.py`: source inspection, `LoadPlan` reports, manual override
    merging, and custom `ModelInspector` registration.
  - `arclm_checkpoint.py`: ArcLM checkpoint loader.
  - `state_dict_loader.py`: raw PyTorch state dict loader.
  - `safetensors_loader.py`: safetensors loader when dependency is installed.
  - `hf_loader.py`: Hugging Face causal LM loader.
  - `adapters.py`: conversion from normalized checkpoint to ArcLM model bundle.
- `examples/`: executable examples for preparation, pretraining, fine-tuning,
  external loading, and broad feature walkthroughs.
- `docs/USAGE.md`: short usage guide.
- `docs/FULL_FEATURE_GUIDE.md`: complete documentation for current public
  workflows and features.
- `docs/VERSIONING.md`: versioning notes.
- `tests/`: pytest coverage for core library flow and external loading.
- `app.py`: optional local Flask prediction API.
- `train.py`: local training script.
- `pyproject.toml`: package metadata and dependency constraints.
- `setup.py`: setuptools release helper.

Architecture rules:

- Keep the library feature-based by workflow: data, model, training, inference,
  loaders, diagnostics, logging.
- Do not add a generic Shared/Core layer unless logic is reused by multiple
  implemented workflows.
- Do not split files only to create micro-abstractions.
- Keep public documentation aligned with implemented code and tests.

[ORPHANS & PENDING]

- `pyproject.toml` currently declares `requires-python = ">=3.9"` while the
  latest observed PyPI releases for several dependencies now require Python
  `>=3.10` or `>=3.12`. Choose whether to preserve broad compatibility with
  older dependency resolutions or raise the package Python baseline.
- `pipeline_v2.py` overlaps with `pipeline.py`; keep it documented as an
  alternate exported surface until it is either consolidated or removed.
- `checkpoint_is_compatible_for_tuining` contains a spelling error in the public
  export. Renaming it would be a breaking API change unless an alias is kept.
- `app.py` is an optional local API demo and is not covered by the current test
  suite.
- No CI workflow is present in the repository.
