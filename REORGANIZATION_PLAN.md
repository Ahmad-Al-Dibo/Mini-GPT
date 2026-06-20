# MiniGPT Project - Reorganized Structure

## Project Organization Plan

### 1. Root Level - Core Files Only
Keep essential files at root level:
- `app.py` - Flask web server (CORE)
- `requirements.txt` - Dependencies (CORE)
- `LICENSE` - License file (CORE)
- `README.md` - Main project README (CORE)
- `.gitignore`, `.git` - Git files (KEEP)

### 2. `/docs` - Documentation (Organized)
```
docs/
├── README.md                          # Docs index (MAIN ENTRY POINT)
├── 01_QUICK_START.md                  # 5-min guide for beginners
├── 02_FULL_DOCUMENTATION.md           # Complete reference
├── 03_API_REFERENCE.md                # API documentation
├── 04_ARCHITECTURE_DEEP_DIVE.md       # Technical details
├── 05_DATASET_GUIDE.md                # Data preparation
│
└── legacy/                            # Old docs to keep for reference
    ├── PROJECT_MAP.md
    ├── last-raport.md
    └── README_INDEX.md
```

### 3. `/src` - Source Code (Primary Library)
```
src/
├── __init__.py
├── config.py                          # Configuration
├── model.py                           # Core model
├── trainer.py                         # Training logic
├── generator.py                       # Text generation
├── inference.py                       # Inference interface
├── tokenizer.py                       # Tokenization
├── dataset.py                         # Data loading
├── pipeline.py                        # Training pipeline
├── diagnostics.py                     # Evaluation tools
├── data.py                            # Data utilities
├── regularization.py                  # Regularization
├── utils.py                           # Helper functions
├── logics/                            # Logical reasoning module
└── __init__.py
```

### 4. `/models` - Pre-trained Models
```
models/
├── MiniGPT.pth
├── MediumGPT.pth
├── MiniGPT-fine-tuned.pth
└── README.md                          # Model descriptions
```

### 5. `/data` - Training Data
```
data/
├── README.md                          # Data guide
├── tokenizer.json                     # Tokenizer
├── sample_data.txt                    # Sample for quick testing
├── training/
│   ├── combined_text.txt
│   ├── instructions.csv
│   └── qa_data.csv
└── raw/                               # Raw downloads
    ├── alpaca.csv
    ├── bookcorpus.txt
    └── ...
```

### 6. `/datasets` - Knowledge Base (Markdown Files)
```
datasets/
├── README.md                          # Available datasets
├── programming/
│   ├── python.md
│   ├── javascript.md
│   ├── java.md
│   └── ...
├── platforms/
│   ├── aws.md
│   ├── azure.md
│   └── gcp.md
└── databases/
    ├── mongodb.md
    ├── mysql.md
    └── postgresql.md
```

### 7. `/examples` - Usage Examples
```
examples/
├── README.md                          # Examples guide
├── 01_basic_generation.py
├── 02_custom_training.py
├── 03_instruction_tuning.py
├── 04_web_api.py
├── 05_advanced_features.py
└── notebooks/                         # Jupyter notebooks
    ├── 01_load_data.ipynb
    ├── 02_train_model.ipynb
    └── 03_generate_text.ipynb
```

### 8. `/evaluation` - Evaluation Tools
```
evaluation/
├── README.md                          # Evaluation guide
├── metrics.py                         # Main metrics
├── diagnostic.py                      # Diagnostics
├── benchmarks.py                      # Benchmarking
└── utils.py                           # Evaluation utilities
```

### 9. `/tests` - Unit Tests
```
tests/
├── __init__.py
├── test_config.py
├── test_model.py
├── test_trainer.py
├── test_inference.py
└── test_data.py
```

### 10. `/config` - Configuration Files
```
config/
├── model_configs.py                   # Model configurations
├── training_configs.py                # Training configs
└── README.md                          # Config guide
```

## Files to Remove
- `gpt_firstVersions/` - Old versions (keep in git history)
- `gpt_lib/` - Move to `src/`
- Duplicate `dataset.json` - Move to `data/`
- `DOCUMENTATION_README.md` - Merge into `docs/README.md`
- `README_INDEX.md` - Merge into `docs/README.md`
- `PROJECT_MAP.md` - Move to `docs/legacy/`
- `last-raport.md` - Move to `docs/legacy/`

## Reorganization Steps

1. Create new directory structure
2. Move source files from `gpt_lib/` to `src/`
3. Move models from `output/` to `models/`
4. Reorganize data files into `data/`
5. Organize documentation
6. Update import statements
7. Create README files for each section
8. Remove old/redundant files

## Benefits
✅ Cleaner project structure
✅ Clear separation of concerns
✅ Easier navigation
✅ Professional organization
✅ Industry-standard layout
✅ Better documentation
✅ Reduced clutter


## friendly-usage

example:
```python
from src.inference import LoadedModel
model = LoadedModel("models/mini_gpt.pth")  
result = model.predict("AI is", max_new_tokens=50)
print(result)
```