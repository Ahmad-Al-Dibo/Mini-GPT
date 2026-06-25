# 🗺️ PROJECT_MAP.md — MiniGPT Architectural Blueprint

**Last Updated:** 2026-06-25  
**Status:** READY FOR IMPLEMENTATION  
**Author:** Staff Tech Lead  

---

## [TECH_STACK]

### Core Dependencies (Pinned Q2 2026)
```
torch==2.1.2 (LTS with CUDA 12.1 support)
transformers==4.38.1
sentencepiece==0.2.1
accelerate==0.26.1
bitsandbytes==0.42.0
flask==3.0.0
numpy==1.24.3
```

### Architecture Layers
```
┌─────────────────────────────────────────────┐
│  PUBLIC INTERFACE LAYER (API + Web UI)      │
│  - /app.py (Flask web app + REST API)      │
│  - /src/miniGPT/__init__.py (SDK exports)  │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  WORKFLOW LAYER (Training & Inference)      │
│  - pipeline.py (unified trainer factory)   │
│  - inference.py (model loading + predict)  │
│  - generator.py (text generation logic)    │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  DOMAIN LAYER (Core ML Logic)               │
│  - trainer.py (optimization + callbacks)   │
│  - model.py (transformer architecture)     │
│  - dataset.py (data loading + batching)    │
│  - instruction_dataset.py (fine-tune data) │
│  - tokenizer.py (multi-tokenizer support)  │
│  - diagnostics.py (eval metrics)           │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  CROSS-CUTTING (Config, Utils, Logics)     │
│  - config.py (centralized parameters)      │
│  - utils.py (helpers, logging)             │
│  - regularization.py (dropout, weight_decay)│
│  - logics/* (logic/constraint modules)      │
└─────────────────────────────────────────────┘
```

---

## [SYSTEM_FLOW]

### User Journey: Workflow Stages

#### **Stage 1: SETUP (Goal Definition)**
```
User Action           → System Module
─────────────────────────────────────────
1. Define task goal     → Config.get_*_config()
2. Choose model size    → config.py (embed_dim, num_blocks)
3. Select tokenizer     → tokenizer.py (init tokenizer)
4. Set training params  → Config + pipeline.py
```
**Verifiable Goal:** User has executable TrainingConfig object

---

#### **Stage 2: DATA (Collection + Preparation)**
```
1. Load raw dataset     → dataset.py (load_text, load_jsonl)
2. Preprocess & clean   → data.py (clean_text, tokenize)
3. Create batches       → dataset.py (DataLoader)
4. Optional: QA format  → instruction_dataset.py
```
**Verifiable Goal:** trainer.train_dataloader ready for iteration

---

#### **Stage 3: TRAINING (Pre-train / Fine-tune)**
```
1. Initialize model     → model.py + inference.py
2. Load optimizer       → trainer.py (build_optimizer / discriminative LR)
3. Freeze layers (opt)  → trainer.py (freeze_layers)
4. Execute training     → pipeline.py.train()
5. Apply stopping       → trainer.py (stopping_criteria, patience)
6. Log metrics          → utils.py.Logger + diagnostics.py
```
**Verifiable Goal:** Model checkpoint saved + training.json report

---

#### **Stage 4: EVALUATION (Validation)**
```
1. Load validation set  → dataset.py
2. Compute metrics      → diagnostics.py (perplexity, F1, etc.)
3. Generate samples     → generator.py (sample_text)
4. Log results          → diagnostics.py.DiagnosticReport
```
**Verifiable Goal:** eval_report.json with metrics + sample outputs

---

#### **Stage 5: OPTIMIZATION (Tuning)**
```
1. Hyperparameter sweep → config.py (create variants)
2. Re-train + eval      → Repeat stages 3-4
3. Choose best          → Compare eval_reports
4. Save best checkpoint → inference.py (export)
```
**Verifiable Goal:** Optimized model weights + best_config.json

---

#### **Stage 6: REPORTING (Metrics + Docs)**
```
1. Aggregate metrics    → diagnostics.py.aggregate_results()
2. Generate plots       → diagnostics.py (matplotlib/plotly)
3. Export report        → diagnostics.py.export_markdown()
4. Log to wandb/mlflow  → utils.py (optional integration)
```
**Verifiable Goal:** reports/training_report_{timestamp}.md

---

#### **Stage 7: DEPLOYMENT (Inference)**
```
1. Load model + config  → inference.py.LoadedModel(path)
2. Expose REST API      → app.py.generate_endpoint()
3. Web UI (optional)    → templates/index.html
```
**Verifiable Goal:** POST /generate returns {"text": "..."}

---

### Data Flow Diagram
```
Raw Text/CSV/JSONL
    ↓
[data.py: clean_text, tokenize]
    ↓
[dataset.py: create batches, DataLoader]
    ↓ (splits: train/val/test)
[trainer.py: epoch loop, loss.backward(), optimizer.step()]
    ↓
[diagnostics.py: eval metrics on val set]
    ↓
[model.py: forward pass, logits → loss]
    ↓
[utils.py: log metrics, save checkpoint]
    ↓ (if eval_loss improves)
[checkpoint.pth + training.json]
    ↓
[inference.py: load → generator.py: predict()]
    ↓
[Web/API Response or Text Output]
```

---

## [ARCHITECTURE]

### Feature-Domain Structure (Simplicity First)

```
src/miniGPT/
├── __init__.py                 # Public SDK: load_model, Config, Trainer
├── config.py                   # Config class + factory methods
├── model.py                    # TransformerModel (nn.Module)
├── trainer.py                  # Trainer class + optimization
├── pipeline.py                 # Pipeline.train() - main orchestrator
├── inference.py                # LoadedModel class + model loading
├── generator.py                # TextGenerator - sampling strategies
├── dataset.py                  # DataLoader + preprocessing
├── instruction_dataset.py      # InstructionTuningDataset
├── tokenizer.py                # TokenizerFactory (SentencePiece, BPE, etc)
├── diagnostics.py              # DiagnosticReport + eval metrics
├── data.py                     # Low-level data functions
├── utils.py                    # Logging, helpers, utilities
├── regularization.py           # Dropout, Weight Decay strategies
└── logics/                     # Logic/constraint modules (symbol logic)
    ├── symbol.py
    ├── sentence.py
    ├── model_check.py
    └── [logic gates...]

config/
├── pre_training.py             # Pre-training setup example
└── tuning.py                   # Fine-tuning setup example

examples/
├── example1_generation.py      # Basic text generation
├── example2_custom_training.py # Custom training loop
├── example3_instruction_training.py # Instruction tuning
├── example4_advanced.py        # Discriminative LR + layer freezing
├── example5_advanced_training.py # Complete workflow
└── example_finetuning.py       # Phase 1 finetuning

evaluation/
├── diagnostic_simple.py        # Simple eval script
├── diagnostic.py               # Full diagnostic suite
├── instruction_tuning.py       # Instruction-specific evals
├── optimize.py                 # Hyperparameter sweep runner
└── roadmap.py                  # Evaluation roadmap

data/
├── *.txt, *.csv, *.md          # Training datasets
├── alpaca.csv, qa.csv          # Instruction data
├── tokenizer.json              # Pre-trained tokenizer

models/
├── *.pth                       # Model checkpoints

app.py                          # Flask web app + REST API
requirements.txt                # Dependencies
```

### Module Responsibilities (No Redundancy)

| Module | Purpose | Owns | Doesn't Own |
|--------|---------|------|-------------|
| `config.py` | Parameter management | Config class, factory methods, pre-trained model configs | Actual training logic |
| `model.py` | Architecture | TransformerModel (nn.Module) | Training loop |
| `trainer.py` | Optimization & training | Trainer class, optimizer building, loss computation, callbacks | Model architecture |
| `pipeline.py` | Orchestration | Pipeline.train() (wraps trainer), model loading strategy | Specific model/data implementations |
| `dataset.py` | Data loading | Batching, shuffling, DataLoader creation | Text cleaning |
| `data.py` | Data processing | Text cleaning, tokenization, formatting | Loading raw files |
| `inference.py` | Model loading | LoadedModel class, weight loading, pre-trained model support | Text generation strategy |
| `generator.py` | Text generation | Sampling (greedy, top-k, top-p), decoding | Model loading |
| `diagnostics.py` | Evaluation | Metrics (perplexity, F1), reports, plots | Actual predictions |
| `tokenizer.py` | Tokenization | Tokenizer factory, multi-backend support | Model training |
| `utils.py` | Cross-cutting | Logging, file I/O, helpers, model registry | Domain logic |

---

## [ORPHANS & PENDING]

### ✅ Completed (Phases 1-3)
- [x] Project reorganization (src/miniGPT consolidation)
- [x] Documentation structure (00_START_HERE + 5 guides)
- [x] Phase 1: Finetuning essentials (layer freezing, discriminative LR)
- [x] Flask web app with /generate endpoint
- [x] Model checkpoints consolidated in models/

### 📋 NEW REQUIREMENT (Step 12)
- **Support Pre-trained Model Integration:** Enable loading, adapting, and fine-tuning external pre-trained models (HuggingFace, local files, custom sources)
- **Impact:** Extends P4-1 scope; adds model adapter layer

### 🚀 PHASE 4: READY TO IMPLEMENT (Next)
- [ ] **P4-1: Unified Training Pipeline + Pre-trained Model Support**
  - [ ] Refactor pipeline.py to support: pre-training, fine-tuning, instruction-tuning flows
  - [ ] Add stopping criteria: early_stopping, max_steps, patience
  - [ ] Integrate with diagnostics for live eval during training
  - [ ] **NEW:** Support loading pre-trained models from HuggingFace, local paths, custom sources
  - [ ] **NEW:** Model adapter layer for seamless integration with external model weights
  - **Files:** pipeline.py, trainer.py, config.py, inference.py
  - **Verifiable:** `pytest tests/test_pipeline.py::test_unified_training` + `test_load_pretrained_model`

- [ ] **P4-2: Multi-Tokenizer Support**
  - [ ] Complete tokenizer.py factory pattern (SentencePiece, BPE, WordPiece)
  - [ ] Add tokenizer caching + validation
  - [ ] Support custom vocab files
  - **Files:** tokenizer.py, data.py
  - **Verifiable:** `pytest tests/test_tokenizer.py` (all tokenizers load)

- [ ] **P4-3: Comprehensive Diagnostics**
  - [ ] Extend diagnostics.py: perplexity, F1, BLEU, custom metrics
  - [ ] Add visualization: loss curves, metric trends
  - [ ] Generate markdown reports with statistics + sample outputs
  - **Files:** diagnostics.py, evaluation/*.py
  - **Verifiable:** `evaluation/diagnostic.py --model <path>` produces report.md

- [ ] **P4-4: Library API & Packaging**
  - [ ] Finalize __init__.py exports (load_model, Config, Trainer, Pipeline)
  - [ ] Add type hints throughout SDK
  - [ ] Create setup.py for PyPI distribution
  - [ ] Choose library name: `miniGPT` (or variant)
  - **Files:** src/miniGPT/__init__.py, setup.py, setup.cfg
  - **Verifiable:** `pip install -e .` works + `from miniGPT import *`

- [ ] **P4-5: End-to-End Examples & Tests**
  - [ ] Complete example: goal → data → train → eval → deploy
  - [ ] Unit tests for each module (50+ test cases)
  - [ ] Integration test: full training pipeline
  - **Files:** examples/*, tests/*
  - **Verifiable:** `pytest tests/` (all pass) + examples/*.py run without errors

- [ ] **P4-6: Documentation & Deployment Guide**
  - [ ] API reference autogen from docstrings
  - [ ] Deployment guide: Docker, cloud platforms
  - [ ] FAQ, troubleshooting guide
  - **Files:** docs/*, Dockerfile
  - **Verifiable:** All docs render correctly; Docker image builds

### 📋 Backlog (Future Phases)
- [ ] Distributed training (multi-GPU via accelerate)
- [ ] Quantization support (bitsandbytes integration)
- [ ] Custom metrics plugin system
- [ ] MLflow/W&B integration (nice-to-have)
- [ ] CLI tool (optional if API sufficient)
- [ ] Browser-based notebook UI (nice-to-have)

---

## ACTION PLAN & MILESTONES
 + Pre-trained Model Support
  ✓ P4-2: Multi-Tokenizer Support
  
Week 2:
  ✓ P4-3: Comprehensive Diagnostics
  ✓ P4-4: Library API & Packaging
  
Week 3:
  ✓ P4-5: Examples & Tests (including pre-trained model examples)
  ✓ P4-6: Documentation (including pre-trained model guide) Packaging
  
Week 3:
  ✓ P4-5: Examples & Tests
  ✓ P4-6: Documentation
  
→ RELEASE v1.0.0+ `pytest tests/test_pretrained_models.py` → all pass
2. **Week 2 End:** `from miniGPT import *` imports successfully; `pip install -e .` works; model registry functional
3. **Week 3 End:** All examples run end-to-end (including HuggingFace adapter example); `pytest tests/` → 55+ passing tests; docs complete with pre-trained model guid
### Verifiable Goals per Milestone
1. **Week 1 End:** `pytest tests/test_pipeline.py` + `pytest tests/test_tokenizer.py` → all pass
2. **Week 2 End:** `from miniGPT import *` imports successfully; `pip install -e .` works
3. **Week 3 End:** All examples run end-to-end; `pytest tests/` → 50+ passing tests; docs complete

---

## ASSUMPTIONS & CONSTRAINTS

### Core Assumptions
1. **GPU-First, CPU-Fallback:** Code assumes CUDA available, falls back to CPU gracefully
2. **HuggingFace Compatible:** Tokenizers/models integrate with transformers library ecosystem
3. **Single Machine:** No distributed training (accelerate for single-GPU optimization only)
4. **Pre-configured Models:** Ship with 3 model sizes (tiny, small, medium)
5. **Evaluation is Non-Blocking:** Diagnostics runs async; doesn't block training

### Implementation Constraints
- **No Feature Creep:** Only implement the 11 specified process steps
- **Simplicity First:** Maximum 1 abstraction layer per concept
- **Shared Core Only:** Create shared module only if used by 2+ components
- **Backward Compatible:** Don't break Phase 1 finetuning API
- **Type Hints:** All new code must include type hints for IDE support

---

## NAMING DECISION: LIBRARY IDENTITY

### Library Name: `miniGPT` (or Python package: `minigpt`)

**Rationale:**
- ✅ Already used in project (src/miniGPT)
- ✅ Descriptive + concise
- ✅ Available on PyPI (to verify)
- ✅ Reflects "mini" (lightweight) + "GPT" (transformer-based)

**PyPI Package:** `minigpt`  
**Import:** `from minigpt import load_model, Config, Trainer`  
**CLI Command:** `minigpt-train config.json`  

---

## SUCCESS CRITERIA

### Definition of "Done" (Per Phase)
```
Phase 4 Complete When:
✓ Unified pipeline supports: pre-training, fine-tuning, instruction-tuning
✓ Pre-trained model support: load from HuggingFace, local files, custom sources
✓ Model adapter layer: seamless weight integration + config alignment
✓ All 4 tokenizers (SentencePiece, BPE, WordPiece, custom) work
✓ Diagnostics: perplexity, F1, BLEU, custom metrics
✓ SDK exports stable API; pip installable
✓ 55+ unit + integration tests passing (including pre-trained model tests)
✓ End-to-end examples: goal → report, AND pre-trained model → fine-tune
✓ Documentation: API reference, examples, troubleshooting, pre-trained model guide
```

---

**🎯 NEXT STEP:** Await approval. Upon approval, implementation begins with P4-1.
