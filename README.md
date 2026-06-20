# MiniGPT - Educational Transformer Language Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A clean, educational implementation of a GPT-style transformer language model designed for learning and experimentation.

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate text
python -c "from src.inference import LoadedModel; m = LoadedModel('models/MiniGPT.pth'); print(m.predict('The future is'))"

# 3. Start web interface
python app.py
# Open: http://localhost:5000
```

## 📚 Documentation

**New to MiniGPT?** → Start here: **[`docs/00_START_HERE.md`](docs/00_START_HERE.md)** ⚡

All documentation is in the `docs/` folder:
- **01_QUICK_START.md** - 5-minute getting started guide
- **02_FULL_DOCUMENTATION.md** - Comprehensive reference
- **03_API_REFERENCE.md** - Complete API documentation  
- **04_ARCHITECTURE_DEEP_DIVE.md** - Technical details
- **05_DATASET_GUIDE.md** - Data preparation guide

## 📁 Project Structure

```
Mini-GPT/
├── 📄 README.md                        ← You are here
├── 📄 requirements.txt                 ← Dependencies
├── 📄 app.py                           ← Flask web server
│
├── 📁 src/                             ← Source code library
│   ├── model.py, trainer.py, inference.py, ...
│   └── See: docs/03_API_REFERENCE.md
│
├── 📁 docs/                            ← Documentation (START HERE)
│   ├── 00_START_HERE.md                ← Main entry point
│   ├── 01-05_*.md                      ← 5 main guides
│   └── legacy/                         ← Old documentation
│
├── 📁 examples/                        ← Usage examples
│   ├── 01_basic_generation.py
│   ├── 02_custom_training.py
│   ├── 03_instruction_tuning.py
│   └── ... (more examples)
│
├── 📁 models/                          ← Pre-trained models
│   ├── MiniGPT.pth
│   ├── MediumGPT.pth
│   └── README.md
│
├── 📁 data/                            ← Training data
│   ├── tokenizer.json
│   ├── training/                       ← Training datasets
│   ├── raw/                            ← Raw data files
│   └── README.md
│
├── 📁 datasets/                        ← Knowledge base markdown
│   ├── programming/
│   ├── platforms/
│   ├── databases/
│   └── README.md
│
├── 📁 evaluation/                      ← Evaluation tools
│   ├── metrics.py, diagnostic.py, ...
│   └── README.md
│
├── 📁 tests/                           ← Unit tests
│   └── test_*.py
│
├── 📁 config/                          ← Configuration
│   ├── model_configs.py
│   └── README.md
│
├── 📁 templates/                       ← Web UI templates
│   └── index.html
│
└── 📁 notebooks/                       ← Jupyter notebooks
    ├── 01_load_data.ipynb
    └── ... (more notebooks)
```

## 🎯 What Can You Do?

- ✅ **Generate text** - Use pre-trained models
- ✅ **Train models** - On your own data
- ✅ **Fine-tune** - Instruction-following tasks
- ✅ **Deploy** - Flask web API
- ✅ **Evaluate** - Perplexity, accuracy, diagnostics
- ✅ **Understand** - Complete documentation & examples

## 💻 Usage Examples

### Generate Text
```python
from src.inference import LoadedModel

model = LoadedModel("models/MiniGPT.pth")
result = model.predict("The future of AI is", max_new_tokens=50)
print(result)
```

### Train a Model
```python
from src.pipeline import Pipeline
from src.dataset import TextDataset
from torch.utils.data import DataLoader

pipeline = Pipeline()
model, trainer = pipeline.build_model_and_trainer("MiniGPT")

dataset = TextDataset("data/my_data.txt", block_size=128)
loader = DataLoader(dataset, batch_size=32)

trainer.train(loader, loader, epochs=10, checkpoint_path="models/my_model.pth")
```

### Use Web API
```bash
# Start server
python app.py

# Generate text via API
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "AI is", "max_new_tokens": 50}'
```

## 🏗️ Architecture

MiniGPT implements a **decoder-only transformer** with:
- **Multi-head self-attention** - Parallel attention heads
- **Position embeddings** - Learns position information
- **Feed-forward networks** - Non-linear transformations
- **Layer normalization** - Gradient stability
- **Causal masking** - Prevents attending to future tokens

For details, see: [`docs/04_ARCHITECTURE_DEEP_DIVE.md`](docs/04_ARCHITECTURE_DEEP_DIVE.md)

## 📦 Requirements

- Python 3.8+
- PyTorch 1.9+
- 4GB+ RAM (8GB+ recommended)
- GPU optional but recommended (NVIDIA with CUDA)

Install:
```bash
pip install -r requirements.txt
```

## 📊 Model Sizes

| Model | Parameters | Speed | Quality |
|-------|-----------|-------|---------|
| **MiniGPT** | ~2M | Fast | Good |
| **MediumGPT** | ~11M | Slow | Better |

## 🎓 Learning Resources

### For Beginners
1. Read: [`docs/01_QUICK_START.md`](docs/01_QUICK_START.md)
2. Try: Generate text example
3. Learn: [`docs/02_FULL_DOCUMENTATION.md`](docs/02_FULL_DOCUMENTATION.md)

### For Developers
1. Read: [`docs/03_API_REFERENCE.md`](docs/03_API_REFERENCE.md)
2. Check: Examples in `examples/`
3. Integrate: Into your application

### For Researchers
1. Read: [`docs/04_ARCHITECTURE_DEEP_DIVE.md`](docs/04_ARCHITECTURE_DEEP_DIVE.md)
2. Study: Mathematical formulations
3. Extend: Model architecture

### For Data Scientists
1. Read: [`docs/05_DATASET_GUIDE.md`](docs/05_DATASET_GUIDE.md)
2. Prepare: Your data
3. Train: Custom models

## 🔧 Common Tasks

| Task | Command | Docs |
|------|---------|------|
| Generate text | `python examples/01_basic_generation.py` | Examples |
| Train model | `python examples/02_custom_training.py` | Training Guide |
| Start web UI | `python app.py` | Web Interface |
| Run tests | `pytest tests/` | Tests |
| View help | `cat docs/00_START_HERE.md` | Quick Start |

## 📈 Performance

| Device | Speed | Notes |
|--------|-------|-------|
| CPU | ~30 tokens/sec | Slow but works |
| GPU (RTX 3090) | ~500 tokens/sec | Fast inference |
| Training | 30-60s per epoch | Depends on data size |

## ❓ FAQ

**Q: Is this production-ready?**  
A: No, it's educational. For production, use Hugging Face Transformers.

**Q: How much data do I need?**  
A: 10MB minimum, 100MB+ recommended.

**Q: Can I use this on CPU?**  
A: Yes, but GPU is 10-50x faster.

**Q: How do I fine-tune?**  
A: See [`docs/02_FULL_DOCUMENTATION.md#instruction-tuning`](docs/02_FULL_DOCUMENTATION.md#instruction-tuning)

**Q: Can I export the model?**  
A: Yes, models are saved as PyTorch .pth files.

## 🤝 Contributing

Contributions welcome! Areas:
- Better documentation
- Additional examples
- Performance optimizations
- New features

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Inspired by GPT-2 and GPT-3
- Based on transformer architecture from "Attention Is All You Need"
- Built with PyTorch

## 📞 Support

- **Quick help:** See [`docs/00_START_HERE.md`](docs/00_START_HERE.md)
- **Issues:** Check [`docs/02_FULL_DOCUMENTATION.md#troubleshooting`](docs/02_FULL_DOCUMENTATION.md#troubleshooting)
- **Questions:** Read relevant docs

---

**Ready to start?** → Open [`docs/00_START_HERE.md`](docs/00_START_HERE.md) 🚀

---

**Last Updated:** June 20, 2026  
**Version:** 1.0
