# Models Directory

Pre-trained MiniGPT models ready to use.

## 📦 Available Models

### MiniGPT.pth (~2M parameters)
- **Speed:** Fast (30 tokens/sec on CPU, 500+ on GPU)
- **Quality:** Good
- **Use case:** Quick experimentation, learning
- **Size:** 8MB

### MediumGPT.pth (~11M parameters)
- **Speed:** Slower (10 tokens/sec on CPU, 300+ on GPU)
- **Quality:** Better
- **Use case:** Serious work, better outputs
- **Size:** 45MB

### Custom Models
Create your own trained models and save them here.

## 🚀 Quick Usage

```python
from src.inference import LoadedModel

# Load model
model = LoadedModel("models/MiniGPT.pth")

# Generate text
result = model.predict("The future is", max_new_tokens=50)
print(result)
```

## 📊 Model Comparison

| Aspect | MiniGPT | MediumGPT |
|--------|---------|-----------|
| Parameters | 2M | 11M |
| Speed (CPU) | Very Fast | Slow |
| Speed (GPU) | 500 tok/s | 300 tok/s |
| Quality | Good | Better |
| VRAM | 100MB | 500MB |
| Training Time | 10 min | 1 hour |

## 🎯 Which Model to Use?

- **Learning/Testing:** Use MiniGPT
- **Production/Quality:** Use MediumGPT
- **Limited resources:** Use MiniGPT
- **Want best output:** Use MediumGPT

## 💾 Saving Your Models

```python
from src.pipeline import Pipeline

pipeline = Pipeline()

# After training...
pipeline.save_model(model, "models/my_custom_model.pth")
```

## 📝 Model Metadata

Each .pth file contains:
- Model weights
- Configuration
- Tokenizer info
- Training metadata

---

*See: ../docs/03_API_REFERENCE.md#loadedmodel*
