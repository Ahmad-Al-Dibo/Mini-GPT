# Examples Directory

Working examples showing how to use MiniGPT.

## 📚 Available Examples

### 01_basic_generation.py
**What:** Generate text using pre-trained model  
**Time:** 2 minutes  
**Use case:** Understand basic usage

```python
from src.inference import LoadedModel

model = LoadedModel("../models/MiniGPT.pth")
result = model.predict("AI is", max_new_tokens=50)
print(result)
```

### 02_custom_training.py
**What:** Train model on custom data  
**Time:** 10 minutes  
**Use case:** Train on your own dataset

```bash
python 02_custom_training.py
```

### 03_instruction_tuning.py
**What:** Fine-tune for instruction following  
**Time:** 5 minutes  
**Use case:** Q&A, chatbot-like behavior

### 04_web_api.py
**What:** Deploy as REST API  
**Time:** 3 minutes  
**Use case:** Production deployment

```bash
python ../app.py  # Already set up!
```

### 05_advanced_features.py
**What:** Advanced techniques  
**Time:** 15 minutes  
**Use case:** Optimization, fine-tuning, extensions

## 🚀 Running Examples

### Run a single example
```bash
python 01_basic_generation.py
```

### Run all examples
```bash
python 01_basic_generation.py
python 02_custom_training.py
python 03_instruction_tuning.py
```

## 📊 Jupyter Notebooks

Interactive notebooks in `notebooks/`:
- `01_load_data.ipynb` - Load and explore data
- `02_train_model.ipynb` - Train step-by-step
- `03_generate_text.ipynb` - Generate examples

## 🎯 Learning Path

1. **Start:** 01_basic_generation.py
2. **Learn:** 02_custom_training.py
3. **Explore:** 05_advanced_features.py
4. **Deploy:** 04_web_api.py / app.py

## 📝 Creating Your Own Examples

```python
# my_example.py
from src.inference import LoadedModel
from src.dataset import TextDataset
from torch.utils.data import DataLoader

# Load model
model = LoadedModel("../models/MiniGPT.pth")

# Or train new model
# ... (see 02_custom_training.py)

# Generate
result = model.predict("Your prompt here", max_new_tokens=100)
print(result)
```

## 🔗 Related

- Documentation: `../docs/01_QUICK_START.md`
- Full Guide: `../docs/02_FULL_DOCUMENTATION.md`
- API Reference: `../docs/03_API_REFERENCE.md`

---

*See: ../docs/02_FULL_DOCUMENTATION.md#examples*
