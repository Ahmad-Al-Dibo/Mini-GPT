# Datasets Directory

Pre-prepared knowledge base markdown files.

## 📚 Available Datasets

### Programming Languages
- Python (programming language).md
- JavaScript.md
- Java (programming language).md
- C-programming-language.md
- C++ (C Plus Plus).md
- C# (C Sharp).md
- Rust (programming language).md
- Go (programming language).md

### Cloud Platforms
- Amazon Web Services.md
- Microsoft Azure.md
- Google Cloud Platform.md

### Databases
- MongoDB.md
- MySQL.md
- PostgreSQL.md

### Web Frameworks
- React (software).md
- Angular.md
- Vue.js.md

## 🚀 Using Datasets

### Train on a single topic
```python
from src.dataset import TextDataset
from torch.utils.data import DataLoader

# Load single dataset
dataset = TextDataset("datasets/programming/Python.md", block_size=128)
loader = DataLoader(dataset, batch_size=32)

# Train
trainer.train(loader, val_loader, epochs=20)
```

### Combine multiple datasets
```python
import os
from src.dataset import TextDataset

# Combine all markdown files
texts = []
for filename in os.listdir("datasets/programming"):
    if filename.endswith(".md"):
        with open(f"datasets/programming/{filename}") as f:
            texts.append(f.read())

# Save combined
with open("data/training/combined.txt", "w") as f:
    f.write("\n\n".join(texts))

# Load combined
dataset = TextDataset("data/training/combined.txt")
```

### Train domain-specific model
```python
# Train on cloud platforms only
import os

texts = []
for filename in os.listdir("datasets/platforms"):
    if filename.endswith(".md"):
        with open(f"datasets/platforms/{filename}") as f:
            texts.append(f.read())

# Create domain model
# ... (see examples/02_custom_training.py)
```

## 📊 Dataset Statistics

| Category | Files | Topics |
|----------|-------|--------|
| Programming | 8 | Languages |
| Platforms | 3 | Cloud services |
| Databases | 3 | Data storage |
| Frameworks | 3 | Web development |
| **Total** | **20** | **Tech topics** |

## 🔍 Exploring Datasets

### View available datasets
```bash
ls datasets/programming/
ls datasets/platforms/
ls datasets/databases/
```

### Check file size
```bash
wc -w datasets/programming/Python.md
du -sh datasets/
```

### Sample content
```bash
head -50 datasets/programming/Python.md
```

## 🎯 Use Cases

- **Learn about topics:** Read the markdown files
- **Train domain models:** Use specific datasets
- **Fine-tune chatbots:** Combine relevant datasets
- **Test examples:** Use as sample data
- **Understand content:** See how data flows

## 🔗 Related

- All Markdown files here: `./`
- Training Guide: `../docs/02_FULL_DOCUMENTATION.md#training-guide`
- Data Preparation: `../docs/05_DATASET_GUIDE.md`
- Examples: `../examples/02_custom_training.py`

## 📝 Adding Datasets

1. Create folder: `datasets/mycategory/`
2. Add markdown files: `datasets/mycategory/topic.md`
3. Load and train (see examples above)

---

**See:** `../docs/05_DATASET_GUIDE.md#available-datasets`
