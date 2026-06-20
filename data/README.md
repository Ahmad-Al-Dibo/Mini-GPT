# Data Directory

Training data, tokenizer, and dataset management.

## 📁 Structure

```
data/
├── tokenizer.json              # SentencePiece tokenizer
├── sample_data.txt             # Sample data for testing
├── training/                   # Training datasets
│   ├── combined_text.txt
│   ├── instructions.csv
│   └── qa_data.csv
└── raw/                        # Raw data files
    ├── alpaca.csv
    ├── bookcorpus.txt
    └── ... (more files)
```

## 🔤 Tokenizer

**File:** `tokenizer.json`

Pre-trained SentencePiece BPE tokenizer with 4096 tokens.

### Using the tokenizer
```python
from src.tokenizer import Tokenizer

tokenizer = Tokenizer("data/tokenizer.json")

# Encode
tokens = tokenizer.encode("Hello world")
print(tokens)  # [123, 456, ...]

# Decode
text = tokenizer.decode(tokens)
print(text)  # "Hello world"

# Vocab size
print(tokenizer.get_vocab_size())  # 4096
```

### Train new tokenizer
```python
from src.tokenizer import Tokenizer

Tokenizer.train(
    text_file="my_data.txt",
    vocab_size=8192,
    output_path="tokenizer_new.json"
)
```

## 📊 Available Data

### sample_data.txt
Quick sample for testing. Use for:
- Verifying setup works
- Quick training experiments
- Testing examples

### training/combined_text.txt
Combined training data from multiple sources.

### training/instructions.csv
Q&A pairs for instruction tuning:
```csv
question,answer
What is AI?,Artificial Intelligence is...
What is ML?,Machine Learning is...
```

### training/qa_data.csv
Similar to instructions but different format.

## 🚀 Using Data

### Load for training
```python
from src.dataset import TextDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = TextDataset("data/sample_data.txt", block_size=128)

# Create loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
trainer.train(loader, val_loader, epochs=10)
```

### Load instruction data
```python
from src.data import create_instruction_following_dataset

dataset, tokenizer = create_instruction_following_dataset(
    "data/training/instructions.csv"
)
```

## 📥 Adding Your Data

### Option 1: Plain text
Save as `data/training/my_data.txt`:
```
This is your training data.
One sentence per line recommended.
```

### Option 2: CSV
Save as `data/training/my_data.csv`:
```csv
question,answer
Q1,A1
Q2,A2
```

### Option 3: JSON
Save as `data/training/my_data.json`:
```json
[
  {"text": "Document 1..."},
  {"text": "Document 2..."}
]
```

## 🧹 Data Cleaning

See: `../docs/05_DATASET_GUIDE.md#cleaning-pipeline`

```python
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Fix spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean your data
with open('data/raw/dirty.txt') as f:
    dirty = f.read()

clean = clean_text(dirty)

with open('data/training/clean.txt', 'w') as f:
    f.write(clean)
```

## 💾 Data Management Tips

- **Version data:** Keep raw and processed separate
- **Document sources:** Know where data comes from
- **Check quality:** Inspect samples before training
- **Train/Val split:** Use 80/20 or 70/15/15
- **Size matters:** 100MB+ for good results

## 🔗 Related

- Documentation: `../docs/05_DATASET_GUIDE.md`
- Examples: `../examples/02_custom_training.py`
- Full Guide: `../docs/02_FULL_DOCUMENTATION.md#data-preparation`

## ⚠️ Important

- Ensure data encoding is UTF-8
- Remove any binary data
- Check for duplicate content
- Verify data quality

---

**See:** `../docs/05_DATASET_GUIDE.md` for complete guide
