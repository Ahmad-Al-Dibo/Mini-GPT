# Data Preparation & Datasets Guide

Complete guide to preparing data for MiniGPT training.

## Table of Contents

1. [Supported Data Formats](#supported-data-formats)
2. [Data Quality](#data-quality)
3. [Dataset Preparation](#dataset-preparation)
4. [Tokenization](#tokenization)
5. [Available Datasets](#available-datasets)
6. [Custom Datasets](#custom-datasets)
7. [Data Splitting](#data-splitting)
8. [Cleaning Pipeline](#cleaning-pipeline)
9. [Best Practices](#best-practices)

---

## Supported Data Formats

### 1. Plain Text Files (.txt)

**Format:** One or more documents in plain text format.

**Example: `data/documents.txt`**
```
This is the first document about machine learning.
It contains multiple sentences and paragraphs.

This is the second document about deep learning.
Transformers are powerful neural networks.
```

**Loading:**
```python
from gpt_lib.dataset import TextDataset

dataset = TextDataset("data/documents.txt", block_size=128)
```

**Advantages:**
- Simple format
- Easy to create and edit
- Works with any text source

### 2. CSV Files (.csv)

**Format:** Comma-separated values with headers.

**Example: `data/qa_pairs.csv`**
```csv
question,answer
What is machine learning?,Machine learning is a subset of artificial intelligence
What is deep learning?,Deep learning uses neural networks with multiple layers
What is a transformer?,A transformer is a neural architecture based on attention
```

**Loading:**
```python
import pandas as pd
from gpt_lib.dataset import TextDataset

# Combine Q&A into single column
df = pd.read_csv("data/qa_pairs.csv")
texts = df['question'] + " " + df['answer']

# Convert to text file
with open("temp.txt", "w") as f:
    f.write("\n".join(texts))

dataset = TextDataset("temp.txt", block_size=128)
```

**Use Cases:**
- Question-answering pairs
- Instruction-following training
- Domain-specific data

### 3. JSON Files (.json)

**Format:** JSON array or lines of JSON objects.

**Example: `data/articles.json`**

**Array format:**
```json
[
  {
    "title": "Introduction to AI",
    "content": "Artificial intelligence is..."
  },
  {
    "title": "Deep Learning Basics",
    "content": "Deep learning uses neural networks..."
  }
]
```

**JSONL format (one object per line):**
```jsonl
{"text": "First document..."}
{"text": "Second document..."}
{"text": "Third document..."}
```

**Loading:**
```python
import json
from gpt_lib.dataset import TextDataset

# Array format
with open("data/articles.json") as f:
    data = json.load(f)
    texts = [item['content'] for item in data]

# Or JSONL format
texts = []
with open("data/articles.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj['text'])

# Save to text file
with open("combined.txt", "w") as f:
    f.write("\n".join(texts))

dataset = TextDataset("combined.txt")
```

### 4. Markdown Files (.md)

**Available in: `dataset/` directory**

Pre-prepared markdown files on various topics:
- Programming Languages: Python, JavaScript, Java, C++, C#, Rust, Go
- Cloud Platforms: AWS, Azure, GCP
- Databases: MongoDB, MySQL, PostgreSQL
- Frameworks: React, Angular

**Loading Markdown Files:**
```python
import os
from gpt_lib.dataset import TextDataset

# Combine all markdown files
text_dir = "dataset/"
all_texts = []

for filename in os.listdir(text_dir):
    if filename.endswith(".md"):
        with open(os.path.join(text_dir, filename)) as f:
            all_texts.append(f.read())

# Save combined text
with open("combined.txt", "w") as f:
    f.write("\n\n".join(all_texts))

# Load dataset
dataset = TextDataset("combined.txt")
```

### 5. Multiple Formats Combined

```python
import os
import json
import pandas as pd

combined_texts = []

# From text files
for txt_file in ["data/doc1.txt", "data/doc2.txt"]:
    with open(txt_file) as f:
        combined_texts.append(f.read())

# From CSV
df = pd.read_csv("data/qa.csv")
for _, row in df.iterrows():
    combined_texts.append(f"{row['question']} {row['answer']}")

# From JSON
with open("data/articles.json") as f:
    articles = json.load(f)
    for article in articles:
        combined_texts.append(article['content'])

# Save combined
with open("all_data.txt", "w") as f:
    f.write("\n".join(combined_texts))
```

---

## Data Quality

### Minimum Requirements

- **Minimum Size**: 1MB for testing, 100MB+ for good results
- **Language**: Consistent language (ideally English for pre-trained tokenizer)
- **Encoding**: UTF-8 (standard text encoding)
- **Format**: Well-formed text without binary content

### Quality Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| Text length | 10+ tokens | 5+ tokens | <5 tokens |
| Repetition | <5% repeated | 5-15% repeated | >15% repeated |
| Language consistency | >95% | 80-95% | <80% |
| Special characters | <5% | 5-20% | >20% |

### Checking Data Quality

```python
import numpy as np
from collections import Counter

def analyze_data_quality(text_file):
    """Analyze text quality"""
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Metrics
    lines = text.split('\n')
    words = text.split()
    chars = list(text)
    
    stats = {
        'total_chars': len(text),
        'total_lines': len(lines),
        'total_words': len(words),
        'avg_line_length': np.mean([len(line) for line in lines]),
        'avg_word_length': np.mean([len(word) for word in words]),
        'unique_words': len(set(words)),
        'special_chars_percentage': sum(1 for c in chars if not c.isalnum() and c != ' ') / len(chars) * 100,
    }
    
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Average line length: {stats['avg_line_length']:.1f} chars")
    print(f"Unique words: {stats['unique_words']:,}")
    print(f"Special characters: {stats['special_chars_percentage']:.2f}%")
    
    return stats

# Run analysis
analyze_data_quality("data/combined_text.txt")
```

---

## Dataset Preparation

### Step 1: Collect Data

**Sources:**
- Public datasets (Wikipedia, Common Crawl, Books)
- Project documentation
- Articles and blog posts
- Research papers
- Domain-specific corpora

### Step 2: Convert to Unified Format

```python
# Create a script to consolidate data
import os

def prepare_dataset(input_files, output_file):
    """Prepare dataset from multiple files"""
    
    all_texts = []
    
    for input_file in input_files:
        if input_file.endswith('.txt'):
            with open(input_file) as f:
                all_texts.append(f.read())
        
        elif input_file.endswith('.md'):
            with open(input_file) as f:
                all_texts.append(f.read())
    
    # Combine with separators
    combined = "\n\n".join(all_texts)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"Prepared dataset: {output_file}")
    print(f"Size: {len(combined):,} characters")

# Example
input_files = [
    "dataset/Python (programming language).md",
    "dataset/JavaScript.md",
    "dataset/Java (programming language).md"
]

prepare_dataset(input_files, "data/combined_data.txt")
```

### Step 3: Clean Data

```python
import re

def clean_text(text):
    """Clean and normalize text"""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters (but keep basic punctuation)
    text = re.sub(r'[^\w\s\.\!\?\-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

# Clean all data
with open('data/combined_data.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)

with open('data/combined_data_cleaned.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned_text)
```

---

## Tokenization

### Understanding Tokenization

Tokenization converts text into integer tokens that the model understands.

**Example:**
```
Text: "Hello world"
Tokens: [256, 189]
```

### Available Tokenizers

MiniGPT uses SentencePiece BPE tokenizer.

### Using the Tokenizer

```python
from gpt_lib.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer("data/tokenizer.json")

# Encode text to tokens
text = "Machine learning is powerful"
tokens = tokenizer.encode(text)
print(tokens)  # [123, 456, 789, ...]

# Decode tokens back to text
decoded = tokenizer.decode(tokens)
print(decoded)  # "Machine learning is powerful"
```

### Training a New Tokenizer

```python
from gpt_lib.tokenizer import Tokenizer

# Train on your data
Tokenizer.train(
    text_file="data/combined_data.txt",
    vocab_size=4096,  # Number of tokens
    output_path="data/my_tokenizer.json"
)

# Use the new tokenizer
tokenizer = Tokenizer("data/my_tokenizer.json")
```

### Vocabulary Size Effects

| Size | Advantages | Disadvantages |
|------|-----------|-----------------|
| 2048 | Fast training, small memory | May lose information |
| 4096 | **Default, balanced** | - |
| 8192 | Better quality | More memory, slower |
| 16384 | Highest quality | Very slow, large memory |

---

## Available Datasets

### Included in Project

The `dataset/` folder contains 20+ markdown files on:

**Programming Languages:**
- Python.md (2.4MB)
- JavaScript.md (1.8MB)
- Java.md (2.1MB)
- C++.md (2.0MB)
- More...

**Cloud Platforms:**
- Amazon Web Services.md
- Microsoft Azure.md
- Google Cloud Platform.md

**Databases:**
- MongoDB.md
- MySQL.md
- PostgreSQL.md

### External Datasets

**Popular options for training:**

1. **Wikipedia Dump** (~20GB)
   - Format: XML, convert to text
   - Source: dumps.wikimedia.org

2. **Common Crawl** (~570TB)
   - Format: Web pages
   - Source: commoncrawl.org

3. **Books3/Project Gutenberg** (~100GB)
   - Format: Plain text
   - Source: pgdp.net

4. **Stack Exchange Data** (~40GB)
   - Q&A format
   - Source: archive.org

5. **OpenWebText** (~170GB)
   - Curated web text
   - Source: github.com/jcpeterson/openwebtext

### Dataset Preparation for Custom Data

```python
# Example: Load and prepare Wikipedia-like dataset
import urllib.request
import gzip

def download_and_extract(url, output_file):
    """Download dataset"""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_file)

# Download small dataset
download_and_extract(
    "https://example.com/data.txt.gz",
    "data/downloaded.txt.gz"
)

# Extract
import shutil
with gzip.open('data/downloaded.txt.gz', 'rb') as f_in:
    with open('data/downloaded.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```

---

## Custom Datasets

### Creating Instruction-Following Dataset

**Format: Q&A pairs**

```csv
instruction,input,output
What is Python?,An interpreted high-level programming language
What is a list?,A collection of items in Python
How do I loop?,Use for loop to iterate over items
```

**Preparation:**

```python
from gpt_lib.data import create_instruction_following_dataset

# Create dataset
dataset, tokenizer = create_instruction_following_dataset(
    "data/instructions.csv"
)

# Use for training
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32)

# Train with this loader
trainer.train(loader, val_loader, epochs=10)
```

### Domain-Specific Dataset

```python
# Example: Create medical domain dataset

import os
from pathlib import Path

def create_domain_dataset(domain_name, input_files):
    """Create domain-specific dataset"""
    
    combined = []
    
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            combined.append(f.read())
    
    output_file = f"data/{domain_name}_data.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(combined))
    
    return output_file

# Create medical dataset
medical_files = [
    "medical_papers/paper1.txt",
    "medical_papers/paper2.txt",
]

dataset_file = create_domain_dataset("medical", medical_files)
```

---

## Data Splitting

### Train/Validation Split

```python
from torch.utils.data import DataLoader, random_split
from gpt_lib.dataset import TextDataset

# Load dataset
dataset = TextDataset("data/text.txt", block_size=128)

# Split 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}")
print(f"Val size: {val_size}")
```

### Custom Train/Val/Test Split

```python
def split_data(text_file, train_ratio=0.7, val_ratio=0.15):
    """Custom data splitting"""
    
    with open(text_file) as f:
        lines = f.readlines()
    
    # Shuffle
    import random
    random.shuffle(lines)
    
    # Split
    n = len(lines)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]
    
    # Save splits
    with open('data/train.txt', 'w') as f:
        f.writelines(train_data)
    
    with open('data/val.txt', 'w') as f:
        f.writelines(val_data)
    
    with open('data/test.txt', 'w') as f:
        f.writelines(test_data)
    
    print(f"Train: {len(train_data)} lines")
    print(f"Val: {len(val_data)} lines")
    print(f"Test: {len(test_data)} lines")

# Use
split_data('data/all_text.txt')
```

---

## Cleaning Pipeline

### Complete Text Cleaning Script

```python
import re
import unicodedata

def clean_and_prepare_data(input_file, output_file):
    """Complete data cleaning pipeline"""
    
    # Read
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Fix multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Fix multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}', '[PHONE]', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown headers but keep content
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra punctuation
    text = re.sub(r'[!?]{2,}', '!', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Write
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Cleaned: {input_file} → {output_file}")
    print(f"Original size: {len(text):,} chars")

# Usage
clean_and_prepare_data('raw_data.txt', 'cleaned_data.txt')
```

---

## Best Practices

### DO's ✅

1. **Start small**: Test with 1-10MB before scaling
2. **Monitor quality**: Check data quality before training
3. **Clean consistently**: Apply same cleaning to train/val
4. **Use appropriate size**: Match model size to data volume
5. **Track metadata**: Keep records of data sources
6. **Validate splits**: Ensure no data leakage between splits
7. **Version control**: Keep track of data versions

### DON'Ts ❌

1. **Don't use test data in training**: Causes overfitting
2. **Don't mix languages**: Confuses tokenizer
3. **Don't include binary data**: Causes encoding errors
4. **Don't forget to clean**: Garbage in = garbage out
5. **Don't use tiny datasets**: <1MB generally insufficient

### Recommended Workflow

```python
# 1. Collect data
# → Download from sources, consolidate

# 2. Analyze quality
analyze_data_quality("raw_data.txt")

# 3. Clean data
clean_and_prepare_data("raw_data.txt", "cleaned_data.txt")

# 4. Train tokenizer (if needed)
Tokenizer.train("cleaned_data.txt", vocab_size=4096)

# 5. Create dataset
dataset = TextDataset("cleaned_data.txt", block_size=128)

# 6. Split data
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# 7. Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 8. Train model
trainer.train(train_loader, val_loader, epochs=20)
```

---

## Common Issues

### Issue: "Encoding Error"

**Cause:** File not UTF-8 encoded

**Solution:**
```python
# Convert to UTF-8
with open('file.txt', 'r', encoding='latin-1') as f:
    text = f.read()

with open('file_utf8.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

### Issue: "Out of Memory During Training"

**Cause:** Dataset too large for RAM

**Solution:**
```python
# Use smaller subset for testing
dataset = TextDataset("data.txt", block_size=128)
# Limit to first N samples
limited_dataset = Subset(dataset, range(10000))
```

### Issue: "Tokenizer Produces Many Unknown Tokens"

**Cause:** Vocabulary too small or data format incompatible

**Solution:**
```python
# Retrain with larger vocabulary
Tokenizer.train(
    "data.txt",
    vocab_size=8192,  # Increase from 4096
    output_path="larger_tokenizer.json"
)
```

---

**Last Updated:** June 20, 2026  
**Version:** 1.0
