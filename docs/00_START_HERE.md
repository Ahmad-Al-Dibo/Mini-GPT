# 📚 MiniGPT Documentation - START HERE

Welcome! This is your entry point to MiniGPT documentation.

## ⚡ Quick Navigation

### 👤 I'm New - Get Me Started (5 min)
👉 **Read:** [`01_QUICK_START.md`](01_QUICK_START.md)

### 👨‍💻 I'm a Developer - Show Me the API
👉 **Read:** [`03_API_REFERENCE.md`](03_API_REFERENCE.md)

### 🧠 I Want to Learn Transformers  
👉 **Read:** [`04_ARCHITECTURE_DEEP_DIVE.md`](04_ARCHITECTURE_DEEP_DIVE.md)

### 📖 I Need Everything (Complete Guide)
👉 **Read:** [`02_FULL_DOCUMENTATION.md`](02_FULL_DOCUMENTATION.md)

### 📊 I Have Data to Train On
👉 **Read:** [`05_DATASET_GUIDE.md`](05_DATASET_GUIDE.md)

---

## 📁 Documentation Overview

| File | Purpose | Read Time |
|------|---------|-----------|
| **01_QUICK_START.md** | 5-min intro, installation, first example | 5 min |
| **02_FULL_DOCUMENTATION.md** | Complete reference guide | 1-2 hours |
| **03_API_REFERENCE.md** | API documentation for developers | 30 min |
| **04_ARCHITECTURE_DEEP_DIVE.md** | Technical deep-dive into transformers | 45 min |
| **05_DATASET_GUIDE.md** | Data preparation & management | 30 min |

---

## 🎯 Choose Your Path

### Path 1: Beginner (Want to use it quickly)
```
1. Start: 01_QUICK_START.md
2. Try: Run first example  
3. Learn: 02_FULL_DOCUMENTATION.md
```

### Path 2: Developer (Want to integrate it)
```
1. API: 03_API_REFERENCE.md
2. Examples: Check ../examples/
3. Reference: 02_FULL_DOCUMENTATION.md as needed
```

### Path 3: Researcher (Want to understand it)
```
1. Architecture: 04_ARCHITECTURE_DEEP_DIVE.md
2. Implementation: 03_API_REFERENCE.md
3. Full Guide: 02_FULL_DOCUMENTATION.md
```

### Path 4: Data Scientist (Want to train it)
```
1. Data: 05_DATASET_GUIDE.md
2. Training: 02_FULL_DOCUMENTATION.md#training-guide
3. Evaluation: 02_FULL_DOCUMENTATION.md#evaluation
```

---

## ⚠️ Requirements

- **Python:** 3.8+
- **PyTorch:** 1.9+ (or use `pip install -r ../requirements.txt`)
- **For GPU:** NVIDIA CUDA 11.0+ (optional, recommended)

---

## ✅ 60-Second Setup

```bash
# 1. Install
pip install -r ../requirements.txt

# 2. Test installation
python -c "import torch; print(torch.__version__)"

# 3. Generate text
python -c "from gpt_lib.inference import LoadedModel; m = LoadedModel('../output/MiniGPT.pth'); print(m.predict('Hello'))"

# 4. Start web interface
python ../app.py
# Open: http://localhost:5000
```

---

## 📞 Getting Help

| Question | Answer |
|----------|--------|
| How do I install it? | See 01_QUICK_START.md#installation |
| How do I generate text? | See 01_QUICK_START.md#generate-text |
| How do I train a model? | See 02_FULL_DOCUMENTATION.md#training-guide |
| How do I use the API? | See 03_API_REFERENCE.md |
| What data should I use? | See 05_DATASET_GUIDE.md |
| I'm getting an error | See 02_FULL_DOCUMENTATION.md#troubleshooting |

---

## 📚 All Documentation Files

```
docs/
├── 00_START_HERE.md                    ← YOU ARE HERE
├── 01_QUICK_START.md                   (5 min guide)
├── 02_FULL_DOCUMENTATION.md            (complete reference)
├── 03_API_REFERENCE.md                 (API docs)
├── 04_ARCHITECTURE_DEEP_DIVE.md        (technical details)
├── 05_DATASET_GUIDE.md                 (data guide)
│
└── legacy/                             (old docs)
    ├── README.md
    ├── PROJECT_MAP.md
    └── last-raport.md
```

---

## 🎓 Learning Tips

1. **Don't read everything at once** - Follow your path from above
2. **Code along** - Try examples as you read
3. **Bookmark the API** - Reference it while coding
4. **Use the table of contents** - Jump to what you need
5. **Check troubleshooting** - For common issues

---

**Ready? Pick a documentation file above and start reading!** 🚀

---

*Last Updated: June 20, 2026*  
*Documentation Version: 1.0*
