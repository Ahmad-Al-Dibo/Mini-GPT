# Tests Directory

Unit tests for MiniGPT components.

## 🧪 Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_model.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=src
```

## 📁 Test Files

- **test_config.py** - Configuration tests
- **test_model.py** - Model architecture tests
- **test_trainer.py** - Training loop tests
- **test_inference.py** - Inference tests
- **test_data.py** - Data loading tests
- **test_tokenizer.py** - Tokenization tests

## ✅ What's Tested

### Model
- [ ] Model initialization
- [ ] Forward pass
- [ ] Generation
- [ ] Parameter counts

### Training
- [ ] Training loop
- [ ] Validation
- [ ] Checkpointing
- [ ] Early stopping

### Inference
- [ ] Model loading
- [ ] Text generation
- [ ] Batch inference
- [ ] Sampling methods

### Data
- [ ] Dataset loading
- [ ] Token processing
- [ ] Train/val splitting
- [ ] Batch loading

### Configuration
- [ ] Config creation
- [ ] Parameter validation
- [ ] Config serialization

## 🚀 Writing Tests

### Test Template
```python
# tests/test_myfeature.py
import pytest
from src.mymodule import MyClass

def test_basic_functionality():
    obj = MyClass()
    result = obj.method()
    assert result == expected

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

## 📊 Test Coverage

Check test coverage:
```bash
pytest tests/ --cov=src --cov-report=html
# View: htmlcov/index.html
```

**Target:** > 80% coverage

## 🔗 Related

- Source Code: `../src/`
- Examples: `../examples/`
- Documentation: `../docs/`

## ⚠️ Before Committing

Always run:
```bash
pytest tests/ -v
```

All tests must pass before merging!

---

**See:** `../docs/02_FULL_DOCUMENTATION.md#testing`
