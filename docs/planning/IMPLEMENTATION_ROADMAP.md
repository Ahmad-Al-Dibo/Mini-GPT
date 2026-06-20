# MiniGPT Implementation Roadmap

**Based on Technical Analysis**: MINIGPT_TECHNICAL_ANALYSIS.md  
**Created**: June 20, 2026  
**Status**: Action items for training capability enhancement

---

## 📊 Capability Assessment Summary

| Training Mode | Capability | Priority | Effort | Status |
|---|---|---|---|---|
| **Pretraining** | ✅ FULL | - | - | **READY NOW** |
| **Finetuning** | ⚠️ PARTIAL | **HIGH** | 7 hrs | NEEDS WORK |
| **Instruction Tuning** | ❌ NOT IMPLEMENTED | **CRITICAL** | 12 hrs | REQUIRES BUILD |

---

## 🎯 PRIORITY 1: FINETUNING SUPPORT (HIGH - 7 hours)

### Why Important
- Finetuning pretrained models is 10x faster than pretraining
- Essential for adapting to new domains/tasks
- Most practical use case after pretraining

### Required Implementations

#### 1.1 Layer Freezing & Unfreezing
**Severity**: 🔴 CRITICAL  
**Effort**: 1 hour  
**File to Modify**: `src/miniGPT/trainer.py`

**Add these methods to `Trainer` class:**
```python
def freeze_layers(self, pattern: str = "blocks", verbose: bool = True) -> int:
    """
    Freeze layers matching pattern.
    
    Args:
        pattern: Layer name pattern to freeze (e.g., "blocks", "token_embedding")
        verbose: Print frozen layer info
    
    Returns:
        Number of frozen parameters
    
    Examples:
        trainer.freeze_layers("blocks")  # Freeze all transformer blocks
        trainer.freeze_layers("embedding")  # Freeze embeddings
        trainer.freeze_layers("")  # Freeze all
    """
    frozen_count = 0
    for name, param in self.model.named_parameters():
        if pattern in name:
            param.requires_grad = False
            frozen_count += 1
            if verbose:
                print(f"  Froze: {name}")
    
    total = sum(1 for _ in self.model.parameters())
    print(f"Frozen {frozen_count}/{total} parameters")
    return frozen_count

def unfreeze_layers(self, pattern: str = None, verbose: bool = True) -> int:
    """Unfreeze all or specific layers."""
    unfrozen_count = 0
    for name, param in self.model.named_parameters():
        if pattern is None or pattern in name:
            param.requires_grad = True
            unfrozen_count += 1
            if verbose:
                print(f"  Unfroze: {name}")
    
    print(f"Unfrozen {unfrozen_count} parameters")
    return unfrozen_count

def get_frozen_layers_info(self) -> dict:
    """Return info about frozen layers."""
    frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
    total = sum(1 for _ in self.model.parameters())
    trainable = total - frozen
    return {
        "frozen": frozen,
        "trainable": trainable,
        "total": total,
        "trainable_pct": (trainable / total * 100) if total > 0 else 0
    }
```

**Integration**: Add to trainer after initialization:
```python
# After Trainer.__init__()
# Freeze backbone for finetuning if config specifies
if hasattr(self.config, 'freeze_backbone') and self.config.freeze_backbone:
    self.freeze_layers("blocks")
```

**Config additions** to `src/miniGPT/config.py`:
```python
freeze_backbone: bool = False  # Line after early_stopping_min_delta
freeze_embedding: bool = False  # Freeze token/position embeddings
```

---

#### 1.2 Discriminative Learning Rates
**Severity**: 🟠 HIGH  
**Effort**: 1.5 hours  
**File to Modify**: `src/miniGPT/pipeline.py`

**Problem**: All parameters use same learning rate. Finetuning typically needs:
- Lower LR for backbone (0.1x)
- Higher LR for new/head layers (1.0x)

**Current Code** (line 15):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),  # ← All same LR
    lr=config.learning_rate,
)
```

**Replace with:**
```python
def build_optimizer_with_discriminative_lr(
    model: MiniGPT,
    learning_rate: float,
    weight_decay: float,
    lr_multiplier: dict = None
) -> torch.optim.Optimizer:
    """
    Build optimizer with different LR for different layer types.
    
    Args:
        lr_multiplier: Dict like {'embeddings': 0.1, 'blocks': 0.1, 'head': 1.0}
                      Multiplies base LR for each group
    """
    if lr_multiplier is None:
        lr_multiplier = {}
    
    param_groups = []
    
    # Group 1: Embeddings
    embedding_params = list(model.token_embedding.parameters()) + \
                      list(model.position_embedding.parameters())
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': learning_rate * lr_multiplier.get('embeddings', 1.0),
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })
    
    # Group 2: Transformer blocks
    block_params = []
    for i, block in enumerate(model.blocks):
        block_params.extend(block.parameters())
    if block_params:
        param_groups.append({
            'params': block_params,
            'lr': learning_rate * lr_multiplier.get('blocks', 1.0),
            'weight_decay': weight_decay,
            'name': 'blocks'
        })
    
    # Group 3: Output head
    head_params = list(model.head.parameters())
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': learning_rate * lr_multiplier.get('head', 1.0),
            'weight_decay': weight_decay,
            'name': 'head'
        })
    
    return torch.optim.AdamW(param_groups, lr=learning_rate)
```

**Integration** in `pipeline.py` (replace line 15):
```python
# Check if using discriminative learning rates
if hasattr(config, 'use_discriminative_lr') and config.use_discriminative_lr:
    lr_multiplier = getattr(config, 'lr_multiplier', {
        'embeddings': 0.1,
        'blocks': 0.1,
        'head': 1.0
    })
    optimizer = build_optimizer_with_discriminative_lr(
        model,
        config.learning_rate,
        config.weight_decay,
        lr_multiplier
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
```

**Config additions**:
```python
use_discriminative_lr: bool = False
lr_multiplier: dict = None  # Set in init as {'embeddings': 0.1, 'blocks': 0.1, 'head': 1.0}
```

---

#### 1.3 Warmup Learning Rate Scheduler
**Severity**: 🟠 HIGH  
**Effort**: 1 hour  
**File**: Integrate existing `LearningRateScheduler` from `src/miniGPT/regularization.py`

**Current Issue**: Scheduler exists but NOT used in training loop.

**Integration** in `src/miniGPT/trainer.py`:

Add to `Trainer.__init__()`:
```python
# After optimizer creation
self.lr_scheduler = None
if hasattr(config, 'use_lr_scheduler') and config.use_lr_scheduler:
    from src.miniGPT.regularization import LearningRateScheduler
    self.lr_scheduler = LearningRateScheduler(
        optimizer=self.optimizer,
        strategy=getattr(config, 'lr_scheduler_strategy', 'cosine'),
        total_epochs=config.epochs,
        warmup_epochs=getattr(config, 'warmup_epochs', 1)
    )
```

Add in `Trainer.train()` after epoch validation (line ~148):
```python
# Learning rate scheduling
if self.lr_scheduler is not None:
    self.lr_scheduler.step(epoch)
    current_lr = self.optimizer.param_groups[0]['lr']
    print(f"  LR: {current_lr:.2e}")
```

**Config additions**:
```python
use_lr_scheduler: bool = False
lr_scheduler_strategy: str = "cosine"  # "cosine", "linear", "exponential"
warmup_epochs: int = 1
```

---

#### 1.4 Finetuning-Specific Config Presets
**Severity**: 🟡 MEDIUM  
**Effort**: 0.5 hours  
**File**: `src/miniGPT/config.py`

**Add functions**:
```python
def get_finetuning_config(
    pretrained_model_path: str,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,  # Small LR for finetuning
    freeze_backbone: bool = True,
    use_discriminative_lr: bool = True,
    **kwargs
) -> 'Config':
    """
    Get config optimized for finetuning pretrained models.
    
    Typical finetuning settings:
    - Small number of epochs (3-5)
    - Small learning rate (1e-5 to 5e-5)
    - Freeze backbone
    - Use discriminative learning rates
    """
    config = Config(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
        use_discriminative_lr=use_discriminative_lr,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
        **kwargs
    )
    return config

def get_instruction_tuning_config(
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    **kwargs
) -> 'Config':
    """Get config optimized for instruction tuning."""
    config = get_finetuning_config(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
    return config
```

---

### Usage Example
```python
from src.miniGPT import Config, Trainer, load_model
from src.miniGPT.config import get_finetuning_config

# Load pretrained model
model = load_model("models/MiniGPT.pth")

# Create finetuning config
config = get_finetuning_config(
    pretrained_model_path="models/MiniGPT.pth",
    num_epochs=3,
    learning_rate=2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True
)

# Create trainer and freeze backbone
trainer = Trainer(model, config)
trainer.freeze_layers("blocks")  # Freeze blocks, train head

# Finetune on new dataset
result = trainer.train(train_loader, val_loader)
```

---

## 🎯 PRIORITY 2: INSTRUCTION TUNING SUPPORT (CRITICAL - 12 hours)

### Why Important
- Instruction-following is key feature of modern LLMs
- Separate from general pretraining
- Critical for deployment

### Required Implementations

#### 2.1 Instruction Dataset Class
**Severity**: 🔴 CRITICAL  
**Effort**: 2 hours  
**File**: Create `src/miniGPT/instruction_dataset.py`

```python
"""
Instruction-following dataset for supervised fine-tuning.
Format: instruction | response
Loss is only computed on response tokens.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following training.
    
    Format: [(instruction, response), ...]
    Loss masked to only response tokens
    
    Example:
        instructions = ["What is 2+2?", "Explain photosynthesis"]
        responses = ["The answer is 4", "Photosynthesis is..."]
        dataset = InstructionDataset(instructions, responses, tokenizer, 256)
        
        for x, mask in dataset:
            x.shape = (256,)  # Input tokens
            mask.shape = (256,)  # 0 for instruction, 1 for response
    """
    
    def __init__(
        self,
        instructions: List[str],
        responses: List[str],
        tokenizer,
        block_size: int = 256,
        instruction_token: str = "<|instruction|>",
        response_token: str = "<|response|>",
        pad_token: int = 0,
        verbose: bool = False
    ):
        """
        Args:
            instructions: List of instruction strings
            responses: List of response strings (same length as instructions)
            tokenizer: Tokenizer with encode() method
            block_size: Maximum sequence length
            instruction_token: Special token for instruction start
            response_token: Special token for response start
            pad_token: Token ID for padding
        """
        assert len(instructions) == len(responses), \
            f"Mismatched lengths: {len(instructions)} != {len(responses)}"
        
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.instruction_token = instruction_token
        self.response_token = response_token
        self.pad_token = pad_token
        
        # Encode all instruction-response pairs
        self.data = []
        self.response_masks = []
        
        for i, (instr, resp) in enumerate(zip(instructions, responses)):
            # Format: <|instruction|> instruction <|response|> response
            formatted = f"{instruction_token} {instr} {response_token} {resp}"
            
            try:
                tokens = self.tokenizer.encode(formatted)
            except:
                # Handle if tokenizer has different interface
                tokens = self.tokenizer.encode_text(formatted)
            
            # Find response start (length of instruction part)
            instr_formatted = f"{instruction_token} {instr} {response_token} "
            try:
                instr_tokens = self.tokenizer.encode(instr_formatted)
            except:
                instr_tokens = self.tokenizer.encode_text(instr_formatted)
            
            response_start = len(instr_tokens)
            
            # Truncate or pad
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
                response_start = min(response_start, block_size)
            
            # Create mask: 0 for instruction/prompt, 1 for response
            mask = [0] * response_start + [1] * (len(tokens) - response_start)
            
            # Pad to block_size
            if len(tokens) < block_size:
                pad_len = block_size - len(tokens)
                tokens = tokens + [pad_token] * pad_len
                mask = mask + [0] * pad_len  # Padding not in loss
            
            self.data.append(torch.tensor(tokens, dtype=torch.long))
            self.response_masks.append(torch.tensor(mask, dtype=torch.float))
            
            if verbose and i < 3:  # Print first 3 examples
                print(f"Example {i}:")
                print(f"  Instruction: {instr[:50]}...")
                print(f"  Response: {resp[:50]}...")
                print(f"  Response start idx: {response_start}")
                print()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input tokens (block_size,)
            mask: Response mask (block_size,) - 1 where loss should be computed
        """
        return self.data[idx], self.response_masks[idx]


def create_instruction_dataloader(
    instructions: List[str],
    responses: List[str],
    tokenizer,
    block_size: int = 256,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create instruction dataset and dataloader."""
    dataset = InstructionDataset(
        instructions, responses, tokenizer,
        block_size=block_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
```

---

#### 2.2 Masked Loss Computation
**Severity**: 🔴 CRITICAL  
**Effort**: 1.5 hours  
**File**: Modify `src/miniGPT/trainer.py`

**Add new method**:
```python
def compute_masked_loss(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute loss only on masked positions (response tokens).
    
    Args:
        logits: (B*T, V) model output
        targets: (B*T,) target tokens
        mask: (B, T) where 1 = compute loss, 0 = ignore
    
    Returns:
        Scalar loss
    """
    B, T = mask.shape
    V = logits.shape[-1]
    
    # Reshape to (B*T,)
    mask_flat = mask.view(B * T)
    
    # Standard CE loss
    loss = self.criterion(logits, targets)  # (B*T,)
    
    # Apply mask: set loss to 0 where mask=0
    masked_loss = loss * mask_flat
    
    # Average only over masked positions
    return masked_loss.sum() / (mask_flat.sum() + 1e-8)
```

**Modify training loop** in `Trainer.train()`:

Current (line 86-91):
```python
loss = self.criterion(
    logits.view(B * T, V),
    y.view(B * T)
)
```

Change to:
```python
# Check if batch contains response masks (for instruction tuning)
if isinstance(batch, dict) and 'mask' in batch:
    # Instruction tuning: compute loss only on response
    loss = self.compute_masked_loss(
        logits.view(B * T, V),
        y.view(B * T),
        batch['mask']  # (B, T)
    )
else:
    # Regular training: all tokens
    loss = self.criterion(
        logits.view(B * T, V),
        y.view(B * T)
    )
```

---

#### 2.3 InstructionTuner Class
**Severity**: 🟠 HIGH  
**Effort**: 2 hours  
**File**: `src/miniGPT/instruction_trainer.py`

```python
"""
Specialized trainer for instruction-following fine-tuning.
Extends Trainer with instruction-specific features.
"""

from typing import Optional, List
import torch
from src.miniGPT.trainer import Trainer
from src.miniGPT.instruction_dataset import InstructionDataset


class InstructionTuner(Trainer):
    """Trainer specialized for instruction-following models."""
    
    def prepare_instruction_data(
        self,
        instructions: List[str],
        responses: List[str],
        split_ratio: float = 0.9
    ) -> tuple:
        """
        Prepare instruction dataset from instruction-response pairs.
        
        Args:
            instructions: List of instruction strings
            responses: List of response strings
            split_ratio: Train/val split (0.9 = 90% train)
        
        Returns:
            (train_dataset, val_dataset)
        """
        n = len(instructions)
        split_idx = int(n * split_ratio)
        
        train_dataset = InstructionDataset(
            instructions[:split_idx],
            responses[:split_idx],
            self.tokenizer,
            block_size=self.config.block_size
        )
        
        val_dataset = InstructionDataset(
            instructions[split_idx:],
            responses[split_idx:],
            self.tokenizer,
            block_size=self.config.block_size
        )
        
        return train_dataset, val_dataset
    
    def train_sft(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None
    ) -> dict:
        """
        Train in supervised fine-tuning (SFT) mode.
        Same as regular training but with masked loss for responses.
        """
        return self.train(train_loader, val_loader, epochs)
    
    def generate_instruction_response(
        self,
        instruction: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response to instruction.
        
        Args:
            instruction: Input instruction
            max_new_tokens: Max generation length
            temperature: Sampling temperature
        
        Returns:
            Generated response text
        """
        formatted = f"<|instruction|> {instruction} <|response|>"
        
        # Encode instruction
        instr_tokens = self.tokenizer.encode(formatted)
        x = torch.tensor(instr_tokens).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(x)
                next_logits = logits[0, -1, :] / temperature
                next_token = torch.multinomial(
                    torch.softmax(next_logits, dim=-1), 1
                ).item()
                x = torch.cat([x, torch.tensor([[next_token]]).to(self.device)], dim=1)
        
        # Decode
        tokens = x[0].cpu().tolist()
        response = self.tokenizer.decode(tokens)
        
        # Extract response part
        if "<|response|>" in response:
            response = response.split("<|response|>")[1]
        
        return response.strip()
```

---

#### 2.4 Instruction Dataset Loading from Files
**Severity**: 🟡 MEDIUM  
**Effort**: 1 hour  
**File**: `src/miniGPT/instruction_dataset.py` (add functions)

```python
def load_instruction_dataset_from_csv(
    csv_path: str,
    instr_column: str = "instruction",
    response_column: str = "response"
) -> Tuple[List[str], List[str]]:
    """Load instruction-response pairs from CSV file."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    instructions = df[instr_column].tolist()
    responses = df[response_column].tolist()
    
    return instructions, responses


def load_instruction_dataset_from_jsonl(
    jsonl_path: str,
    instr_key: str = "instruction",
    response_key: str = "response"
) -> Tuple[List[str], List[str]]:
    """Load instruction-response pairs from JSONL file."""
    import json
    
    instructions, responses = [], []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            instructions.append(data[instr_key])
            responses.append(data[response_key])
    
    return instructions, responses
```

---

### Usage Example
```python
from src.miniGPT import load_model
from src.miniGPT.instruction_trainer import InstructionTuner
from src.miniGPT.config import get_instruction_tuning_config
from src.miniGPT.instruction_dataset import load_instruction_dataset_from_csv

# Load pretrained model
model = load_model("models/MiniGPT.pth")

# Create instruction tuner
config = get_instruction_tuning_config(num_epochs=3)
tuner = InstructionTuner(model, config)

# Load instruction data
instructions, responses = load_instruction_dataset_from_csv("data/instructions.csv")
train_dataset, val_dataset = tuner.prepare_instruction_data(instructions, responses)

# Create dataloaders with custom collate to handle masks
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train
tuner.train_sft(train_loader, val_loader)

# Generate response to new instruction
response = tuner.generate_instruction_response(
    "What is machine learning?"
)
print(response)
```

---

## 🎯 PRIORITY 3: REGULARIZATION INTEGRATION (MEDIUM - 5.5 hours)

### Why Important
- Prevent overfitting on small finetuning datasets
- Better generalization
- Production quality

### Required Implementations

#### 3.1 Integrate Learning Rate Scheduler
**Effort**: 1 hour  
**File**: `src/miniGPT/trainer.py`

Already documented in Priority 1 (Section 1.3)

---

#### 3.2 Integrate Generalization Monitor
**Effort**: 1 hour  
**File**: `src/miniGPT/trainer.py`

Add to `__init__()`:
```python
from src.miniGPT.regularization import GeneralizationMonitor

self.gen_monitor = GeneralizationMonitor()
```

Add in training loop (after validation):
```python
self.gen_monitor.update(
    train_loss=train_loss,
    val_loss=val_loss,
    epoch=epoch
)

if self.gen_monitor.is_overfitting():
    print(f"  ⚠️ Overfitting detected: {self.gen_monitor.get_report()}")
```

---

#### 3.3 Integrate Label Smoothing
**Effort**: 1.5 hours  
**File**: `src/miniGPT/trainer.py`

Add to `setup_regularization()` (new method):
```python
from src.miniGPT.regularization import LabelSmoothing

if getattr(self.config, 'use_label_smoothing', False):
    self.criterion = LabelSmoothing(
        num_classes=self.model.vocab_size,
        smoothing=getattr(self.config, 'label_smoothing_factor', 0.1)
    )
```

---

#### 3.4 Add Periodic Checkpointing
**Effort**: 1 hour  
**File**: `src/miniGPT/trainer.py`

```python
def save_checkpoint_if_interval(self, epoch: int):
    """Save checkpoint every N epochs."""
    interval = getattr(self.config, 'checkpoint_interval', 0)
    if interval > 0 and epoch % interval == 0:
        ckpt_path = f"{self.config.model_path}.epoch_{epoch}"
        self.save(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
```

---

## 📋 Implementation Order (Recommended)

### Week 1: Finetuning Essentials
1. **Day 1**: Layer freezing + unfreezing (1 hour)
2. **Day 2**: Discriminative learning rates (1.5 hours)
3. **Day 3**: LR warmup scheduler integration (1 hour)
4. **Day 4**: Finetuning config presets (0.5 hours)
5. **Day 5**: Testing & docs (1 hour)

**Total**: 5 hours | **Outcome**: Full finetuning support ✅

### Week 2: Instruction Tuning
1. **Day 1**: Instruction dataset class (2 hours)
2. **Day 2**: Masked loss computation (1.5 hours)
3. **Day 3**: InstructionTuner class (2 hours)
4. **Day 4**: File loading utilities (1 hour)
5. **Day 5**: Testing & examples (1.5 hours)

**Total**: 8 hours | **Outcome**: Full instruction tuning support ✅

### Week 3: Polish & Testing
1. **Day 1-2**: Integrate remaining regularization (2 hours)
2. **Day 3-5**: Testing, docs, examples (3 hours)

**Total**: 5 hours | **Outcome**: Production-ready training ✅

---

## 🚀 Implementation Dependencies

```
Finetuning Essentials (Week 1)
    ├── Layer Freezing ✅
    ├── Discriminative LR ✅
    └── LR Warmup ✅
         ↓
Instruction Tuning (Week 2)
    ├── Requires Finetuning ✅
    ├── Instruction Dataset ✅
    ├── Masked Loss ✅
    └── InstructionTuner ✅
         ↓
Advanced Features (Week 3)
    ├── Regularization Integration ✅
    ├── Monitoring ✅
    └── Periodic Checkpointing ✅
```

---

## 📊 Before vs After

### Before Implementation
```
Pretraining:     ✅ ✅ ✅ ✅ ✅ (FULL)
Finetuning:      🟡 (Limited - can load & resume only)
Inst. Tuning:    ❌ (None)
```

### After Implementation
```
Pretraining:     ✅ ✅ ✅ ✅ ✅ (FULL)
Finetuning:      ✅ ✅ ✅ ✅ ✅ (COMPLETE)
Inst. Tuning:    ✅ ✅ ✅ ✅ ✅ (COMPLETE)
Regularization:  ✅ ✅ ✅ (INTEGRATED)
```

---

## 📄 Documentation Updates Needed

1. **README.md**: Add finetuning & instruction tuning sections
2. **docs/**: Create tutorials for each training mode
3. **examples/**: Add example scripts for finetuning & instruction tuning
4. **API reference**: Document new classes and methods

---

## 🎯 Success Criteria

- [ ] Layer freezing works (manual test)
- [ ] Discriminative LRs work (verify different param groups)
- [ ] LR scheduler integrates without breaking training
- [ ] Finetuning config presets exist
- [ ] Instruction dataset class works
- [ ] Masked loss computes correctly
- [ ] InstructionTuner trains without errors
- [ ] All examples run successfully
- [ ] No regression on pretraining

---

**Status**: Ready to implement  
**Estimated Total Effort**: 18 hours  
**Estimated Timeline**: 2-3 weeks with full focus

See MINIGPT_TECHNICAL_ANALYSIS.md for detailed technical findings.
