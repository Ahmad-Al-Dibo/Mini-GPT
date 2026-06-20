# Architecture Deep Dive

Technical deep-dive into MiniGPT's transformer architecture.

## Table of Contents

1. [Overview](#overview)
2. [Transformer Basics](#transformer-basics)
3. [MiniGPT Architecture](#minigpt-architecture)
4. [Self-Attention Mechanism](#self-attention-mechanism)
5. [Multi-Head Attention](#multi-head-attention)
6. [Positional Encoding](#positional-encoding)
7. [Feed-Forward Networks](#feed-forward-networks)
8. [Layer Normalization](#layer-normalization)
9. [Causal Masking](#causal-masking)
10. [Mathematical Formulations](#mathematical-formulations)

---

## Overview

MiniGPT implements a decoder-only transformer architecture optimized for autoregressive text generation. The key principle is simplicity without sacrificing core functionality.

### Architecture Stack

```
Input Tokens
    ↓
Token Embedding (embeddings from token IDs)
    ↓
Positional Embedding (add position information)
    ↓
Stack of N Transformer Blocks:
    ├─ Self-Attention Layer (with causal mask)
    ├─ Feed-Forward Layer
    ├─ Layer Normalization & Residual Connections
    ↓
Output Linear Projection
    ↓
Logits (vocab_size probabilities)
    ↓
Softmax → Token Probabilities
    ↓
Sample Next Token (for generation)
```

---

## Transformer Basics

### What is a Transformer?

A transformer is a neural network architecture based on the self-attention mechanism. It processes sequences in parallel (unlike RNNs) while maintaining positional context.

**Key Advantages:**
- Parallelizable (all positions computed simultaneously)
- Long-range dependencies (each token sees all previous tokens)
- Interpretable attention weights
- Scalable to large datasets

### Decoder-Only Architecture (GPT-style)

MiniGPT uses a **decoder-only** transformer because:
- **Efficiency**: Only one stack of blocks needed
- **Autoregressive**: Naturally suited for next-token prediction
- **Simplicity**: Easier to understand and implement

### Alternatives

| Type | Use Case | Example |
|------|----------|---------|
| Encoder-only | Text classification | BERT |
| Decoder-only | Text generation | GPT, MiniGPT |
| Encoder-Decoder | Translation | T5, Transformer |

---

## MiniGPT Architecture

### Layer Structure

```python
class MiniGPT(nn.Module):
    def __init__(self, config):
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(config.block_size, config.embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.num_blocks)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
```

### Forward Pass (Training)

```python
def forward(self, token_ids: Tensor) -> Tensor:
    batch_size, seq_length = token_ids.shape
    
    # 1. Embed tokens
    token_emb = self.token_embedding(token_ids)  # (B, T, D)
    
    # 2. Add positional embeddings
    pos_ids = torch.arange(seq_length, device=token_ids.device)
    pos_emb = self.positional_embedding(pos_ids)  # (T, D)
    x = token_emb + pos_emb  # (B, T, D)
    
    # 3. Apply dropout
    x = self.dropout(x)
    
    # 4. Apply transformer blocks
    for block in self.blocks:
        x = block(x)  # (B, T, D)
    
    # 5. Project to vocabulary
    logits = self.output_layer(x)  # (B, T, vocab_size)
    
    return logits
```

### Forward Pass (Generation)

```python
def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
    """Generate tokens autoregressively"""
    for _ in range(max_new_tokens):
        # Truncate to block_size (context window)
        idx_cond = idx[:, -self.config.block_size:]
        
        # Forward pass
        logits = self(idx_cond)  # (B, T, vocab_size)
        
        # Get last token's logits
        logits = logits[:, -1, :]  # (B, vocab_size)
        
        # Sample next token
        idx_next = torch.multinomial(
            F.softmax(logits, dim=-1),
            num_samples=1
        )  # (B, 1)
        
        # Append to sequence
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx
```

---

## Self-Attention Mechanism

### What is Attention?

Attention computes a weighted sum of all input values, where weights are determined by the relevance between queries and keys.

### Scaled Dot-Product Attention

```python
def attention(Q: Tensor, K: Tensor, V: Tensor, 
              mask: Tensor = None) -> Tensor:
    """
    Q: Query matrix (B, T, D)
    K: Key matrix (B, T, D)
    V: Value matrix (B, T, D)
    mask: Causal mask to prevent attending to future tokens
    """
    
    # 1. Compute attention scores
    scores = Q @ K.transpose(-2, -1) / sqrt(D)  # (B, T, T)
    
    # 2. Apply causal mask (prevent attending to future)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Apply softmax to get attention weights
    weights = softmax(scores, dim=-1)  # (B, T, T)
    
    # 4. Apply attention to values
    output = weights @ V  # (B, T, D)
    
    return output, weights
```

### Mathematical Formulation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$, $K$, $V$ are Query, Key, Value matrices
- $d_k$ is the dimension of keys (scaled by $\sqrt{d_k}$ for stability)

### Implementation Details

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, 
                 block_size: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.W_k = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.W_v = nn.Linear(embed_dim, embed_dim)  # Value projection
        self.W_o = nn.Linear(embed_dim, embed_dim)  # Output projection
        
        # Causal mask (upper triangular matrix)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (B, H, T, d_h)
        
        # Similar for K and V
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # (B, H, T, d_h)
        
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)  # (B, H, T, d_h)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.mask[:, :seq_length, :seq_length] == 0,
            float('-inf')
        )
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        attn_output = weights @ V  # (B, H, T, d_h)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2)  # (B, T, H, d_h)
        attn_output = attn_output.contiguous().view(
            batch_size, seq_length, self.embed_dim
        )
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output
```

---

## Multi-Head Attention

### Why Multiple Heads?

Different attention heads can learn different types of relationships:
- Head 1: Attends to subject tokens
- Head 2: Attends to adjacent tokens
- Head 3: Attends to all tokens equally
- etc.

### Head Configuration

```
Input: (B, T, embed_dim=512)
    ↓
Split into 8 heads: (B, 8, T, 64)
    ↓
Attention on each head
    ↓
Concatenate heads: (B, T, 512)
    ↓
Output projection: (B, T, 512)
```

### Code Representation

```python
num_heads = 8
embed_dim = 512
head_dim = embed_dim // num_heads  # 64

# Each head processes 64-dimensional subspace
# Heads run in parallel for efficiency
```

---

## Positional Encoding

### The Problem

Without positional information, transformer treats these identically:
- "Alice loves Bob"
- "Bob loves Alice"

Both would have the same token embeddings + attention pattern.

### Solution: Positional Embeddings

Add position-aware embeddings to token embeddings:

```python
# Token embedding
token_emb = embedding_layer(token_id)  # (D,)

# Positional embedding
pos_emb = positional_embedding(position)  # (D,)

# Combined
combined = token_emb + pos_emb  # (D,)
```

### Learned vs Fixed Positional Embeddings

**MiniGPT uses Learned Embeddings:**
```python
self.positional_embedding = nn.Embedding(block_size, embed_dim)
```

**Alternative: Sinusoidal (from original Transformer):**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Visualization

```
Position 0: [0.1, -0.2, 0.3, ...]
Position 1: [0.15, -0.18, 0.32, ...]
Position 2: [0.2, -0.16, 0.34, ...]
...
Position T: [0.5, -0.1, 0.5, ...]
```

---

## Feed-Forward Networks

### Structure

Each transformer block contains a position-wise FFN:

```
Linear(embed_dim → 4*embed_dim)
    ↓
ReLU Activation
    ↓
Dropout
    ↓
Linear(4*embed_dim → embed_dim)
```

### Code

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
```

### Why 4x Expansion?

- MLP needs enough capacity to learn complex transformations
- 4x is empirically found to work well
- Trade-off between expressiveness and efficiency

---

## Layer Normalization

### Pre-Norm vs Post-Norm

MiniGPT uses **Pre-Normalization** (applied before sublayers):

```python
class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, block_size, dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # Pre-Norm: normalize before sublayer
        x = x + self.attn(self.ln1(x))      # Attention block
        x = x + self.ffn(self.ln2(x))       # FFN block
        
        return x
```

### Benefits of Pre-Norm

- **Stability**: Prevents gradient explosion in deep networks
- **Better training**: Convergence is more stable
- **Residual flow**: Gradients can flow directly through residuals

### Comparison

```
Post-Norm (GPT-2):
x → Attn → LN → Add → FFN → LN → Add → out

Pre-Norm (MiniGPT):
x → LN → Attn → Add → LN → FFN → Add → out
```

---

## Causal Masking

### The Problem

Without masking, during training position 0 could attend to positions 1, 2, 3, etc., which is cheating during generation.

### Solution: Causal Mask

Mask prevents attention to future positions:

```
Position 0: can attend to [0] only
Position 1: can attend to [0, 1] only
Position 2: can attend to [0, 1, 2] only
...
Position T: can attend to [0, 1, ..., T] only
```

### Attention Mask Pattern

```
Mask (lower triangular):
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

Applied to attention scores:
- 1 means "keep attention score"
- 0 means "set to -∞ (becomes 0 after softmax)"
```

### Implementation

```python
def create_causal_mask(seq_length, device):
    """Create lower triangular matrix for causal masking"""
    mask = torch.tril(
        torch.ones(seq_length, seq_length, device=device)
    )
    return mask  # (T, T)

# During attention
scores = Q @ K.T / sqrt(d_k)
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = softmax(scores, dim=-1)
```

---

## Mathematical Formulations

### Token Embedding Lookup

$$\text{emb}(i) = E_{token}[i]$$

Where $E_{token}$ is the embedding matrix.

### Positional Addition

$$x_0 = \text{emb}(i) + \text{emb}_{pos}(t)$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Feed-Forward

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

### Transformer Block

$$x' = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$
$$x'' = x' + \text{FFN}(\text{LayerNorm}(x'))$$

### Next Token Prediction

$$p(\text{token}_t | \text{tokens}_{1:t-1}) = \text{softmax}(\text{logits}_t)$$

Where logits come from the output projection layer.

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| Pre-Norm | Better training stability |
| Causal Masking | Prevents cheating during training |
| Learned Positional Embeddings | Simpler than sinusoidal |
| Decoder-Only | Simpler, efficient for generation |
| Residual Connections | Enables deep networks |
| Layer Normalization | Gradient stability |

---

## Computational Complexity

### Time Complexity

- **Self-Attention**: O(T²·D) where T=sequence length, D=embed_dim
- **Feed-Forward**: O(T·D²)
- **Per Block**: O(T²·D + T·D²) ≈ O(T²·D) for typical D
- **Total (N blocks)**: O(N·T²·D)

### Memory Complexity

- **Activations**: O(N·T·D)
- **Attention Weights**: O(N·H·T²) where H=num_heads
- **Model Parameters**: O(N·D²)

### Optimization

MiniGPT is efficient because:
- Small embed_dim (256 vs 768 in BERT)
- Few blocks (4-8 vs 12-24)
- Small block_size (128 vs 512)

---

## Attention Visualization

### Example Attention Pattern

```
Query position 5 attends to:
Position: 0  1  2  3  4  5
Weight:   5% 3% 8% 12% 20% 52%

Attends most to itself (52%) and recent tokens (20%, 12%)
Less attention to old tokens
```

This is learned during training - model learns what to pay attention to!

---

## Extensions & Improvements

### Possible Enhancements

1. **Flash Attention**: O(T) memory instead of O(T²)
2. **Rotary Positional Embeddings**: Better rotation invariance
3. **Grouped Query Attention**: Fewer K,V projections
4. **ALiBi**: Attention with Linear Biases (no positional embeddings)
5. **Multi-Query Attention**: Single head K,V (faster inference)

---

## Resources

- **Attention Is All You Need**: Original Transformer paper
- **Language Models are Unsupervised Multitask Learners**: GPT-2 paper
- **d2l.ai**: Interactive tutorials on transformers

---

**Last Updated:** June 20, 2026  
**Difficulty:** Advanced
