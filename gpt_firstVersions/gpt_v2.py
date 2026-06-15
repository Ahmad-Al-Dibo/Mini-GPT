import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====================================================
# CONFIG
# ====================================================

BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
EPOCHS = 20
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================
# DATA
# ====================================================

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()

vocab = sorted(set(tokens))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

vocab_size = len(vocab)
data = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)

# ====================================================
# DATASET
# ====================================================

class DatasetLM(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - BLOCK_SIZE

    def __getitem__(self, idx):
        x = self.data[idx:idx+BLOCK_SIZE]
        y = self.data[idx+1:idx+BLOCK_SIZE+1]
        return x, y

loader = DataLoader(DatasetLM(data), batch_size=BATCH_SIZE, shuffle=True)

# ====================================================
# MULTI-HEAD ATTENTION
# ====================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)

# ====================================================
# FEED FORWARD
# ====================================================

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

# ====================================================
# TRANSFORMER BLOCK (GPT-2 STYLE PRE-NORM)
# ====================================================

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ====================================================
# GPT-2 MINI MODEL
# ====================================================

class GPT2Mini(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.Sequential(*[
            Block(EMBED_DIM, N_HEADS)
            for _ in range(N_LAYERS)
        ])

        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.head(x)

# ====================================================
# INIT
# ====================================================

model = GPT2Mini().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ====================================================
# TRAIN
# ====================================================

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        B, T, V = logits.shape

        loss = loss_fn(
            logits.view(B*T, V),
            y.view(B*T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss:.4f}")



# ====================================================
# GENERATE
# ====================================================

def generate(model, prompt, max_new_tokens=30):
    model.eval()

    words = prompt.lower().split()
    idx = [stoi[w] for w in words if w in stoi]

    if len(idx) == 0:
        idx = [0]

    idx = torch.tensor([idx], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -BLOCK_SIZE:]

            logits = model(idx_cond)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_id], dim=1)

    return " ".join([itos[i.item()] for i in idx[0]])

# test
print(generate(model, "python is"))