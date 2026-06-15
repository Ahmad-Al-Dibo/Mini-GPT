import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ====================================================
# CONFIGURATION
# ====================================================

CONFIG = {
    "embed_dim": 64,
    "block_size": 8,
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-3,
    "num_blocks": 2,
    "model_path": "output/mini_gpt.pth",
    "data_path": "data/data.txt",
}

# ====================================================
# DATA LOADING
# ====================================================


def load_data(data_path):
    """Load and tokenize text data."""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text


def create_tokenizer(text, max_vocab=50000):
    """
    @ max_vocab: Maximum vocabulary size (including UNK token)

    Create tokenizer with UNK token for rare words.
    @ Returns vocab list, stoi dict, and itos dict.
    vocab[0] is reserved for UNK token.
    vocab: List of tokens in the vocabulary, with vocab[0] = "<UNK>".
    stoi: Dictionary mapping token to index, with stoi["<UNK>"] = 0.
    itos: Dictionary mapping index to token, with itos[0] = "<UNK>".
    <UNK> token is used for any token not in the top max_vocab-1 most common tokens.
    example:
    text = "hello world hello"
    vocab = ["<UNK>", "hello", "world"]
    stoi = {"<UNK>": 0, "hello": 1, "world": 2}
    itos = {0: "<UNK>", 1: "hello", 2: "world"}
    """
    from collections import Counter
    tokens = text.split()
    
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Keep only top max_vocab tokens
    most_common = token_counts.most_common(max_vocab - 1)  # -1 for UNK token
    vocab = ["<UNK>"] + [token for token, _ in most_common]
    
    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return vocab, stoi, itos 


def encode_data(tokens, stoi):
    """
    @ tokens: List of tokens to encode
    @ stoi: Dictionary mapping token to index
    Encode list of tokens into list of indices using stoi mapping.
    Tokens not in stoi will be mapped to the index of the UNK token.
    @ Returns list of indices corresponding to the input tokens.
    example:
    tokens = ["hello", "world", "foo"]
    stoi = {"<UNK>": 0, "hello": 1, "world": 2}
    encoded = [1, 2, 0]  # "foo" is not in stoi, so it maps to index 0 (UNK)
    """
    unk_idx = stoi.get("<UNK>", 0)  # Default to 0 if no UNK token
    return [stoi.get(token, unk_idx) for token in tokens if token]

# ====================================================
# DATASET
# ====================================================


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(
            self.data[idx : idx + self.block_size]
        )

        y = torch.tensor(
            self.data[idx + 1 : idx + self.block_size + 1]
        )

        return x, y


def create_dataloader(encoded_data, block_size, batch_size):
    """Create DataLoader for training."""
    dataset = TextDataset(encoded_data, block_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return loader

# ====================================================
# SELF ATTENTION
# ====================================================


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(C)

        mask = torch.tril(
            torch.ones(T, T)
        ).to(x.device)

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        return output


# ====================================================
# GPT BLOCK
# ====================================================


class GPTBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.attn = SelfAttention(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ff(
            self.norm2(x)
        )

        return x


# ====================================================
# MINI GPT
# ====================================================


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        block_size=8,
        num_blocks=2
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim
        )

        self.blocks = nn.Sequential(
            *[GPTBlock(embed_dim) for _ in range(num_blocks)]
        )

        self.ln = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            vocab_size
        )

        self.block_size = block_size

    def forward(self, idx):

        B, T = idx.shape

        tok_emb = self.token_embedding(
            idx
        )

        pos = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(
            pos
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits


# ====================================================
# TRAINING
# ====================================================


def train_model(model, loader, optimizer, criterion, epochs, device):
    """Train the model."""
    model.train()
    
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            B, T, V = logits.shape

            loss = criterion(
                logits.view(B * T, V),
                y.view(B * T)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Loss = {total_loss:.4f} | "
            f"Time = {epoch_time:.2f}s"
        )
    
    total_train_time = time.time() - train_start
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    print(f"\n[OK] Training completed in {hours}h {minutes}m {seconds}s")


# ====================================================
# SAVE MODEL
# ====================================================


def save_model(model, optimizer, vocab, stoi, itos, block_size, vocab_size, model_path):
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "vocab_size": vocab_size,
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")


# ====================================================
# LOAD MODEL
# ====================================================


def load_model(model_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint

# ====================================================
# GENERATION
# ====================================================


def generate(model, start_text, stoi, itos, block_size, device, max_new_tokens=20):
    """Generate text from start text."""
    model.eval()

    words = start_text.lower().split()

    idx = [
        stoi[w]
        for w in words
        if w in stoi
    ]

    idx = torch.tensor(
        [idx],
        device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits = model(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

    output = [
        itos[i.item()]
        for i in idx[0]
    ]

    return output


# ====================================================
# MAIN
# ====================================================


def main():
    
    """Main training pipeline."""
    
    main_start = time.time()
    
    # Setup device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load and prepare data
    text = load_data(CONFIG["data_path"])
    tokens = text.split()
    
    # Use sample of data for faster training
    sample_size = min(1000000, len(tokens))  # Limit to 1M tokens
    tokens = tokens[:sample_size]
    text = " ".join(tokens)
    
    vocab, stoi, itos = create_tokenizer(text, max_vocab=50000)
    encoded = encode_data(tokens, stoi)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")
    print(f"Data length: {len(encoded)}")

    # Create dataloader
    loader = create_dataloader(
        encoded,
        CONFIG["block_size"],
        CONFIG["batch_size"]
    )

    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=CONFIG["embed_dim"],
        block_size=CONFIG["block_size"],
        num_blocks=CONFIG["num_blocks"]
    )
    model = model.to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # Train model
    print("\nStarting training...")
    # if exists, load model
    if Path(CONFIG["model_path"]).exists():
        print("Loading existing model...")
        checkpoint = load_model(
            CONFIG["model_path"],
            device
        )
        
        # Check if vocab sizes match
        if checkpoint["vocab_size"] != vocab_size:
            print(f"[WARNING]  Vocab size mismatch!")
            print(f"   Checkpoint: {checkpoint['vocab_size']}, Current: {vocab_size}")
            print(f"   Starting fresh training...")
            train_model(
                model,
                loader,
                optimizer,
                criterion,
                CONFIG["epochs"],
                device
            )
        else:
            print("[OK] Vocab sizes match, loading checkpoint...")
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            vocab = checkpoint["vocab"]
            stoi = checkpoint["stoi"]
            itos = checkpoint["itos"]
            block_size = checkpoint["block_size"]
            vocab_size = checkpoint["vocab_size"]

    else:
        train_model(
            model,
            loader,
            optimizer,
            criterion,
            CONFIG["epochs"],
            device
        )

    # Save model
    print("\nSaving model...")
    save_model(
        model,
        optimizer,
        vocab,
        stoi,
        itos,
        CONFIG["block_size"],
        vocab_size,
        CONFIG["model_path"]
    )

    # Generate sample text
    print("\nGenerating sample text...")
    gen_start = time.time()
    sample_text = generate(
        model,
        "English is",
        stoi,
        itos,
        CONFIG["block_size"],
        device,
        max_new_tokens=20
    )
    gen_time = time.time() - gen_start
    print(f"Generated: {' '.join(sample_text)}")
    print(f"Generation time: {gen_time:.2f}s")
    
    # Total execution time
    total_time = time.time() - main_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\n{'='*50}")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*50}")



if __name__ == "__main__":
    main()
