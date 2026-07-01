import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from arclm import train_model


SPECIAL_TOKENS = [
    "<|qa_start|>",
    "<|res_start|>",
    "<|end|>",
]


Path("models").mkdir(exist_ok=True)

result = train_model(
    mode="pretrain",
    data="data/data.txt",
    output="models/arclm_pretrained.pth",
    tokenizer_type="sentencepiece",
    user_defined_symbols=SPECIAL_TOKENS,
    max_vocab=5000,
    embed_dim=64,
    num_blocks=2,
    block_size=128,
    batch_size=128,
    num_epochs=4,
    checkpoint_batch_interval=100,
    validation_split=0.0,
    training_log_interval=10,
    early_stopping_patience= 2,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


prompt = "<|qa_start|> wat is minigpt? <|res_start|>"
encoded = result.tokenizer.encode_text(prompt)

print("Saved:", result.model_path)
print("Prompt ids:", encoded)
print("Prompt decoded:", result.tokenizer.decode(encoded))

for token in SPECIAL_TOKENS:
    token_id = result.tokenizer.encode_text(token)[0]
    print(token, token_id)
