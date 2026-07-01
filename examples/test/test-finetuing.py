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
    "<|pad|>",
]


Path("models").mkdir(exist_ok=True)

result = train_model(
    mode="finetune",
    checkpoint="models/arclm_pretrained.pth",
    data="data/finetune.txt",
    output="models/arclm_finetuned.pth",
    user_defined_symbols=SPECIAL_TOKENS,
    num_epochs=200,
    batch_size=128,
    learning_rate=6.2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True,
    checkpoint_batch_interval=50,
    validation_split=0.06,
    early_stopping_patience= 2,
    training_log_interval=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print("Saved:", result.model_path)
for token in SPECIAL_TOKENS:
    token_id = result.tokenizer.encode_text(token)[0]
    print(token, token_id)
