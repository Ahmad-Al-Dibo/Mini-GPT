from src.miniGPT import load_model, prepare_data, Generator
from src.miniGPT.config import get_finetuning_config
from src.miniGPT.pipeline import build_trainer
from src.miniGPT.inference import LoadedModel, load_model
import torch


# Finetuning with defaults
config = get_finetuning_config(
    learning_rate=3e-4,
)  # 3 epochs, 2e-5 LR, frozen backbone
config.data_path = "data/data_qa.txt"

model = load_model("models/MeduimGPT_fineTuined.pth")
trainer = build_trainer(model, config)

data = prepare_data(config)
trainer = trainer.train(data.train_loader, epochs=10, val_loader=data.val_loader)
# trainer.save("models/MeduimGPT_fineTuined_1.pth", vocab_size=model.vocab_size, stoi=data.tokenizer.stoi, itos=data.tokenizer.itos)
generator = Generator(
        model,
        data.tokenizer.stoi if hasattr(data.tokenizer, 'stoi') else None,
        data.tokenizer.itos if hasattr(data.tokenizer, 'itos') else None,
        config.block_size,
        torch.device(config.device),
    )


model_l = LoadedModel(
    model=model,
    generator=generator,
    config=config,
    model_path="models/MeduimGPT_fineTuined.pth",
    device=torch.device(config.device)
)


txt1 = model.predict("<qa> wat is programming? </qa>", max_new_tokens=50, temperature=0.7)
txt2 = model.predict("wat is programming? ", max_new_tokens=50, temperature=0.7)

print(txt1)
print(txt2)
