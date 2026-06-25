from src.miniGPT.inference import LoadedModel, load_model
import torch

model = load_model(model_path = "models/Meduim-T.pth", device = "cuda" if torch.cuda.is_available() else "cpu")


with open("data/bookcorpus.txt", "r") as f:
    text = f.read()
prompt = "fall down to the "
print(prompt, "\n")

result = model.predict(prompt, max_new_tokens=1)
print(result, "\n")

# try with all temperatures
for temp in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]: # temrature values are used to control the randomness of predictions. Lower values make the model more deterministic, while higher values increase randomness.
    result = model.predict(prompt, max_new_tokens=300, temperature=temp)
    print(f"Temperature: {temp}")
    print(result, "\n")

