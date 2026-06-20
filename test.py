from src.inference import LoadedModel
model = LoadedModel("models/mini_gpt.pth")  
result = model.predict("AI is", max_new_tokens=50)
print(result)