import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from arclm import load_model


model = load_model("models/arclm10M.pth")
prompt = "The game began development in 2010"

output = model.generator.generate(
    prompt,
    max_new_tokens=80,
)
print(output)


print(model.predict(prompt, max_new_tokens=300, top_p=0.9, top_k = 100)) 
