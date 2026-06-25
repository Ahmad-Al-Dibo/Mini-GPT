from src.miniGPT.inference import load_model, SentencePieceTokenizer
from src.miniGPT.utils import PreModelBenchmark, BenchmarkMetrics
import torch
import time
import json

PROMPTS = {
    "story_easy": [
        "Once upon a time",
        "The old man looked at her and said",
        "She opened the door and saw",
        "The rain fell heavily as",
        "He took a deep breath and",
        "The captain walked into the room and",
        "She turned around slowly and",
    ],

    "story_development": [
        "The detective arrived at the crime scene. The first thing he noticed was",
        "When Sarah woke up, she realized that",
        "The spaceship landed on the planet, and the crew discovered",
        "The king called his advisors together because",
        "After ten years away from home, he finally returned and",
    ],

    "dialogue": [
        "\"Where have you been?\" she asked.",
        "\"I don't think that's a good idea,\" he said.",
        "\"What do you mean?\" she asked.",
        "The two friends sat quietly for a moment before one of them said,",
    ],

    "description": [
        "The castle stood on top of a hill overlooking",
        "The forest was silent except for",
        "The city at night looked like",
        "The laboratory was filled with",
        "The abandoned house smelled of",
    ],

    "suspense": [
        "The lights suddenly went out and",
        "Something was moving in the darkness behind her",
        "The radio crackled to life and a voice said",
        "He knew he was being followed because",
        "A strange noise came from the basement",
    ],

    "reasoning_basic": [
        "The advantages and disadvantages of artificial intelligence are",
        "The scientific method requires",
        "Explain why democracy is important",
        "The economic consequences of inflation include",
        "Why do airplanes stay in the air?",
    ],

    "reasoning_intermediate": [
        "Compare renewable energy and fossil fuels",
        "Explain the causes and effects of climate change",
        "Why is statistical sampling important in science?",
        "Describe the trade-offs between privacy and security",
        "Explain how vaccines help prevent disease",
    ],

    "instruction_following": [
        "List 5 benefits of exercise.",
        "Give three reasons to learn programming.",
        "Write a short summary of World War II.",
        "Provide a step-by-step plan for learning machine learning.",
        "Explain photosynthesis in simple terms.",
    ],

    "knowledge": [
        "Who was Albert Einstein?",
        "What is the capital of Japan?",
        "Explain how the internet works.",
        "What caused the Industrial Revolution?",
        "Describe the structure of a cell.",
    ],

    "coding": [
        "Write a Python function that computes the factorial of a number.",
        "Explain what a for loop does in Python.",
        "Write a Python class representing a bank account.",
        "What is the difference between a list and a tuple in Python?",
        "Write code to reverse a string.",
    ],

    "mathematics": [
        "Solve: 2x + 5 = 17",
        "What is the derivative of x^2?",
        "Explain the Pythagorean theorem.",
        "Find the next number: 2, 4, 8, 16, ?",
        "What is the difference between mean and median?",
    ],

    "long_context": [
        "Write a detailed explanation of how machine learning models are trained from data.",
        "Explain the history of computing from the 1940s to today.",
        "Describe the complete lifecycle of a software project.",
        "Explain how a transformer-based language model works.",
    ],

    "out_of_distribution": [
        "Write a detailed explanation of quantum mechanics.",
        "Explain the transformer architecture used in large language models.",
        "Derive Newton's second law from first principles.",
        "Compare PyTorch and TensorFlow for deep learning research.",
        "Explain the mathematical intuition behind backpropagation.",
    ],

    "creative_writing": [
        "Write the opening paragraph of a science fiction novel.",
        "Write a fantasy story about a forgotten kingdom.",
        "Invent a new superhero and describe their powers.",
        "Write a poem about the future of humanity.",
        "Create a mystery story set on a train.",
    ],
}


def run_compute(prompt:str, model:load_model, benchmark_matrix: BenchmarkMetrics, predict_tokens: int = 80):

    gen_start = time.time()
    output = model.predict(prompt, predict_tokens)
    gen_time = time.time() - gen_start

    compute_data = benchmark_matrix.compute(
        prompt=prompt,
        output= output,
        generation_time= gen_time
    )
    return {
        "computed": compute_data,
        "input": prompt,
        "output": output,
        "generation_time": gen_time
    }




device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(
    "models/Meduim-T.pth",
    device=device
)

benchmark = PreModelBenchmark(
    model=model,
    model_name="Medium-T",
    prompts=PROMPTS,
    temperatures=[0.1, 0.25, 0.5, 1.0],
    max_new_tokens=100
)

tokenizer = SentencePieceTokenizer(
    max_vocab=50_000
).load("data/tokenizer.json")

benchmark_matrix = BenchmarkMetrics(tokenizer=tokenizer)
# results = benchmark.run()

compute_output = run_compute(
    prompt= "The old man looked at her and said",
    model = model,
    benchmark_matrix = benchmark_matrix,
    predict_tokens=1_000
)

print(compute_output.get("computed"))

with open("Benchmark_Meduim-T_computed", "w", encoding="utf-8") as f:
    f.write(json.dumps(compute_output))


# benchmark.save_json(results)
# benchmark.save_txt(results)  # optional


