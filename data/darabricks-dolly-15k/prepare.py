import pandas as pd
import numpy
import sys


df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
df = pd.DataFrame(df)

only = {
    "response": "return only the response.",
    "json": "retrun it as json format.",
    "context": "explain"
}

"""
bsdhfbdsfbsd bdf bdsf b 




"""


if len(sys.argv) == 2:
    data = df.to_json(sys.argv[1])
else:
    raise ValueError("Expected exactly one output filename argument.")