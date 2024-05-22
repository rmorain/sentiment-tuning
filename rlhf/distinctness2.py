# Load csv
import csv

import pandas as pd
import pudb
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

tqdm.pandas()


def perplexity(row, tokenizer, model, device):
    text = row["text"]
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()


def distinctness(generations_data):
    dist1, dist2, dist3 = [], [], []
    total_words = 0
    unigrams, bigrams, trigrams = set(), set(), set()

    for gen in generations_data:
        o = gen.split(" ")
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + "_" + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + "_" + o[i + 1] + "_" + o[i + 2])

    if total_words == 0:
        return 0.0, 0.0, 0.0

    dist1 = len(unigrams) / total_words
    dist2 = len(bigrams) / total_words
    dist3 = len(trigrams) / total_words

    return dist1, dist2, dist3


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
csv_filename = "test_results/mmxbmk63.csv"
run_id = "mmxbmk63"
df = pd.read_csv(
    f"test_results/{run_id}.csv",
    names=["prefix", "prompt", "text", "prediction", "reward"],
)
df["text"] = df.apply(lambda x: x["text"][len(x["prompt"]) :], axis=1)

d = distinctness(df.text)
print("Dist-1", d[0])
print("Dist-2", d[1])
print("Dist-3", d[2])
