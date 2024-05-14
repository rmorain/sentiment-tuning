# Load csv
import csv

import pandas as pd
import pudb
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

tqdm.pandas()


def perplexity(text, tokenizer, model, device):
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


def perspective_api(client, generation):
    analyze_request = {
        "comment": {f"text": f"{generation}"},
        "requestedAttributes": {"TOXICITY": {}},
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

API_KEY = "AIzaSyB-oRhM-t8NVEYqAkTYsyN9r0W-Lt0D4Gg"

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


with open("test_results/7ure2edv.csv", "r", newline="") as csv_file:
    reader = csv.reader(csv_file)
    with open(
        "test_results/7ure2edv_perplexity.csv", "w", newline=""
    ) as csv_file_perplexity:
        writer = csv.writer(csv_file_perplexity)
        for row in reader:
            p = perplexity(row[3], tokenizer, model, device)
            row.append(p)
            writer.writerow(row)

df = pd.read_csv(
    "test_results/7ure2edv_perplexity.csv",
    names=[
        "dataset_name",
        "prefix",
        "prompt",
        "text",
        "prediction",
        "target",
        "reward",
        "perplexity",
    ],
)

# Compute perplexity by dataset
for dataset_name in df.dataset_name.unique():
    for target in df.target.unique():
        selection = df.loc[(df.dataset_name == dataset_name) & df.target == target]
        mean_perplexity = selection.perplexity.mean()
        print(f"{dataset_name}_{target} PPL: {mean_perplexity}")
