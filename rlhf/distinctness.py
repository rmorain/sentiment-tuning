# Load csv
import csv

import pandas as pd
import pudb
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

tqdm.pandas()


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


csv_filename = "test_results/7ure2edv.csv"
df = pd.read_csv(
    csv_filename,
    names=[
        "dataset_name",
        "prefix",
        "prompt",
        "text",
        "prediction",
        "target",
        "reward",
    ],
)

for dataset_name in df.dataset_name.unique():
    for target in df.target.unique():
        selection = df.loc[(df.dataset_name == dataset_name) & df.target == target]
        selected_text = selection.text
        text_distictness = distinctness(selected_text)
        print(f"{dataset_name}_{target} Distinctness: {text_distictness}")
