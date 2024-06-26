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


csv_filename = "test_results/io7m8cfg.csv"
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

pos_df = df.loc[(df["target"] == 1)]
neg_df = df.loc[(df["target"] == 0)]

pos_dist = distinctness(pos_df.text)
neg_dist = distinctness(neg_df.text)
print("positive dist: ", pos_dist)
print("negative dist: ", neg_dist)
