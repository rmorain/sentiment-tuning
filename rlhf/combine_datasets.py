import random

import pudb

from datasets import ClassLabel, Value, concatenate_datasets, load_dataset


def label(x):
    if x["label"] > 2:
        x["sentiment"] = 1
    if x["label"] < 2:
        x["sentiment"] = 0
    else:
        x["sentiment"] = random.randint(0, 1)
    return x


imdb_dataset = load_dataset("stanfordnlp/imdb")
sst_dataset = load_dataset("stanfordnlp/sst2")
# yelp_dataset = load_dataset("yelp_review_full")

ds_list = []

features = imdb_dataset["train"].features
features["label"] = ClassLabel(num_classes=2, names=["negative", "positive"])
for ds in imdb_dataset.values():
    ds = ds.cast(features)
    ds_list.append(ds)

for ds in sst_dataset.values():
    ds = ds.remove_columns(["idx"]).rename_column("sentence", "text")
    ds = ds.cast(features)
    ds_list.append(ds)

# for ds in yelp_dataset.values():
#     ds = ds.map(label, batched=False)
#     ds = ds.remove_columns(["label"]).rename_column("sentiment", "label")
#     ds = ds.cast(features)
#     ds_list.append(ds)

corpus = concatenate_datasets(ds_list)
corpus.save_to_disk("datasets/imdb_sst2")
