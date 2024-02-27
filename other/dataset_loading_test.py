import os
from datasets import load_dataset
import pudb


ds = load_dataset("imdb", split=["train"])[0]
ds.save_to_disk("datasets/imdb")