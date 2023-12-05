from torch.utils.data import Dataset
from random import choice

from transformers import AutoTokenizer


class ColorDataset(Dataset):
    def __init__(self, model_name, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, *args):
        target = choice(["red", "blue"])
        prompt = " What is the color of the book?"
        target_ids = self.tokenizer("The book is " + target + ".", return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        return {"target": target_ids, "prompt": prompt_ids}
