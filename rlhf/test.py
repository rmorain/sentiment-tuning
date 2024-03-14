import sys

import jsonlines
import numpy as np
import pudb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class Senti_Prompt_Data(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        is_training=False,
        args=None,
        emotions=["negative", "positive"],
    ):
        super(Senti_Prompt_Data, self).__init__()
        self.emotions = emotions
        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args

        self.is_training = False

        self.record = []
        self.read_content(json_path)

    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)

        with open(str(json_path), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                prompt = item["prompt"]["text"]

                context = self.tokenizer(prompt.strip(), return_tensors="np")[
                    "input_ids"
                ][0].tolist()

                if len(context) < 1:
                    continue

                concept_set_input_ids = self.tokenizer(
                    f"Sentiment: Positive", return_tensors="np"
                )["input_ids"][0].tolist()
                concept_set_input_ids_ = self.tokenizer(
                    f"Sentiment: Negative", return_tensors="np"
                )["input_ids"][0].tolist()

                self.record.append(
                    {
                        "encode_input": concept_set_input_ids,
                        "encode_input_": concept_set_input_ids_,
                        "context": context,
                        "input_ids": context,
                    }
                )

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item


tokenizer = AutoTokenizer.from_pretrained("gpt2")
neutral_test_ds = Senti_Prompt_Data(
    "datasets/test/neutral_prompts.jsonl",
    tokenizer,
)
save_model_path = "checkpoints/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLMWithValueHead.from_pretrained(save_model_path).to(device)
