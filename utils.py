import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler


# Load data and models
def build_dataset(
    config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb dataset
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def prepare_target(target_dist, emotions, tokenizer):
    target = F.log_softmax(target_dist.sample(), dim=1)
    target_strings = []
    for t in target:
        truncated_score_strs = [f"{flt.item():.2f}" for flt in t]
        target_dict = dict(zip(emotions, truncated_score_strs))
        target_strings.append(str(target_dict))
    target_tokens = tokenizer.batch_encode_plus(target_strings, return_tensors="pt")[
        "input_ids"
    ].to(device)
    return target_tokens


def prepare_target_easy(target_dist, emotions, tokenizer):
    target = target_dist.sample().argmax(1)
    target_strings = []
    for t in target:
        target_string = f"Target: {emotions[t]}"
        target_strings.append(target_string)
    target_tokens = tokenizer.batch_encode_plus(target_strings, return_tensors="pt")[
        "input_ids"
    ]
    return target, target_tokens
