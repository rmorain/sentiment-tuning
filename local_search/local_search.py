import configparser
import time
from random import randint

import pudb
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import wandb
from rlhf.rlhf import IMDBDataset, compute_reward
from rlhf.utils import collator


def local_search(
    batch,
    tokenizer,
    prefix_length,
    sentiment_pipeline,
    reward_model,
    run_config,
    device,
    prefix=None,
):
    prefixes = []
    prompts = []
    targets = []
    if not prefixes:
        # Generate random prefix
        prefixes = [
            [randint(0, tokenizer.vocab_size) for _ in range(prefix_length)]
            for _ in range(len(batch["target"]))
        ]
        prefixes = torch.tensor(prefixes, device=device).unsqueeze(0)
    prompts = [p.to(device) for p in batch["prompt"]]
    targets = [t.to(device) for t in batch["target"]]
    start = time.time()
    pu.db
    reward = compute_reward(
        prefixes,
        prompts,
        targets,
        sentiment_pipeline,
        reward_model,
        tokenizer,
        run_config,
    )

    end = time.time()
    duration = end - start
    print(duration)
    exit()


def main():
    model_name = "gpt2"
    batch_size = 1
    prefix_length = 5
    run_config = configparser.ConfigParser()
    run_config.set("DEFAULT", "text_max_new_tokens", "10")
    run_config = run_config["DEFAULT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = IMDBDataset(model_name, batch_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    # Perform local search on each element of the dataset
    for batch in dataloader:
        prefix, reward = local_search(
            batch,
            tokenizer,
            prefix_length,
            sentiment_pipeline,
            reward_model,
            run_config,
            device,
        )


if __name__ == "__main__":
    main()
