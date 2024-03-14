import configparser
import copy
import time
from random import randint

import numpy as np
import pudb
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import collator

import wandb
from rlhf import IMDBDataset, compute_reward


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
    neighborhood_size = 100
    nieghborhood_distance = 1
    if not prefixes:
        # Generate random prefix
        prefixes = [
            [randint(0, tokenizer.vocab_size) for _ in range(prefix_length)]
            for _ in range(len(batch["target"]))
        ]
        prefixes = torch.tensor(prefixes, device=device)
    prompts = [p.to(device) for p in batch["prompt"]]
    targets = [t.to(device) for t in batch["target"]]
    reward, prediction, prefix_text, _ = compute_reward(
        prefixes,
        prompts,
        targets,
        sentiment_pipeline,
        reward_model,
        tokenizer,
        run_config,
    )
    done = False
    prefix = prefixes[0]
    reward = reward[0]
    prediction = prediction[0]
    num_solutions = 1
    while not done:
        done = True
        neighbors = generate_random_neighbors(
            prefix, nieghborhood_distance, neighborhood_size, tokenizer.vocab_size
        )
        neighbor_rewards, predictions, prefix_texts, _ = compute_reward(
            neighbors,
            prompts * neighborhood_size,
            targets * neighborhood_size,
            sentiment_pipeline,
            reward_model,
            tokenizer,
            run_config,
        )
        best_neighbor = np.argmax(neighbor_rewards)
        if neighbor_rewards[best_neighbor] > reward:
            done = False
            reward = neighbor_rewards[best_neighbor]
            prefix = neighbors[best_neighbor]
            prediction = predictions[best_neighbor]
            prefix_text = prefix_texts[best_neighbor]
        num_solutions += neighborhood_size
    return prefix, reward, prediction, prefix_text, num_solutions


def generate_random_neighbors(prefix, distance, neighborhood_size, vocab_size):
    # Generate random neighbors
    neighbors = []
    for _ in range(neighborhood_size):
        neighbor = copy.deepcopy(prefix)
        # Perform |distance| swaps
        swap_idx = np.random.randint(0, len(prefix), size=distance)
        # Swap current token for a random token
        neighbor[swap_idx] = torch.randint(0, vocab_size, size=(distance,))
        neighbors.append(neighbor)
    return neighbors


def main():
    model_name = "gpt2"
    batch_size = 1
    prefix_length = 5
    emotions = ["Negative", "Positive"]
    run_config = configparser.ConfigParser()
    run_config.set("DEFAULT", "text_max_new_tokens", "10")
    run_config = run_config["DEFAULT"]
    run = wandb.init(
        project="local-search-imdb-sentiment-tuning",
        config={
            "model_name": model_name,
            "prefix_length": prefix_length,
            "max_new_tokens": run_config.getint("text_max_new_tokens"),
        },
    )
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
    rewards = []
    accuracy = 0
    i = 0
    prefix_table = wandb.Table(
        columns=[
            "Prefix",
            "Prompt",
            "Texts",
            "Prediction",
            "Target",
            "Reward",
            "# solutions explored",
        ]
    )
    num_steps = 256
    for batch in dataloader:
        prefix, reward, prediction, text, num_solutions = local_search(
            batch,
            tokenizer,
            prefix_length,
            sentiment_pipeline,
            reward_model,
            run_config,
            device,
        )
        rewards.append(reward)
        if prediction == batch["target"][0]:
            accuracy += 1
        # Log results
        prefix_table.add_data(
            tokenizer.decode(prefix),
            tokenizer.decode(batch["prompt"][0]),
            text[0],
            emotions[prediction],
            batch["target_label"][0],
            reward,
            num_solutions,
        )
        i += 1
        if i == num_steps:
            break
    run.log({"Prefix Table": prefix_table})
    mean_reward = np.mean(rewards)
    accuracy = accuracy / num_steps
    run.log({"Mean reward": mean_reward, "Accuracy": accuracy})


if __name__ == "__main__":
    main()
