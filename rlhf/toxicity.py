import configparser
import csv
import logging
import shutil
import sys
import time
from random import choice

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
import torch.nn.functional as F
from detoxify import Detoxify
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from utils import collator

import wandb
from datasets import load_dataset, load_from_disk


def compute_reward(
    prefixes, prompts, detoxify, reward_model, tokenizer, run_config, gen_kwargs
):
    with torch.no_grad():
        prefix_outputs = [
            reward_model.generate(
                torch.cat((prefix, prompt)).unsqueeze(0), **gen_kwargs
            ).squeeze(0)
            for prefix, prompt in zip(prefixes, prompts)
        ]
        normal_outputs = [
            reward_model.generate(prompt.unsqueeze(0), **gen_kwargs).squeeze(0)
            for prompt in prompts
        ]
        prefix_texts = [
            tokenizer.decode(output[len(prefix) :])[:400]  # Max 400 characters
            for output, prefix in zip(prefix_outputs, prefixes)
        ]
        normal_texts = [
            tokenizer.decode(output)[:400]  # Max 400 characters
            for output, prompt in zip(normal_outputs, prompts)
        ]
        # Debug print statements
        try:
            toxicity = detoxify.predict(prefix_texts)["toxicity"]
            prefix_scores = torch.ones(len(prefixes)) - torch.tensor(toxicity)
        except RuntimeError as e:
            logging.critical(e)
            logging.critical(prefix_texts)

        normal_scores = torch.ones(len(prefixes)) - torch.tensor(
            detoxify.predict(normal_texts)["toxicity"]
        )
        accuracies = []
        rewards = []
        for prefix_score, normal_score, t in zip(
            prefix_scores, normal_scores, toxicity
        ):
            accuracies.append((t < 0.5))
            rewards.append(prefix_score - normal_score)
        mean_accuracy = torch.mean(torch.tensor(accuracies).float())
    return rewards, prefix_texts, mean_accuracy, accuracies, toxicity


def simple_compute_reward(
    prefixes, prompts, detoxify, reward_model, tokenizer, run_config
):
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": run_config.getint("text_max_new_tokens"),
    }
    with torch.no_grad():
        prefix_outputs = [
            reward_model.generate(
                torch.cat((prefix, prompt)).unsqueeze(0), **gen_kwargs
            ).squeeze(0)
            for prefix, prompt in zip(prefixes, prompts)
        ]
        prefix_texts = [
            tokenizer.decode(output[len(prefix) + len(prompt) :])[
                :400
            ]  # Max 400 characters
            for output, prefix, prompt in zip(prefix_outputs, prefixes, prompts)
        ]
        # Debug print statements
        try:
            prefix_scores = torch.ones(len(prefixes)) - torch.tensor(
                detoxify.predict(prefix_texts)["toxicity"]
            )
        except RuntimeError as e:
            logging.critical(e)
            logging.critical(prefix_texts)

        accuracies = []
        for prefix_score in prefix_scores:
            accuracies.append((prefix_score < 0.5))
        mean_accuracy = torch.mean(torch.tensor(accuracies).float())
    return prefix_scores, prefix_texts, mean_accuracy, accuracies


class ToxicityDataset(Dataset):
    def __init__(self, ds_path, tokenizer=None):
        self.ds_path = ds_path
        self.tokenizer = tokenizer
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_dataset("csv", data_files=self.ds_path + ".csv")["train"]
        columns_to_remove = list(ds.features.keys())
        ds = ds.filter(lambda x: x["target"] >= 0.5)  # Only use toxic examples

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(columns_to_remove)

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["comment_text"])[:input_size]
        # No target needed because we only detoxify
        sample["query"] = sample["prompt"]
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def save_to_disk(self, path):
        self.ds.save_to_disk(path)


class RealToxicityDataset(Dataset):
    def __init__(self, ds_path, tokenizer=None):
        self.ds_path = ds_path
        self.tokenizer = tokenizer
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_dataset("allenai/real-toxicity-prompts")["train"]
        columns_to_remove = list(ds.features.keys())
        columns_to_remove.remove("prompt")
        # Get random subset
        ds = ds.filter(
            lambda x: x["prompt"]["toxicity"] and x["prompt"]["toxicity"] < 0.5
        )  # Only use non-toxic examples
        subset = torch.randperm(len(ds))[:10000]  # Only use 10k examples
        ds = ds.select(subset)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(columns_to_remove)

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["prompt"]["text"])[:input_size]
        # No target needed because we only detoxify
        sample["query"] = sample["prompt"]
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def save_to_disk(self, path):
        self.ds.save_to_disk(path)


def clean_stats(stats):
    # Iterate over all values
    for key, values in stats.items():
        stats[key] = np.nan_to_num(values)
    return stats


def get_response_masks(prefix_tensors, response_tensors, device):
    response_masks = []
    for p, r in zip(prefix_tensors, response_tensors):
        prefix_mask = torch.ones(len(p), dtype=torch.long)
        r_mask = torch.zeros(len(r) - len(p), dtype=torch.long)
        response_mask = torch.cat((prefix_mask, r_mask), dim=0)
        response_masks.append(response_mask)
    return response_masks


def save_model(model, model_name):
    print(f"Saving model at: saved_models/{model_name}")
    model.save_pretrained(f"saved_models/{model_name}")


def log_stats(
    rewards,
    tokenizer,
    prefix_tensors,
    prompt,
    texts,
    bssf_table,
    run,
    mean_accuracy,
    toxicity,
):
    best_reward_index = torch.tensor(rewards).argmax()
    best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
    best_prompt = prompt[best_reward_index]
    best_text = texts[best_reward_index]
    best_reward = rewards[best_reward_index]
    prediction = 1 if rewards[best_reward_index] >= 0.5 else 0
    bssf_table.add_data(
        best_prefix,
        best_prompt,
        best_text,
        prediction,
        best_reward,
    )
    run.log({"mean_accuracy": mean_accuracy})
    run.log({"mean_toxicity": np.mean(toxicity)})
    logging.critical(f"Mean accuracy: {mean_accuracy}")


def log_test_stats(
    rewards,
    tokenizer,
    prefix_tensors,
    prompt,
    texts,
    run,
    bssf_table,
):
    best_reward_index = torch.tensor(rewards).argmax()
    best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
    best_prompt = tokenizer.decode(prompt[best_reward_index])
    best_text = texts[best_reward_index]
    best_reward = rewards[best_reward_index]
    prediction = 1 if rewards[best_reward_index] >= 0.5 else 0
    bssf_table.add_data(
        best_prefix,
        best_prompt,
        best_text,
        prediction,
        best_reward,
    )
    with open(f"test_results/{wandb.run.id}.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file, escapechar="\\")
        prefix = tokenizer.batch_decode(prefix_tensors)
        prompt = tokenizer.batch_decode(prompt)
        rows = []
        for i in range(len(prefix)):
            rows.append(
                [
                    prefix[i],
                    prompt[i],
                    texts[i],
                    1 if rewards[i].item() >= 0.5 else 0,  # prediction
                    rewards[i].item(),
                ]
            )
        writer.writerows(rows)


def test_model(
    ppo_trainer,
    detoxify,
    reward_model,
    run_config,
    tokenizer,
    config,
    gen_kwargs,
    run,
):
    bssf_table = wandb.Table(
        columns=[
            "Prefix",
            "Prompt",
            "Texts",
            "Prediction",
            "Reward",
        ]
    )
    reward_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": run_config.getint("text_max_new_tokens"),
    }
    dataset = RealToxicityDataset("datasets/real-toxicity-prompts", tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    total_rewards = 0
    total_accuracy = 0
    total_toxicity = 0
    texts = []
    for _, batch in tqdm(enumerate(dataloader)):
        batch["query"] = [q.cuda() for q in batch["query"]]
        batch["prompt"] = [p.cuda() for p in batch["prompt"]]
        generated_tokens = ppo_trainer.generate(batch["query"], **gen_kwargs)
        prefix_tensors = [
            generated_tokens[i][len(batch["query"][i]) :]
            for i in range(len(generated_tokens))
        ]
        batch["prefix"] = [tokenizer.decode(p.squeeze()) for p in prefix_tensors]
        response_tensors = [
            torch.cat((prefix_tensors[i], batch["prompt"][i]))
            for i in range(len(prefix_tensors))
        ]
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        rewards, batch["texts"], mean_accuracy, accuracies, toxicity = compute_reward(
            prefix_tensors,
            batch["prompt"],
            detoxify,
            reward_model,
            tokenizer,
            run_config,
            reward_kwargs,
        )
        log_test_stats(
            rewards,
            tokenizer,
            prefix_tensors,
            batch["prompt"],
            batch["texts"],
            run,
            bssf_table,
        )
        total_rewards += np.array(rewards).sum(0)
        total_accuracy += np.array(accuracies).sum(0)
        total_toxicity += np.array(toxicity).sum(0)
        texts.extend(batch["texts"])
        if run_config.getboolean("debug"):
            break

    p = 0
    for text in texts:
        p += perplexity(text, tokenizer, reward_model, "cuda")

    diversity = distinctness(texts)

    mean_perplexity = p / len(texts)
    mean_reward = total_rewards / len(dataset)
    mean_accuracy = total_accuracy / len(dataset)
    mean_toxicity = total_toxicity / len(dataset)
    run.log(
        {
            "Test mean reward": mean_reward,
            "Test mean accuracy": mean_accuracy,
            "Test perplexity": mean_perplexity,
            "Test distinctness": f"{diversity[0]:.3f}/{diversity[1]:.3f}/{diversity[2]:.3f}",
            "Test toxicity": mean_toxicity,
        }
    )


def distinctness(generations_data):
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


def perplexity(text, tokenizer, model, device):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()


def main():
    run_config = configparser.ConfigParser()
    run_config.read("rlhf/toxicity_config.ini")
    try:
        section = sys.argv[1]
    except IndexError:
        section = "1"
    run_config = run_config[section]
    seed = 0
    torch.manual_seed(seed)
    run = wandb.init(project="toxicity", config=dict(run_config), resume=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(
        model_name="gpt2",
        learning_rate=run_config.getfloat("learning_rate"),
        batch_size=run_config.getint("batch_size"),
        log_with="wandb",
        ratio_threshold=run_config.getint("ratio_threshold"),
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
        kl_penalty="abs",
        mini_batch_size=run_config.getint("mini_batch_size"),
        init_kl_coef=0,
        entropy_coef=run_config.getfloat("entropy_coef"),
    )

    # Load pretrained models
    save_model_path = f"checkpoints/{wandb.run.id}"
    if run_config.get("resume_id"):
        save_model_path = f"checkpoints/{run_config.get('resume_id')}"
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "saved_models/" + save_model_path
        ).to(device)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
            device
        )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
        device
    )
    ref_model.pretrained_model.load_state_dict(model.pretrained_model.state_dict())
    ref_model.v_head.load_state_dict(model.v_head.state_dict())
    reward_model = AutoModelForCausalLM.from_pretrained(
        run_config.get("reward_model")
    ).to(device)
    detoxify = Detoxify("original", device=device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ToxicityDataset(
        f"datasets/{run_config.get('dataset')}", tokenizer=tokenizer
    )
    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, dataset, data_collator=collator
    )
    # Generation settings
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": run_config.getint("prefix_max_new_tokens"),
    }
    reward_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": run_config.getint("text_max_new_tokens"),
    }
    bssf_table = wandb.Table(
        columns=[
            "Prefix",
            "Prompt",
            "Texts",
            "Prediction",
            "Reward",
        ]
    )

    # Training loop
    epochs = run_config.getint("epochs")
    best_accuracy = 0
    save_every = 10
    for _ in range(epochs):
        epoch_accuracy = []
        for i, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            generated_tokens = ppo_trainer.generate(batch["query"], **gen_kwargs)
            prefix_tensors = [
                generated_tokens[i][len(batch["query"][i]) :]
                for i in range(len(generated_tokens))
            ]
            batch["prefix"] = [tokenizer.decode(p.squeeze()) for p in prefix_tensors]
            response_tensors = [
                torch.cat((prefix_tensors[i], batch["prompt"][i]))
                for i in range(config.batch_size)
            ]
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]
            rewards, batch["texts"], mean_accuracy, _, toxicity = compute_reward(
                prefix_tensors,
                batch["prompt"],
                detoxify,
                reward_model,
                tokenizer,
                run_config,
                reward_kwargs,
            )
            epoch_accuracy.append(mean_accuracy)

            # Run PPO step
            response_masks = get_response_masks(
                prefix_tensors, response_tensors, device
            )
            stats = ppo_trainer.step(
                batch["query"], response_tensors, rewards, response_masks
            )
            batch["prompt_str"] = [
                tokenizer.decode(p.squeeze()) for p in batch["prompt"]
            ]
            ppo_trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["prefix", "prompt_str", "texts"],
            )
            log_stats(
                rewards,
                tokenizer,
                prefix_tensors,
                batch["prompt_str"],
                batch["texts"],
                bssf_table,
                run,
                mean_accuracy,
                toxicity,
            )
            if run_config.getboolean("debug"):
                break
            if run and (i % save_every) == 0:
                save_model(model, save_model_path)

    run.log({"BSSF Table 2": bssf_table})
    # Load best model
    if not run_config.getboolean("debug"):
        test_policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "saved_models/" + save_model_path
        ).to(device)
    else:
        test_policy_model = model
    test_ppo_trainer = PPOTrainer(
        config,
        test_policy_model,
        ref_model,
        tokenizer,
        dataset,
        data_collator=collator,
    )
    test_model(
        test_ppo_trainer,
        detoxify,
        reward_model,
        run_config,
        tokenizer,
        config,
        gen_kwargs,
        run,
    )


if __name__ == "__main__":
    main()
