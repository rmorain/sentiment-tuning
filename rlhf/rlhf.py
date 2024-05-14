import configparser
import csv
import logging
import shutil
import sys
from random import choice

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from utils import collator

import wandb
from datasets import load_dataset, load_from_disk


def compute_reward(
    prefixes, prompts, targets, sentiment_pipeline, reward_model, tokenizer, run_config
):
    """
    Computes a reward value for each response based on the likelihood of of the target.

    Args:
        response (`torch.LongTensor`):
            A tensor of shape (`batch_size`, `response_length`) containing response ids
        target :
    """
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }
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
            for output in normal_outputs
        ]
        # Debug print statements
        try:
            prefix_scores = sentiment_pipeline(prefix_texts, **pipe_kwargs)
        except RuntimeError as e:
            logging.critical(e)
            logging.critical(prefix_texts)

        normal_scores = sentiment_pipeline(normal_texts, **pipe_kwargs)
        rewards = []
        predictions = []
        accuracies = []
        for prefix_score, normal_score, target in zip(
            prefix_scores, normal_scores, targets
        ):
            prefix_emotion_scores = F.softmax(
                torch.tensor([emotion["score"] for emotion in prefix_score]), dim=0
            )
            normal_emotion_scores = F.softmax(
                torch.tensor([emotion["score"] for emotion in normal_score]), dim=0
            )
            emotion_scores = prefix_emotion_scores - normal_emotion_scores
            prediction = np.argmax(prefix_emotion_scores)
            rewards.append(emotion_scores[target])
            predictions.append(prediction)
            accuracies.append((prediction == target))
        mean_accuracy = torch.mean(torch.tensor(accuracies).float())
    return rewards, predictions, prefix_texts, mean_accuracy, accuracies


def simple_compute_reward(
    prefixes,
    prompts,
    targets,
    sentiment_pipeline,
    reward_model,
    tokenizer,
    run_config,
    test,
):
    """
    Computes a reward value for each response based on the likelihood of of the target.

    Args:
        response (`torch.LongTensor`):
            A tensor of shape (`batch_size`, `response_length`) containing response ids
        target :
    """
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }
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
        if not test:
            prefix_texts = [
                tokenizer.decode(output[len(prefix) + len(prompt) :])[
                    :400
                ]  # Max 400 characters
                for output, prefix, prompt in zip(prefix_outputs, prefixes, prompts)
            ]
        else:
            prefix_texts = [
                tokenizer.decode(output[len(prefix) :])[:400]  # Max 400 characters
                for output, prefix, prompt in zip(prefix_outputs, prefixes, prompts)
            ]
        # Debug print statements
        try:
            prefix_scores = sentiment_pipeline(prefix_texts, **pipe_kwargs)
        except RuntimeError as e:
            logging.critical(e)
            logging.critical(prefix_texts)

        rewards = []
        predictions = []
        accuracies = []
        for prefix_score, target in zip(prefix_scores, targets):
            prefix_emotion_scores = F.softmax(
                torch.tensor([emotion["score"] for emotion in prefix_score]), dim=0
            )
            emotion_scores = prefix_emotion_scores
            prediction = np.argmax(prefix_emotion_scores)
            rewards.append(emotion_scores[target])
            predictions.append(prediction)
            accuracies.append((prediction == target))
        mean_accuracy = torch.mean(torch.tensor(accuracies).float())
    return rewards, predictions, prefix_texts, mean_accuracy, accuracies


class IMDBDataset(Dataset):
    def __init__(
        self,
        model_name,
        batch_size,
        split="train",
        emotions=["negative", "positive"],
        target=None,
    ):
        self.emotions = emotions
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.split = split
        self.ds = self._build_dataset()

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        try:
            ds = load_from_disk("/home/rmorain2/sentiment_tuning/datasets/imdb")
        except FileNotFoundError:
            ds = load_dataset("imdb", split=self.split)
            ds.save_to_disk("datasets/imdb")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["review", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        if sample["label"] == 0:
            sample["target"] = 1
        else:
            sample["target"] = 0
        sample["target_label"] = self.emotions[sample["target"]]
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["review"])[:input_size]
        sample["query"] = self.tokenizer.encode(
            f"Sentiment: {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


class Senti_Prompt_Data(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        args=None,
        emotions=["negative", "positive"],
        target=None,
    ):
        super(Senti_Prompt_Data, self).__init__()
        self.emotions = emotions
        self.tokenizer = tokenizer
        np.set_printoptions(threshold=sys.maxsize)
        self.args = args
        self.target = target

        self.record = []
        self.read_content(json_path)

    def read_content(self, json_path):
        print("reading data from %s ..." % json_path)

        with open(str(json_path), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if self.target is not None:
                    target = self.target
                else:
                    target = np.random.randint(len(self.emotions))
                prompt = item["prompt"]["text"]

                context = self.tokenizer(prompt.strip(), return_tensors="np")[
                    "input_ids"
                ][0].tolist()

                if len(context) < 1:
                    continue

                target_label = self.tokenizer(
                    f"Sentiment: {self.emotions[target]}. ", return_tensors="np"
                )["input_ids"][0].tolist()
                self.record.append(
                    {
                        "query": torch.tensor(target_label + context, dtype=torch.long),
                        "prompt": torch.tensor(context, dtype=torch.long),
                        "target": target,
                    }
                )

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        item = self.record[index]
        return item


class SST2Dataset(Dataset):
    def __init__(
        self,
        emotions=["negative", "positive"],
        target=None,
        sentiment=None,
        tokenizer=None,
    ):
        self.emotions = emotions
        self.target = target
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.ds = self._build_dataset()

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        try:
            ds = load_from_disk("/home/rmorain2/sentiment_tuning/datasets/sst-2")
        except FileNotFoundError:
            ds = load_dataset("stanfordnlp/sst2", split="train")
            ds.save_to_disk("datasets/sst-2")
        if self.sentiment is not None:
            ds = ds.filter(lambda x: x["label"] == self.sentiment, batched=False)

        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["sentence", "label", "idx"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        if sample["label"] == 0:
            sample["target"] = 1
        else:
            sample["target"] = 0
        sample["target_label"] = self.emotions[sample["target"]]
        sample["prompt"] = self.tokenizer.encode(sample["sentence"])
        sample["query"] = self.tokenizer.encode(
            f"Sentiment: {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


class CombinedDataset(Dataset):
    def __init__(
        self,
        ds_path,
        emotions=["negative", "positive"],
        tokenizer=None,
    ):
        self.ds_path = ds_path
        self.emotions = emotions
        self.tokenizer = tokenizer
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_from_disk(self.ds_path)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["text", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        if sample["label"] == 0:
            sample["target"] = 1
        else:
            sample["target"] = 0
        sample["target_label"] = self.emotions[sample["target"]]
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["text"])[:input_size]
        sample["query"] = self.tokenizer.encode(
            f"Sentiment: {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def save_to_disk(self, path):
        self.ds.save_to_disk(path)


class CombinedTokenizedDataset(Dataset):
    def __init__(self, ds_path):
        self.ds = load_from_disk(ds_path)
        self.emotions = ["negative", "positive"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


class ToxcityDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.toknizer = tokenizer
        self.ds_path = f"datasets/jigsaw_toxicity/{split}.csv"
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_dataset("csv", data_files=self.ds_path)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["text", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        # Todo
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["text"])[:input_size]
        # No target needed because we only detoxify
        sample["query"] = sample["prompt"]
        return sample


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
    predictions,
    targets,
    bssf_table,
    run,
    mean_accuracy,
    emotions,
):
    best_reward_index = torch.tensor(rewards).argmax()
    best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
    best_prompt = prompt[best_reward_index]
    best_text = texts[best_reward_index]
    best_reward = rewards[best_reward_index]
    prediction = emotions[predictions[best_reward_index]]
    target = targets[best_reward_index]
    bssf_table.add_data(
        best_prefix,
        best_prompt,
        best_text,
        prediction,
        target,
        best_reward,
    )
    run.log({"mean_accuracy": mean_accuracy})
    logging.critical(f"Mean accuracy: {mean_accuracy}")


def log_test_stats(
    rewards,
    tokenizer,
    prefix_tensors,
    prompt,
    texts,
    predictions,
    targets,
    bssf_table,
    emotions,
    dataset,
    run,
):
    best_reward_index = torch.tensor(rewards).argmax()
    best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
    best_prompt = tokenizer.decode(prompt[best_reward_index])
    best_text = texts[best_reward_index]
    best_reward = rewards[best_reward_index]
    prediction = emotions[predictions[best_reward_index]]
    target = targets[best_reward_index]
    bssf_table.add_data(
        dataset,
        best_prefix,
        best_prompt,
        best_text,
        prediction,
        target,
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
                    dataset,
                    prefix[i],
                    prompt[i],
                    texts[i],
                    predictions[i].item(),
                    targets[i],
                    rewards[i].item(),
                ]
            )
        writer.writerows(rows)


def test_model(
    ppo_trainer,
    sentiment_pipeline,
    reward_model,
    run_config,
    tokenizer,
    config,
    gen_kwargs,
    run,
):
    test_table = wandb.Table(
        columns=[
            "Dataset",
            "Prefix",
            "Prompt",
            "Texts",
            "Prediction",
            "Target",
            "Reward",
        ]
    )
    test_total_table = wandb.Table(
        columns=[
            "Dataset",
            "Target",
            "Reward",
            "Accuracy",
            "Perplexity",
            "Distinctness",
        ]
    )
    datasets = [
        "positive_prompts",
        "neutral_prompts",
        "negative_prompts",
    ]
    emotions = ["negative", "positive"]
    for dataset_name in datasets:
        for target in range(2):  # Only positive or negative for now
            reward, accuracy, perplexity, dist = test_dataset(
                dataset_name,
                target,
                ppo_trainer,
                gen_kwargs,
                tokenizer,
                sentiment_pipeline,
                reward_model,
                run_config,
                test_table,
                emotions,
                config,
                run,
            )
            test_total_table.add_data(
                dataset_name,
                target,
                reward,
                accuracy,
                perplexity,
                f"{dist[0]:.3f}/{dist[1]:.3f}/{dist[2]:.3f}",
            )
    run.log({"Test table": test_table})
    run.log({"Test total table": test_total_table})


def test_dataset(
    dataset_name,
    target,
    ppo_trainer,
    gen_kwargs,
    tokenizer,
    sentiment_pipeline,
    reward_model,
    run_config,
    test_table,
    emotions,
    config,
    run,
):
    dataset = Senti_Prompt_Data(
        f"datasets/test/{dataset_name}.jsonl", tokenizer, target=target
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    total_rewards = 0
    total_accuracy = 0
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
        rewards, predictions, batch["texts"], _, accuracies = simple_compute_reward(
            prefix_tensors,
            batch["prompt"],
            batch["target"],
            sentiment_pipeline,
            reward_model,
            tokenizer,
            run_config,
            True,  # Testing uses prompt and continuation
        )
        log_test_stats(
            rewards,
            tokenizer,
            prefix_tensors,
            batch["prompt"],
            batch["texts"],
            predictions,
            batch["target"],
            test_table,
            emotions,
            dataset_name,
            run,
        )
        total_rewards += np.array(rewards).sum(0)
        total_accuracy += np.array(accuracies).sum(0)
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
    return mean_reward, mean_accuracy, mean_perplexity, diversity


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
    run_config.read("rlhf/rlhf_config.ini")
    try:
        section = sys.argv[1]
    except IndexError:
        section = "1"
    run_config = run_config[section]
    seed = 0
    torch.manual_seed(seed)
    run = wandb.init(
        project="imdb-sentiment-tuning", config=dict(run_config), resume=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(
        model_name="gpt2",
        # learning_rate=1.41e-5,
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
    sentiment_pipeline = pipeline("sentiment-analysis", device=device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = CombinedDataset(
        "/home/rmorain2/sentiment_tuning/datasets/imdb_sst2", tokenizer=tokenizer
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
    bssf_table = wandb.Table(
        columns=[
            "Prefix",
            "Prompt",
            "Texts",
            "Prediction",
            "Target",
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
            rewards, predictions, batch["texts"], mean_accuracy, _ = (
                simple_compute_reward(
                    prefix_tensors,
                    batch["prompt"],
                    batch["target"],
                    sentiment_pipeline,
                    reward_model,
                    tokenizer,
                    run_config,
                    False,  # Use only continuation to compute reward
                )
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
                columns_to_log=["target_label", "prefix", "prompt_str", "texts"],
            )
            log_stats(
                rewards,
                tokenizer,
                prefix_tensors,
                batch["prompt_str"],
                batch["texts"],
                predictions,
                batch["target_label"],
                bssf_table,
                run,
                mean_accuracy,
                dataset.emotions,
            )
            if run_config.getboolean("debug"):
                break
            if run and (i % save_every) == 0 and mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                save_model(model, save_model_path)

    run.log({"BSSF Table 2": bssf_table})
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
        sentiment_pipeline,
        reward_model,
        run_config,
        tokenizer,
        config,
        gen_kwargs,
        run,
    )


if __name__ == "__main__":
    main()
