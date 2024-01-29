import time
from random import choice

import numpy as np
import pudb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

import wandb
from datasets import load_dataset, load_from_disk
from utils import collator


def compute_reward(
    responses, targets, sentiment_pipeline, reward_model, tokenizer, gen_kwargs
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
    with torch.no_grad():
        outputs = [
            reward_model.generate(response.unsqueeze(0), **gen_kwargs).squeeze(0)
            for response in responses
        ]
        texts = [tokenizer.decode(output) for output in outputs]
        scores = sentiment_pipeline(texts, **pipe_kwargs)
        rewards = []
        predictions = []
        accuracies = []
        for score, target in zip(scores, targets):
            emotion_scores = [emotion["score"] for emotion in score]
            prediction = np.argmax(emotion_scores)
            rewards.append(torch.tensor(score[target]["score"]))
            predictions.append(prediction)
            accuracies.append((prediction == target))
        mean_accuracy = torch.mean(torch.tensor(accuracies).float())
    return rewards, predictions, mean_accuracy


class IMDBDataset(Dataset):
    def __init__(self, model_name, batch_size):
        self.emotions = ["negative", "positive"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.ds = self._build_dataset()

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        try:
            ds = load_from_disk("datasets/imdb")
        except FileNotFoundError:
            ds = load_dataset("imdb", split="train")
            ds.save_to_disk("datasets/imdb")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["review", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        sample["target"] = 1
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["review"])[:input_size]
        sample["query"] = self.tokenizer.encode(
            f"The review is {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


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


def save_model(model, tokenizer, model_name):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model.save_pretrained(f"saved_models/{model_name}_{time_str}.pt")
    tokenizer.save_pretrained(f"saved_models/{model_name}_tokenizer_{time_str}.pt")


def log_stats(
    rewards,
    tokenizer,
    prefix_tensors,
    predictions,
    targets,
    bssf_table,
    run,
    mean_accuracy,
):
    best_reward_index = torch.tensor(rewards).argmax()
    best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
    best_reward = rewards[best_reward_index]
    prediction = tokenizer.decode(predictions[best_reward_index])
    prediction_token = predictions[best_reward_index]
    target_token = targets[best_reward_index]
    target = tokenizer.decode(target_token)
    bssf_table.add_data(
        best_prefix,
        best_reward,
        prediction,
        prediction_token,
        target,
        target_token,
    )
    run.log({"mean_accuracy": mean_accuracy})


def main():
    torch.manual_seed(0)
    run = wandb.init(project="imdb-sentiment-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(
        model_name="gpt2",
        learning_rate=1.41e-5,
        batch_size=256,
        log_with="wandb",
        ratio_threshold=10,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
        kl_penalty="abs",
        mini_batch_size=64,
    )

    # Load pretrained models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
        device
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
        device
    )
    ref_model.pretrained_model.load_state_dict(model.pretrained_model.state_dict())
    ref_model.v_head.load_state_dict(model.v_head.state_dict())
    reward_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    sentiment_pipeline = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = IMDBDataset(config.model_name, config.batch_size)
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
        "max_new_tokens": 5,
    }
    bssf_table = wandb.Table(
        columns=[
            "Prefix",
            "Reward",
            "Prediction",
            "Prediction Token",
            "Target",
            "Target Token",
        ]
    )

    # Training loop
    max_steps = 1
    for i in range(max_steps):
        for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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
            rewards, predictions, mean_accuracy = compute_reward(
                response_tensors,
                batch["target"],
                sentiment_pipeline,
                reward_model,
                tokenizer,
                gen_kwargs,
            )

            # Run PPO step
            response_masks = get_response_masks(
                prefix_tensors, response_tensors, device
            )
            stats = ppo_trainer.step(
                batch["query"], response_tensors, rewards, response_masks
            )
            ppo_trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "prefix"],
            )
            log_stats(
                rewards,
                tokenizer,
                prefix_tensors,
                predictions,
                batch["target"],
                bssf_table,
                run,
                mean_accuracy,
            )

    # Save model
    model_name = "test"
    # save_model(model, tokenizer, model_name)


if __name__ == "__main__":
    main()
