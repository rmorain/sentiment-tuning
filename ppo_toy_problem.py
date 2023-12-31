from random import choice

import numpy as np
import pudb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import wandb
from utils import collator


def compute_reward(responses, target, reward_model, device):
    """
    Computes a reward value for each response based on the likelihood of of the target.

    Args:
        response (`torch.LongTensor`):
            A tensor of shape (`batch_size`, `response_length`) containing response ids
        target :
    """
    rewards = []
    predictions = []
    with torch.no_grad():
        for i, response in enumerate(responses):
            labels = torch.full(response.shape, -100)
            labels[-len(target[i]) :] = target[i]
            output = reward_model(response, labels=labels.to(device))
            rewards.append(-output.loss)
            predictions.append(output.logits[-1].argmax())
    return rewards, predictions


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


def clean_stats(stats):
    # Iterate over all values
    for key, values in stats.items():
        stats[key] = np.nan_to_num(values)
    return stats


def main():
    torch.manual_seed(0)
    run = wandb.init(project="debugging-ppo", name="debug")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PPOConfig(
        model_name="gpt2",
        learning_rate=1.41e-5,
        batch_size=128,
        log_with="wandb",
        ratio_threshold=100,
        use_score_scaling=True,
        whiten_rewards=True,
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
    reward_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ColorDataset(config.model_name, config.batch_size)
    # Initialize PPOTrainer
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
    }

    # Training loop
    done = False
    best_reward_so_far = -float("inf")
    prev_reward_mean = -float("inf")
    debounce = 0
    debounce_limit = 20
    debug = False
    bssf_table = wandb.Table(columns=["Prefix", "Reward", "Prediction"])
    max_steps = 1000
    step = 0
    while not done:
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            target_tensors = batch["target"]
            prompt_tensors = batch["prompt"]
            query_tensors = [
                torch.cat((target_tensors[i], prompt_tensors[i]))
                for i in range(len(prompt_tensors))
            ]
            batch["query"] = [tokenizer.decode(q.squeeze()) for q in query_tensors]

            try:
                generated_tokens, scores = ppo_trainer.generate(
                    query_tensors, **gen_kwargs
                )
            except RuntimeError:
                pu.db
            prefix_tensors = [
                generated_tokens[i][len(query_tensors[i]) :]
                for i in range(len(generated_tokens))
            ]
            batch["prefix"] = [tokenizer.decode(p.squeeze()) for p in prefix_tensors]
            response_tensors = [
                torch.cat((prefix_tensors[i], prompt_tensors[i]))
                for i in range(config.batch_size)
            ]
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]
            rewards, predictions = compute_reward(
                response_tensors, batch["target"], reward_model, device
            )

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            best_reward_index = torch.tensor(rewards).argmax()
            mean_reward = torch.tensor(rewards).mean()
            best_prefix = tokenizer.decode(prefix_tensors[best_reward_index])
            best_reward = rewards[best_reward_index]
            prediction = tokenizer.decode(predictions[best_reward_index])
            bssf_table.add_data(best_prefix, best_reward, prediction)
            # Filter stats for infinity
            stats = clean_stats(stats)
            ppo_trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "prefix"],
            )

            # Check if done
            if debug:
                done = True
                continue
            if max_steps is not None and step < max_steps:
                step += 1
                done = False
                continue
            if best_reward > best_reward_so_far:
                best_reward_so_far = best_reward
                debounce = 0
            elif mean_reward > prev_reward_mean:
                prev_reward_mean = mean_reward
                debounce = 0
            elif debounce > debounce_limit:
                done = True
            else:
                debounce += 1

    # Save model
    run.log({"BSSF Table": bssf_table})
    model.save_pretrained("saved_models/ppo_color_model.pt")
    tokenizer.save_pretrained("saved_models/ppo_color_model_tokenizer.pt")


if __name__ == "__main__":
    main()
