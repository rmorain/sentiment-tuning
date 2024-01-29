import configparser
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.distributions import Multinomial
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextClassificationPipeline,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

import wandb
from utils import build_dataset, collator, prepare_target, prepare_target_easy

tqdm.pandas()


def main():
    run_config = configparser.ConfigParser()
    run_config.read("config.ini")
    section = sys.argv[1]
    run_config = run_config[section]

    wandb.init(
        project="General sentiment tuning", name=section, config=dict(run_config)
    )

    # Configuration
    config = PPOConfig(
        model_name=run_config["model_name"],
        learning_rate=run_config.getfloat("lr"),
        log_with="wandb",
    )

    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": run_config.getint("batch_size"),
    }

    dataset = build_dataset(config)

    # Load pretrained models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
    )

    # Load classifier
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available else "cpu"

    # sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb",
    # device=device)
    reward_pipe = pipeline(
        "text-generation", model=config.model_name, tokenizer=tokenizer, device=device
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )

    # Test classifier
    text = "this movie was really bad!!"
    sentiment_preds = sentiment_pipe(text, **sent_kwargs)
    print(sentiment_preds)

    text = "this movie was really good!!"
    sentiment_preds = sentiment_pipe(text, **sent_kwargs)
    print(sentiment_preds)

    # Generation settings
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Optimize model

    # Training loop
    output_min_length = run_config.getint("output_min_length")
    output_max_length = run_config.getint("output_max_length")
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # Number of emotions
    num_emotions = run_config.getint("num_emotions")
    emotions = run_config["emotions"].split(",")
    space_token = tokenizer(" ", return_tensors="pt")["input_ids"].to(device).squeeze(0)
    # Tokenize metaprompt
    metaprompt_tokens = (
        tokenizer(run_config["metaprompt"], return_tensors="pt")["input_ids"]
        .to(device)
        .squeeze(0)
    )
    # Create target distribution
    target_dist = Multinomial(
        run_config.getint("top_k_emotions"),
        probs=torch.ones((config.batch_size, num_emotions)),
    )
    max_steps = 100
    for i in range(max_steps):
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            prompt_tensors = batch["input_ids"]
            # A single target for simplicity
            target = "This review is positive."
            target_tokens = tokenizer(target)["input_ids"].squeeze(0)
            query_tensors = [
                torch.cat((target_tokens, prompt_tensors[i]), dim=0)
                for i in range(len(prompt_tensors))
            ]
            batch["query"] = [tokenizer.decode(q) for q in query_tensors]
            generated_tokens = ppo_trainer.generate(query_tensors, **gen_kwargs)

            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            # Compute reward
            # Prepend `response`, which is the `prefix`, to the `query`, which is the text
            # from the dataset
            # texts = [r + q for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(generated_texts, **sent_kwargs)
            predicted_sentiments = []
            for output in pipe_outputs:
                # Convert to tensor
                preds = torch.tensor([p["score"] for p in output])
                # pred_logprobs = F.log_softmax(preds, dim=0)
                predicted_sentiments.append(preds)
            # Compute reward
            rewards = -F.cross_entropy(
                torch.stack(predicted_sentiments), target, reduction="none"
            )
            rewards = [r for r in rewards]

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    # Save model
    model.save_pretrained("saved_models/" + run_config["save_name"])
    tokenizer.save_pretrained("saved_models/" + run_config["save_name"])


if __name__ == "__main__":
    main()
