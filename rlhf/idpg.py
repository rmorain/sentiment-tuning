import pudb
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOConfig, PPOTrainer
from utils import collator

from rlhf import get_response_masks, compute_reward


class IDPGPromptGenerator(nn.Module):
    def __init__(self, d, t, m=256):
        super(IDPGPromptGenerator, self).__init__()
        self.down_project = nn.Linear(d, m)
        self.up_project = nn.ModuleList([nn.Linear(m, d) for _ in range(t)])
        self.value_head = nn.Linear(d, 1) 

    def forward(self, x):
        hidden = F.relu(self.down_project(x))
        W_p = []
        for i in range(len(self.up_project)):
            E_t = self.up_project[i](hidden)
            W_p.append(E_t)
        prefix = torch.stack(W_p, dim=1) 
        value = self.value_head(prefix) 
        return prefix, value


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    reward_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prefix_length = 3
    model = IDPGPromptGenerator(reward_model.config.n_embd, prefix_length).to(device)
    ref_model = IDPGPromptGenerator(reward_model.config.n_embd, prefix_length).to(
        device
    )
    run_config = configparser.ConfigParser()
    run_config.set("DEFAULT", "text_max_new_tokens", "10")
    run_config = run_config["DEFAULT"]
    config = PPOConfig(
        model_name="idpg",
        learning_rate=1.41e-5,
        batch_size=2,
        ratio_threshold=10,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
        kl_penalty="abs",
        mini_batch_size=1,
        init_kl_coef=0,
    )
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )

    dataset = [
        "Hello, world!",
        "I think that",
        "Pizza, pizza, pizza!",
        "I pledge allegiance to the flag of the United States of",
    ]
    ppo_trainer = PPOTrainer(
        config, tokenizer, dataset, data_collator=collator
    )
    input_ids = tokenizer(dataset[3], return_tensors="pt")["input_ids"]
    # Get sequence embedding
    wte = reward_model.transformer.wte
    sequence_embedding = wte(input_ids).mean(1)
    prefix = model(sequence_embedding)  # (batch, prefix_length, d)
    prompt_embeddings = wte(input_ids)  # (batch, prompt_length, d)
    prefix_prompt = torch.cat((prefix, prompt_embeddings), dim=1)
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": 10,
    }
    output = reward_model.generate(inputs_embeds=prompt_embeddings, **gen_kwargs)
    output = [o[1:] for o in output]  # Trim bos token
    response = torch.cat((prefix, output), dim=1)
    response_mask = get_response_masks(prefix, response, device)
    target = [1]
    pu.db
    reward = compute_reward(prefix, prompt_embeddings, target, sentiment_pipeline, reward_model, tokenizer, run_config)
    stats = ppo_trainer.step(prefix_prompt, response, reward, response_mask)


if __name__ == "__main__":
    main()
