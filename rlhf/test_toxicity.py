import configparser
import csv
import sys

import pudb
import torch
from detoxify import Detoxify
from torch.utils.data import Dataset
from toxicity import ToxicityDataset, test_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import collator

import wandb
from datasets import load_dataset, load_from_disk

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
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "saved_models/checkpoints/e8ckmz83"
)
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
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
    device
)
ref_model.pretrained_model.load_state_dict(model.pretrained_model.state_dict())
ref_model.v_head.load_state_dict(model.v_head.state_dict())
reward_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
detoxify = Detoxify("original", device=device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = ToxicityDataset(f"datasets/{run_config.get('dataset')}", tokenizer=tokenizer)
test_ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model,
    tokenizer,
    dataset,
    data_collator=collator,
)
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "output_scores": True,
    "max_new_tokens": run_config.getint("prefix_max_new_tokens"),
}
for _ in range(25):
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
