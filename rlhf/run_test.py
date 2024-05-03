import configparser
import sys

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from utils import collator

import wandb
from datasets import load_dataset, load_from_disk
from rlhf import test_model


class CombinedTokenizedDataset(Dataset):
    def __init__(self):
        self.ds = load_from_disk("datasets/combined_corpus_tokenized")
        self.emotions = ["negative", "positive"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]


run_config = configparser.ConfigParser()
run_config.read("rlhf/rlhf_config.ini")
try:
    section = sys.argv[1]
except IndexError:
    section = "1"
run_config = run_config[section]
seed = 0
torch.manual_seed(seed)
run = wandb.init(id="hdkkrib8", resume="allow")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "saved_models/checkpoints/prefix_generator_large_step_230"
)
config = PPOConfig(
    model_name="gpt2-large",
    learning_rate=1.41e-5,
    batch_size=256,
    log_with="wandb",
    ratio_threshold=10,
    use_score_scaling=True,
    use_score_norm=True,
    whiten_rewards=True,
    kl_penalty="abs",
    mini_batch_size=128,
    init_kl_coef=0,
    entropy_coef=run_config.getfloat("entropy_coef"),
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(
    device
)
ref_model.pretrained_model.load_state_dict(model.pretrained_model.state_dict())
ref_model.v_head.load_state_dict(model.v_head.state_dict())
reward_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
sentiment_pipeline = pipeline("sentiment-analysis", device=device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = CombinedTokenizedDataset()
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
