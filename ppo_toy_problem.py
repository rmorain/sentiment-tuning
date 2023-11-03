from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
)
