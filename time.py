import time

import pudb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
controller = AutoModelForCausalLMWithValueHead.from_pretrained(
    "saved_models/checkpoints/k1nb1tli"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
prefix_gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "output_scores": True,
    "max_new_tokens": 15,
}
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
    "output_scores": True,
    "max_new_tokens": 20,
}
control_instruction = "Sentiment: positive."
prompt = "I really can't stand"
control_prompt = control_instruction + prompt
input_ids = tokenizer(control_prompt, return_tensors="pt")["input_ids"].to(device)
prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


start = time.time()
response = controller.generate(input_ids, **prefix_gen_kwargs)
prefix = response[:, len(input_ids[0]) :]
prefix_prompt = torch.cat((prefix, prompt_ids), dim=1)
continuation = model.generate(prefix_prompt, **gen_kwargs)
end = time.time()

print(f"Duration: {end - start}")
