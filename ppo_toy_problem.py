from random import choice

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from utils import collator

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 512,
}


# Load pretrained models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token


class ColorDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self):
        target = choice([" Red", " Blue"])
        prompt = "What color is this book?"
        target_ids = tokenizer(target, return_tensors="pt")["input_ids"]
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        return prompt_ids, target_ids


# Initialize PPOTrainer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, data_collator=collator)
# Generation settings
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# Training loop
output_min_length = 2
output_max_length = 8
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    # Get response from policy
    response_tensors = []
    # Create emotion target
    # target = F.log_softmax(torch.normal(mean=torch.zeros((config.batch_size,
    # num_emotions)), std=1.0), dim=1)
    target, target_tokens = prepare_target_easy(target_dist, emotions, tokenizer)
    target_tokens = target_tokens.to(device)
    generated_texts = []
    for i, query in enumerate(query_tensors):
        gen_len = output_length_sampler()
        gen_kwargs["max_new_tokens"] = gen_len
        query_with_emotion = torch.cat(
            (metaprompt_tokens, space_token, target_tokens[i], space_token, query),
            dim=0,
        ).long()
        response = ppo_trainer.generate(query_with_emotion, **gen_kwargs)
        response = response.squeeze()[-gen_len:]
        response_tensors.append(response)
        # Generate text for reward
        response_query = tokenizer.decode(torch.cat((response, query), dim=0))
        generated_text = reward_pipe(
            response_query, **gen_kwargs, return_full_text=False
        )[0]["generated_text"]
        generated_texts.append(generated_text)
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute reward
    ## Prepend `response`, which is the `prefix`, to the `query`, which is the text
    ## from the dataset
    # texts = [r + q for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(generated_texts, **sent_kwargs)
    predicted_sentiments = []
    for output in pipe_outputs:
        # Convert to tensor
        preds = torch.tensor([p["score"] for p in output])
        # pred_logprobs = F.log_softmax(preds, dim=0)
        predicted_sentiments.append(preds)
    # Compute reward
    # rewards = -F.kl_div(torch.stack(predicted_sentiments), target,
    # reduction="none", log_target=True).sum(1)
    rewards = -F.cross_entropy(
        torch.stack(predicted_sentiments), target, reduction="none"
    )
    rewards = [r for r in rewards]
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# Save model
model.save_pretrained("saved_models/" + run_config["save_name"])
tokenizer.save_pretrained("saved_models/" + run_config["save_name"])
