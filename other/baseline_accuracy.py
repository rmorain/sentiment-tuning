import pudb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rlhf import IMDBDataset
from rlhf.utils import collator


def main():
    # load dataset
    model_name = "gpt2"
    batch_size = 256
    ds = IMDBDataset(model_name, batch_size)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collator) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "max_new_tokens": 10,
    }
    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": batch_size,
    }
    num_correct = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Generate response
        response = [model.generate(query.unsqueeze(0).to(device), **gen_kwargs).squeeze(0) for query in batch["prompt"]]
        response_str = tokenizer.batch_decode(response)
        scores = sentiment_pipeline(response_str, **pipe_kwargs)
        for score, target in zip(scores, batch["target"]): 
            emotion_scores = F.softmax(
                    torch.tensor([emotion["score"] for emotion in score]), dim=0
                )
            prediction = emotion_scores.argmax()
            if prediction == target:
                num_correct += 1
    accuracy = num_correct / len(dataloader)
    print("Baseline accuracy: ", accuracy)


if __name__ == "__main__":
    main()
