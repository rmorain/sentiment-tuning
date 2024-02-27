from datasets import ColorDataset
from torch.utils.data import DataLoader
from core import (
    generate_prefix,
    generate_response,
    score_response,
    update_dataset,
    train_prompt_generator,
)


def main():
    batch_size = 64
    color_dataset = ColorDataset("gpt2", batch_size)
    color_dataloader = DataLoader(color_dataset, batch_size=batch_size, shuffle=True)
    # Generate initial prefixes
    data = next(iter(color_dataloader))
    target = data["target"]
    prompt = data["prompt"]

    prefixes = generate_prefix(target, prompt)


if __name__ == "__main__":
    main()
