from transformers import AutoModelForCausalLM


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")


if __name__ == "__main__":
    main()
