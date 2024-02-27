import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Tokenizer
from trl.core import LengthSampler

from datasets import load_dataset


# Load data and models
def build_dataset(
    config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb dataset
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def prepare_target(target_dist, emotions, tokenizer):
    target = F.log_softmax(target_dist.sample(), dim=1)
    target_strings = []
    for t in target:
        truncated_score_strs = [f"{flt.item():.2f}" for flt in t]
        target_dict = dict(zip(emotions, truncated_score_strs))
        target_strings.append(str(target_dict))
    target_tokens = tokenizer.batch_encode_plus(target_strings, return_tensors="pt")[
        "input_ids"
    ].to(device)
    return target_tokens


def prepare_target_easy(target_dist, emotions, tokenizer):
    target = target_dist.sample().argmax(1)
    target_strings = []
    for t in target:
        target_string = f"Target: {emotions[t]}"
        target_strings.append(target_string)
    target_tokens = tokenizer.batch_encode_plus(target_strings, return_tensors="pt")[
        "input_ids"
    ]
    return target, target_tokens


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == " " and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
        _isupper = True
    return _isupper


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop("trigger_mask")
        predict_mask = model_inputs.pop("predict_mask")
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits, *_ = self._model(**model_inputs)
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(
            logits.size(0), -1
        )
        return predict_logits


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs["input_ids"]
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out["input_ids"] = filled
    return out


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """

    def __init__(
        self,
        template,
        config,
        tokenizer,
        label_field="label",
        label_map=None,
        tokenize_labels=False,
        add_special_tokens=False,
        use_ctx=False,
    ):
        if not hasattr(tokenizer, "predict_token") or not hasattr(
            tokenizer, "trigger_token"
        ):
            raise ValueError(
                "Tokenizer missing special trigger and predict tokens in vocab."
                "Use `utils.add_special_tokens` to add them."
            )
        self._template = template
        self._config = config
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._tokenize_labels = tokenize_labels
        self._add_special_tokens = add_special_tokens
        self._use_ctx = use_ctx

    @property
    def num_trigger_tokens(self):
        return sum(token == "[T]" for token in self._template.split())

    def __call__(self, format_kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop(self._label_field)
        text = self._template.format(**format_kwargs)
        if label is None:
            raise Exception(f"Bad data: {text}")

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer.encode_plus(
            text, add_special_tokens=self._add_special_tokens, return_tensors="pt"
        )
        input_ids = model_inputs["input_ids"]
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs["trigger_mask"] = trigger_mask
        model_inputs["predict_mask"] = predict_mask

        # For relation extraction with BERT, update token_type_ids to reflect the two different sequences
        if self._use_ctx and self._config.model_type == "bert":
            sep_token_indices = (
                (
                    input_ids.squeeze(0)
                    == self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)
                )
                .nonzero()
                .flatten()
            )
            sequence_b_indices = (
                torch.arange(sep_token_indices[0], sep_token_indices[1] + 1)
                .long()
                .unsqueeze(0)
            )
            model_inputs["token_type_ids"].scatter_(1, sequence_b_indices, 1)

        # Encode the label(s)
        if self._label_map is not None:
            label = self._label_map[label]
        label_id = encode_label(
            tokenizer=self._tokenizer, label=label, tokenize=self._tokenize_labels
        )

        return model_inputs, label_id


def encode_label(tokenizer, label, tokenize=False):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token
            # if it gets split into multiple tokens. TODO: Make sure this is
            # desired behavior.
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded
