from pathlib import Path
from typing import Generator, Union

import datasets
from datasets import load_dataset

SENT_TEXT_SCRIPT = str((Path(__file__).parent / "loading" / "sent_text.py").resolve().absolute())


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def load_corpora(dir_path, corpus_type="docu_json"):
    corpora_dir = Path(dir_path).absolute()
    extension = corpus_type.split("_")[-1]

    if extension == "json":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.json")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.json")
    elif extension == "text":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.txt")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.txt")
    else:
        raise Exception(f"{extension} is not supported.")

    if corpus_type == "docu_text":
        return load_dataset("text", data_files=list_of_file_paths, split="train")
    elif corpus_type == "docu_json":
        return load_dataset("json", data_files=list_of_file_paths, split="train")
    elif corpus_type == "sent_text":
        return load_dataset(SENT_TEXT_SCRIPT, data_files=list_of_file_paths, split="train")
    elif corpus_type == "sent_json":
        raise NotImplementedError("sent_json will be supported soon.")
    else:
        raise ValueError(f"{corpus_type} must be one of ['docu_text', 'docu_json', 'sent_text', 'sent_json']")


def get_params_without_weight_decay_ln(named_params: Union[list, Generator], weight_decay: float = 0.1):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def compute_indv_chunk_size(target_length, noise_density, mean_span_length):
    '''pre-corruption token length approximation for T5 and UL2'''
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        '''
            https://github.com/google-research/text-to-text-transfer-transformer/blob/c3be7cf1c20e5f6d83e6de99377b653a3a0bc44a/t5/data/preprocessors.py#L2648 
        '''
        # setting mean_span_length to None means prefix-lm that masks last 25% tokens
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = 1 if mean_span_length is None else int(round(num_noise_tokens / mean_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        return (
            num_nonnoise_tokens +
            num_noise_spans + 1,
            num_noise_tokens +
            num_noise_spans + 1)
    
    tokens_length = target_length - 1
    while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
            <= target_length):
        tokens_length += 1
    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length (tokens_length)
    return tokens_length, targets_length