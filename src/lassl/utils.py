import random
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import datasets
import torch
from datasets import load_dataset

SENT_TEXT_SCRIPT = str((Path(__file__).parent / "loading" / "sent_text.py").resolve().absolute())


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def load_corpora(dirpath, corpus_type="docu_json", **kwargs):
    """kwargs can contain arguments such as `cache_dir`"""
    corpora_dir = Path(dirpath).absolute()
    extension = corpus_type.split("_")[-1]

    if extension == "json":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.json")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.json")
    elif extension == "text":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.txt")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.txt")
    elif extension == "parquet":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.parquet")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.parquet")
    else:
        raise Exception(f"{extension} is not supported.")

    if corpus_type == "docu_text":
        return load_dataset("text", data_files=list_of_file_paths, split="train", **kwargs)
    elif corpus_type == "docu_json":
        return load_dataset("json", data_files=list_of_file_paths, split="train", **kwargs)
    elif corpus_type == "sent_text":
        return load_dataset(SENT_TEXT_SCRIPT, data_files=list_of_file_paths, split="train", **kwargs)
    elif corpus_type == "sent_json":
        raise NotImplementedError("sent_json will be supported soon.")
    elif corpus_type == "parquet":
        return load_dataset("parquet", data_files=list_of_file_paths, split="train", **kwargs)
    else:
        raise ValueError(
            f"{corpus_type} must be one of ['docu_text', 'docu_json', 'sent_text', 'sent_json', 'parquet']"
        )


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


def compute_indv_chunk_size(
    target_length: int, noise_density: float, mean_span_length: Union[float, int]
) -> Tuple[int]:
    """pre-corruption token length approximation for T5 and UL2"""

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        """
        https://github.com/google-research/text-to-text-transfer-transformer/blob/c3be7cf1c20e5f6d83e6de99377b653a3a0bc44a/t5/data/preprocessors.py#L2648
        """
        # setting mean_span_length to None means prefix-lm that masks last 25% tokens
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = 1 if mean_span_length is None else int(round(num_noise_tokens / mean_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        return (num_nonnoise_tokens + num_noise_spans + 1, num_noise_tokens + num_noise_spans + 1)

    tokens_length = target_length - 1
    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= target_length:
        tokens_length += 1
    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    if mean_span_length is None:
        mean_span_length = targets_length - 2
    return tokens_length, targets_length, mean_span_length


def random_spans_noise_mask(noise_density: float, mean_span_length: float, length: int) -> torch.BoolTensor:
    """pytorch-ported version of https://github.com/google-research/text-to-text-transfer-transformer/blob/bb545f19ec221e6203dd05505573fbc0c0a9001f/t5/data/preprocessors.py#L2901"""
    orig_len = length
    length = max(length, 2)  # set minumum to 2 to avoid degeneracy
    num_noise_tokens = round(noise_density * length)
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)  # set maximum to length-1
    num_noise_spans = round(num_noise_tokens / mean_span_length)
    num_noise_spans = max(num_noise_spans, 1)  # set minumum to 1
    num_nonnoise_tokens = length - num_noise_tokens

    def _random_segmentation(num_items, num_segments):
        # affected by global seed
        bars = torch.arange(num_items - 1) < num_segments - 1
        bars = bars[torch.randperm(bars.size(0))]
        bars = torch.cat((torch.tensor([0]), bars), dim=0)  # to make segment 0 nonzero
        segment_id = torch.cumsum(bars, dim=0)
        segment_length = torch.zeros(num_segments, dtype=torch.long).scatter_add(
            0, segment_id, torch.ones_like(segment_id)
        )
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = torch.stack((nonnoise_span_lengths, noise_span_lengths), dim=1).reshape(-1)
    span_starts = torch.cumsum(interleaved_span_lengths, dim=0)[:-1]
    span_start_indicator = torch.zeros(length).long().scatter(0, span_starts, torch.ones_like(span_starts))
    span_num = torch.cumsum(span_start_indicator, dim=0)
    is_noise = span_num % 2 == 1
    return is_noise[:orig_len]


def noise_span_to_unique_sentinel(
    tokenizer,
    tokens: Union[List[int], torch.Tensor],
    noise_mask: torch.BoolTensor,
    first_sentinel_index: int,
    append_last_sentinel: bool = False,
    denoiser_prefix: Optional[str] = None,
    is_sentinel_index_descending: bool = True,
) -> torch.LongTensor:
    """pytorch-ported version of https://github.com/google-research/text-to-text-transfer-transformer/blob/bb545f19ec221e6203dd05505573fbc0c0a9001f/t5/data/preprocessors.py#L3074"""
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)

    # sample consecutive substring from tokens if len(tokens) > len(noise_mask)
    # in case of T5, these two should match. In case of UL2, due to use of several denoisers, number of tokens could be larger than length of noise masks.
    if len(tokens) > len(noise_mask):
        offset = len(tokens) - len(noise_mask)
        random.seed(tokens[0].item())  # seed that makes same example to match in both making inputs and targets
        start_idx = random.randint(0, offset)
        tokens = tokens[start_idx : start_idx + len(noise_mask)]
        assert len(tokens) == len(noise_mask)

    prev_token_is_noise = torch.cat((torch.tensor([0]), noise_mask[:-1]), dim=0).bool()
    first_noise_tokens = torch.logical_and(noise_mask, torch.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    # apply sentinel tokens to noise tokens indices based on the order of the sentinel tokens
    if is_sentinel_index_descending:
        sentinel = first_sentinel_index + 1 - torch.cumsum(first_noise_tokens.long(), dim=0)
    else:
        sentinel = first_sentinel_index - 1 + torch.cumsum(first_noise_tokens.long(), dim=0)
    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    ret = torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))

    # target masking needs additional sentinel token at last position
    if append_last_sentinel and is_sentinel_index_descending:
        last_sentinel_id = sentinel.min().reshape(-1) - 1
        ret = torch.cat((ret, last_sentinel_id), dim=0)
    elif append_last_sentinel and not is_sentinel_index_descending:
        last_sentinel_id = sentinel.max().reshape(-1) + 1
        ret = torch.cat((ret, last_sentinel_id), dim=0)
    ret = torch.cat((ret, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)), dim=0)  # add eos token

    if denoiser_prefix:
        # used only for UL2, which prepends one of [S2S], [NLG], [NLU] during training.
        # These tokens are not treated as special tokens but they are tokenized as normal tokens.
        denoiser_prefix_enc = torch.tensor(tokenizer.encode(denoiser_prefix)[:1], dtype=torch.long)
        ret = torch.cat((denoiser_prefix_enc, ret), dim=0)
    return ret
