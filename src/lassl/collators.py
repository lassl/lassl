import random
from typing import Any, Dict, List, Optional
import torch

from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


class DataCollatorForBert(DataCollatorForWholeWordMask):
    """
    Processing training examples to mini-batch for Bert (mlm+wwm+sop).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_wwm_and_sop_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch_mask = batch.pop("mask_label")
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
        return batch

    def _prepare_wwm_and_sop_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            seq_length = len(chunk_ids)
            start, end = seq_length // 3, seq_length // 3 * 2
            split_position = random.randrange(start, end)
            reverse = random.random() < 0.5

            if reverse:
                token_a = chunk_ids[split_position:]
                token_b = chunk_ids[:split_position]
            else:
                token_a = chunk_ids[:split_position]
                token_b = chunk_ids[split_position:]

            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_a, token_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_a, token_b)
            sentence_order_label = 1 if reverse else 0
            ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            mask_label = self._whole_word_mask(ref_tokens)

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "next_sentence_label": sentence_order_label,
                    "mask_label": mask_label,
                }
            )
        return output_examples


class DataCollatorForAlbert(DataCollatorForLanguageModeling):
    """
    Processing training examples to mini-batch for Albert (mlm+sop).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_sop_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def _prepare_sop_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            seq_length = len(chunk_ids)
            start, end = seq_length // 3, seq_length // 3 * 2
            split_position = random.randrange(start, end)
            reverse = random.random() < 0.5

            if reverse:
                token_a = chunk_ids[split_position:]
                token_b = chunk_ids[:split_position]
            else:
                token_a = chunk_ids[:split_position]
                token_b = chunk_ids[split_position:]

            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_a, token_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_a, token_b)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
            sentence_order_label = 1 if reverse else 0

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "special_tokens_mask": special_tokens_mask,
                    "sentence_order_label": sentence_order_label,
                }
            )
        return output_examples


class DataCollatorForRoberta(DataCollatorForLanguageModeling):
    """
    Processing training examples to mini-batch for Roberta (mlm).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of
        )


class DataCollatorForGpt2:
    """
    Processing training examples to mini-batch for Gpt2 (clm).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        batch = {
            "input_ids": _torch_collate_batch(
                examples, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        }
        batch["labels"] = batch["input_ids"].clone()
        return batch


class DataCollatorForBart:
    """
    Processing training examples to mini-batch for Bart (text-infilling)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        masking_rate : float = 0.3,
        span_length_param : float = 3.

    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.masking_rate = masking_rate
        self.span_length_param = span_length_param

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        # make labels and decoder_input_ids
        batch = {
            "labels": _torch_collate_batch(
                examples, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        }
        batched_bos = torch.full((len(examples), 1), self.tokenizer.bos_token_id)
        batch["decoder_input_ids"] = torch.cat((batched_bos, batch["labels"][:,:-1].clone()), dim=1)
        
        # corrupt input for text-infilling
        mask_token = self.tokenizer.mask_token_id
        corrupt_examples = []
        for e in examples:
            masking_length = int(len(e) * self.masking_rate)
            masked_length = 0
            ex_rest_len = len(e)
            while masking_length > masked_length:
                span_length = torch.min(torch.poisson(torch.tensor([self.span_length_param])), torch.tensor([ex_rest_len-1])).long().item()
                start_index = ((ex_rest_len - span_length)*torch.rand(1)).long().item()
                e = e[:start_index] + [mask_token] + e[start_index + span_length:]
                ex_rest_len -= span_length - 1
                masked_length += span_length
            corrupt_examples.append(e)
        # pad & batchfy input_ids
        batch["input_ids"] = _torch_collate_batch(
                corrupt_examples, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        return batch

class DataCollatorForT5:
    """
    Processing training examples to mini-batch for T5
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        noise_density : float = 0.15,
        mean_span_length : float = 3.0,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length

    def _random_spans_noise_mask(self, length : int) -> torch.BoolTensor:
        ''' pytorch-ported version of https://github.com/google-research/text-to-text-transfer-transformer/blob/bb545f19ec221e6203dd05505573fbc0c0a9001f/t5/data/preprocessors.py#L2901'''
        assert torch.__version__.split(".")[1] == '11', "Currently supporting only version 1.11.x due to scatter_reduce compatibility."
        orig_len = length
        length = max(length, 2) # set minumum to 2 to avoid degeneracy
        num_noise_tokens = round(self.noise_density * length)
        num_noise_tokens = min(max(num_noise_tokens, 1), length-1) # set maximum to length-1 
        num_noise_spans = round(num_noise_tokens / self.mean_span_length)
        num_noise_spans = max(num_noise_spans, 1) # set minumum to 1
        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            # affected by global seed
            bars = torch.arange(num_items-1) < num_segments-1
            bars = bars[torch.randperm(bars.size(0))]
            bars = torch.cat((torch.tensor([0]), bars), dim=0) # to make segment 0 nonzero
            segment_id = torch.cumsum(bars, dim=0)
            segment_length = torch.scatter_reduce(torch.ones_like(segment_id), 0, segment_id, reduce = 'sum')
            return segment_length 
        
        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack((nonnoise_span_lengths, noise_span_lengths), dim=1).reshape(-1)
        span_starts = torch.cumsum(interleaved_span_lengths, dim=0)[:-1]
        span_start_indicator = torch.zeros(length).long().scatter(0, span_starts, torch.ones_like(span_starts))
        span_num = torch.cumsum(span_start_indicator, dim=0)
        is_noise = span_num % 2 == 1
        return is_noise[:orig_len]

    def _noise_span_to_unique_sentinel(self, tokens, noise_mask) -> torch.LongTensor:
        ''' pytorch-ported version of https://github.com/google-research/text-to-text-transfer-transformer/blob/bb545f19ec221e6203dd05505573fbc0c0a9001f/t5/data/preprocessors.py#L3074'''
        tokens = torch.tensor(tokens)
        prev_token_is_noise = torch.cat((torch.tensor([0]), noise_mask[:-1]), dim=0).bool()
        first_noise_tokens = torch.logical_and(
            noise_mask, torch.logical_not(prev_token_is_noise))
        subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

        sentinel = self.tokenizer.get_vocab()["<extra_id_0>"] + 1 - torch.cumsum(first_noise_tokens.long(), dim=0)

        tokens = torch.where(first_noise_tokens, sentinel, tokens)
        return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        example_len = len(examples[0])
        noise_masks = [self._random_spans_noise_mask(example_len) for _ in range(example_n)]
        inputs = [self._noise_span_to_unique_sentinel(example, noise_mask) for example, noise_mask in zip(examples, noise_masks)]
        targets = [self._noise_span_to_unique_sentinel(example, ~noise_mask) for example, noise_mask in zip(examples, noise_masks)]
        # make labels and decoder_input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs, tokenizer=self.tokenizer, pad_to_multiple_of=None # already set to 512 by design
            ),
            "labels": _torch_collate_batch(
                targets, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        }
        return batch