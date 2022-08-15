import random
from typing import Any, Dict, List, Optional
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.data.data_collator import _torch_collate_batch
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import noise_span_to_unique_sentinel, random_spans_noise_mask, compute_indv_chunk_size

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


# Ref: https://github.com/cosmoquester/transformers-bart-pretrain/blob/master/transformers_bart_pretrain/data.py
class DataCollatorForBart:
    """
    Processing training examples to mini-batch for Bart (text-infilling)
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability=0.15,
        poisson_lambda=3,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.poisson_dist = torch.distributions.Poisson(poisson_lambda)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        batch = {"labels": _torch_collate_batch(examples, tokenizer=self.tokenizer, pad_to_multiple_of=None)}
        batch["labels"] = torch.where(batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["decoder_input_ids"])
        )
        batch["input_ids"] = self._infilling(examples)
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["input_ids"])
        )
        return batch

    def _infilling(self, examples):
        buffer = []
        for example in examples:
            source_tokens_ids = example
            source_tokens_ids_length = example.size(0)
            masking_length = int(source_tokens_ids_length * self.mlm_probability)
            masked_length = 0

            while masked_length < masking_length:
                span_length = int(min(self.poisson_dist.sample().item(), source_tokens_ids_length - 1))
                start_index = torch.randint(0, source_tokens_ids_length - span_length, (1,)).item()
                source_tokens_ids = torch.concat(
                    [
                        source_tokens_ids[:start_index],
                        torch.tensor([self.tokenizer.mask_token_id]),
                        source_tokens_ids[start_index + span_length :],
                    ]
                )
                source_tokens_ids_length -= span_length - 1
                masked_length += span_length
            buffer.append(source_tokens_ids)
        return pad_sequence(buffer, batch_first=True, padding_value=self.tokenizer.pad_token_id)


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


    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        example_len = len(examples[0])
        noise_masks = [random_spans_noise_mask(self.noise_density, self.mean_span_length, example_len) for _ in range(example_n)]
        inputs = [noise_span_to_unique_sentinel(self.tokenizer, example, noise_mask) for example, noise_mask in zip(examples, noise_masks)]
        targets = [noise_span_to_unique_sentinel(self.tokenizer,example, ~noise_mask, append_last_sentinel=True) for example, noise_mask in zip(examples, noise_masks)]
        # make labels and input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs, tokenizer=self.tokenizer, pad_to_multiple_of=None # all samples' length are set to self.max_length by design
            ),
            "labels": _torch_collate_batch(
                targets, tokenizer=self.tokenizer, pad_to_multiple_of=None # labels' length are all sample by design
            )
        }
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["decoder_input_ids"])
        )
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["input_ids"])
        )
        return batch

class DataCollatorForUL2:
    """
    Processing training examples to mini-batch for UL2
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        noise_densities : List = [0.15,0.15,0.5,0.5,0.15,0.5,0.25], 
        mean_span_lengths : List = [3.0,8.0,3.0,8.0,64.0,64.0,None],
        optional_task_prefixes : List[str] = ["<r_denoiser_token>","<r_denoiser_token>","<x_denoiser_token>","<x_denoiser_token>","<x_denoiser_token>","<x_denoiser_token>","<s_denoiser_token>"]
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.noise_densities = noise_densities 
        self.mean_span_lengths = mean_span_lengths
        self.optional_task_prefixes = optional_task_prefixes

    def _noise_mask_with_index(self, index:int):
        index = self.get_index(index) # get shuffled index
        _noise_density = self.noise_densities[index]
        _mean_span_length = self.mean_span_lengths[index]
        inp_size = compute_indv_chunk_size(512, _noise_density, _mean_span_length)[0]
        return random_spans_noise_mask(_noise_density, _mean_span_length, inp_size)


    def __call__(self, examples):
        denoiser_order = np.random.permutation(len(self.noise_densities))
        self.get_index = lambda idx : denoiser_order[idx % len(self.noise_densities)]
        denoiser_prefix_order = [self.optional_task_prefixes[i] for i in denoiser_order] 

        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        noise_masks = [self._noise_mask_with_index(idx) for idx in range(example_n)]
        inputs = [noise_span_to_unique_sentinel(self.tokenizer, example, noise_mask, denoiser_prefix_order=denoiser_prefix_order, first_extra_id = "[new_id_27]") for example, noise_mask in zip(examples, noise_masks)]
        targets = [noise_span_to_unique_sentinel(self.tokenizer, example, ~noise_mask, append_last_sentinel=True, first_extra_id = "[new_id_27]") for example, noise_mask in zip(examples, noise_masks)]
        # make labels and input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs, tokenizer=self.tokenizer, pad_to_multiple_of=None # all samples' length are set to self.max_length by design
            ),
            "labels": _torch_collate_batch(
                targets, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of # labels' length are all sample by design
            )
        }
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["decoder_input_ids"])
        )
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["input_ids"])
        )
        return batch
