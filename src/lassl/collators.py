import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.data.data_collator import _torch_collate_batch
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import (
    compute_indv_chunk_size,
    noise_span_to_unique_sentinel,
    random_spans_noise_mask,
)


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def _token_type_ids_with_pad(
    examples: Any, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8
) -> torch.Tensor:
    """
    Create "token_type_ids" for Bert-like models
    used when token_a & token_b already in the same chunk separated by [SEP] token
    (for those not in the same chunk, use "tokenizer.create_token_type_ids_from_sequences(token_a, token_b)")
    """
    if isinstance(examples, torch.Tensor):
        examples = examples.tolist()
    if max([len(example) for example in examples]) % pad_to_multiple_of == 0:
        max_seq_len = max([len(example) for example in examples])
    else:
        max_seq_len = (
            pad_to_multiple_of
            + (max([len(example) for example in examples]) // pad_to_multiple_of) * pad_to_multiple_of
        )
    token_type_ids_with_padding = []
    for example in examples:
        for idx in range(len(example)):
            if example[idx] == tokenizer.sep_token_id and idx != len(example) - 1:
                token_type_ids_with_padding.append([0] * (idx + 1) + [1] * (max_seq_len - idx - 1))
                break
            if idx == len(example) - 1:
                token_type_ids_with_padding.append([0 for _ in range(max_seq_len)])
    return torch.tensor(token_type_ids_with_padding).long()


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

    def __call__(self, examples) -> List[Dict[str, Any]]:
        batch = super().__call__([example["input_ids"] for example in examples])
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["input_ids"])
        )
        if "position_ids" in examples[0].keys():
            batch["position_ids"] = _torch_collate_batch(
                [example["position_ids"] for example in examples],
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch["position_ids"] = (
                torch.arange(0, batch["input_ids"].size(-1)).expand(batch["input_ids"].size(0), -1).clone()
            )
        return batch


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
        permute_ratio=0.0,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.poisson_dist = torch.distributions.Poisson(poisson_lambda)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.permute_ratio = permute_ratio
        self.mask_token_index = self.tokenizer.mask_token_id
        self.eos_index = tokenizer.eos_token_id
        self.full_stop_index = tokenizer.vocab["."]

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        batch = {"labels": _torch_collate_batch(examples, tokenizer=self.tokenizer, pad_to_multiple_of=None)}
        batch["labels"] = torch.where(batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.bos_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["decoder_input_ids"])
        )
        examples = self._chunk_permutation(examples, self.permute_ratio, self.full_stop_index, self.eos_index)
        examples = self._infilling(examples, self.mask_token_index)
        batch["input_ids"] = _torch_collate_batch(
            examples, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch["attention_mask"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, torch.ones_like(batch["input_ids"])
        )
        return batch

    def _infilling(self, examples, mask_token_index):
        buffer = []
        for example in examples:
            source_tokens_ids = example
            source_tokens_ids_length = len(example)
            masking_length = int(source_tokens_ids_length * self.mlm_probability)
            masked_length = 0

            while masked_length < masking_length:
                span_length = int(min(self.poisson_dist.sample().item(), source_tokens_ids_length - 1))
                start_index = torch.randint(0, source_tokens_ids_length - span_length + 1, (1,)).item()
                source_tokens_ids = (
                    source_tokens_ids[:start_index]
                    + [mask_token_index]
                    + source_tokens_ids[start_index + span_length :]
                )
                source_tokens_ids_length -= span_length - 1
                masked_length += span_length
            buffer.append(source_tokens_ids)

        return buffer

    # Ref: https://github.com/facebookresearch/fairseq/blob/aa79bb9c37b27e3f84e7a4e182175d3b50a79041/fairseq/data/denoising_dataset.py
    def _chunk_permutation(self, examples, p, full_stop_index, eos_index):
        if p == 0.0:
            return examples

        buffer = []
        examples = [torch.tensor(example) for example in examples]
        for example in examples:
            full_stops = (example == full_stop_index) + (example == eos_index)
            full_stops[-1] = 1
            sentence_ends = (full_stops * ~torch.cat([full_stops[1:], torch.tensor([False])])).nonzero() + 1
            num_sentences = sentence_ends.size(0)
            num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
            substitutions = torch.randperm(num_sentences)[:num_to_permute]
            ordering = torch.arange(0, num_sentences)
            ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]
            result = example.clone()
            index = 0
            for i in ordering:
                sentence = example[(sentence_ends[i - 1] if i > 0 else 0) : sentence_ends[i]]
                result[index : index + sentence.size(0)] = sentence
                index += sentence.size(0)
            buffer.append(result.tolist())

        return buffer


class DataCollatorForElectra(DataCollatorForWholeWordMask):
    """
    Processing training examples to mini-batch for Electra (fake input discrimination).
    Modified implementation: discriminator only version
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        examples = [
            example["input_ids"].tolist() if isinstance(example["input_ids"], torch.Tensor) else example["input_ids"]
            for example in examples
        ]
        batch = self._generate_fake_inputs(examples)
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["token_type_ids"] = _token_type_ids_with_pad(
            batch["input_ids"], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        return batch

    def _generate_fake_inputs(self, examples: List[List[int]]) -> Dict[str, Any]:
        input_ids = [self.tokenizer.prepare_for_model(example, padding=False)["input_ids"] for example in examples]
        original_input_ids = _torch_collate_batch(
            input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        torch_masked_boolean: torch.Tensor = super().__call__(input_ids)["input_ids"] == self.tokenizer.mask_token_id
        whole_random_ids = torch.randint(0, self.tokenizer.vocab_size - 1, torch_masked_boolean.size())
        fake_generated_ids = torch.where(torch_masked_boolean, whole_random_ids, original_input_ids)
        labels = (original_input_ids != fake_generated_ids).long()
        return {"input_ids": fake_generated_ids, "labels": labels, "original_input_ids": original_input_ids}


class DataCollatorForT5:
    """
    Processing training examples to mini-batch for T5
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int = 8,
        noise_density: float = 0.15,
        mean_span_length: float = 3.0,
        first_extra_id: str = "<extra_id_0>",
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self._decoder_start_token_id = (
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        )
        self.first_sentinel_index: int = self.tokenizer.get_vocab()[first_extra_id]
        self.descending_sentinel: bool = self.is_sentinel_index_descending(first_extra_id)

    # check whether sentinel tokens indices are in descending order or not
    def is_sentinel_index_descending(self, first_extra_id: str) -> bool:
        """
        The logic for "noise_span_to_unique_sentinel()" totally depends on the order of sentinel token ids indices in vocabs.
        So, it is necessary to check whether the vocab indices of <extra_id_0> to <extra_id_99> are increasing or decreasing.
        For example, if "<extra_id_0>" is mapped to 10 in vocab and "<extra_id_99>" is mapped to 109 in vocab,
        the output of is_sentinel_index_descending() should be "False" and vice versa.
        """
        if "0" in first_extra_id:
            last_extra_id = "99".join(first_extra_id.split("0"))
        else:
            raise ValueError("first sentinel tokens must include zero")
        return self.tokenizer.get_vocab()[last_extra_id] - self.tokenizer.get_vocab()[first_extra_id] < 0

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        example_len = len(examples[0])
        noise_masks = [
            random_spans_noise_mask(self.noise_density, self.mean_span_length, example_len) for _ in range(example_n)
        ]
        inputs = [
            noise_span_to_unique_sentinel(
                self.tokenizer,
                example,
                noise_mask,
                first_sentinel_index=self.first_sentinel_index,
                is_sentinel_index_descending=self.descending_sentinel,
            )
            for example, noise_mask in zip(examples, noise_masks)
        ]
        targets = [
            noise_span_to_unique_sentinel(
                self.tokenizer,
                example,
                ~noise_mask,
                first_sentinel_index=self.first_sentinel_index,
                append_last_sentinel=True,
                is_sentinel_index_descending=self.descending_sentinel,
            )
            for example, noise_mask in zip(examples, noise_masks)
        ]
        # make labels and input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=None,  # all samples' length are set to self.max_length by design
            ),
            "labels": _torch_collate_batch(
                targets, tokenizer=self.tokenizer, pad_to_multiple_of=None  # labels' length are all sample by design
            ),
        }
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self._decoder_start_token_id
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
        noise_densities: List = [0.15, 0.15, 0.5, 0.5, 0.15, 0.5, 0.25],
        mean_span_lengths: List = [3.0, 8.0, 3.0, 8.0, 64.0, 64.0, None],
        optional_task_prefixes: List[str] = ["[NLU]", "[NLU]", "[NLG]", "[NLG]", "[NLG]", "[NLG]", "[S2S]"],
        first_extra_id: str = "[new_id_27]",
    ):
        mean_span_lengths = [None if isinstance(e, str) else e for e in mean_span_lengths]
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.noise_densities = noise_densities
        self.mean_span_lengths = mean_span_lengths
        self.optional_task_prefixes = optional_task_prefixes
        self.first_sentinel_index: int = self.tokenizer.get_vocab()[first_extra_id]
        self.descending_sentinel: bool = self.is_sentinel_index_descending(first_extra_id)

    # check whether sentinel tokens indices are in descending order or not
    def is_sentinel_index_descending(self, first_extra_id: str) -> bool:
        """
        The logic for "noise_span_to_unique_sentinel()" totally depends on the order of sentinel token ids indices in vocabs.
        So, it is necessary to check whether the vocab indices of <new_id_27> to <new_id_0> are increasing or decreasing.
        For example, if "<new_id_27>" is mapped to 100 in vocab and "<new_id_0>" is mapped to 73 in vocab,
        the output of is_sentinel_index_descending() should be "True" and vice versa.
        """
        if "27" in first_extra_id:
            last_extra_id = "0".join(first_extra_id.split("27"))
        else:
            raise ValueError("first sentinel tokens must include 27")
        return self.tokenizer.get_vocab()[last_extra_id] - self.tokenizer.get_vocab()[first_extra_id] < 0

    def _noise_mask_with_index(self, index: int) -> torch.BoolTensor:
        index = self.get_index(index)  # get shuffled index
        _noise_density = self.noise_densities[index]
        _mean_span_length = self.mean_span_lengths[index]
        inp_size, _, _mean_span_length = compute_indv_chunk_size(510, _noise_density, _mean_span_length)
        return random_spans_noise_mask(_noise_density, _mean_span_length, inp_size)

    def _unique_sentinel_input_with_index(
        self, example: List[int], noise_mask: torch.BoolTensor, index: int
    ) -> torch.LongTensor:
        denoiser_prefix = self.optional_task_prefixes[self.get_index(index)]
        return noise_span_to_unique_sentinel(
            self.tokenizer,
            example,
            noise_mask,
            first_sentinel_index=self.first_sentinel_index,
            denoiser_prefix=denoiser_prefix,
            is_sentinel_index_descending=self.descending_sentinel,
        )

    def _unique_sentinel_target(self, example: List[int], noise_mask: torch.BoolTensor) -> torch.LongTensor:
        return noise_span_to_unique_sentinel(
            self.tokenizer,
            example,
            ~noise_mask,
            first_sentinel_index=self.first_sentinel_index,
            is_sentinel_index_descending=self.descending_sentinel,
        )

    def __call__(self, examples):
        denoiser_order = np.random.permutation(len(self.noise_densities)).tolist()
        self.get_index = lambda idx: denoiser_order[idx % len(self.noise_densities)]
        # denoiser_prefix_order = [self.optional_task_prefixes[i] for i in denoiser_order]

        examples = [example["input_ids"] for example in examples]
        example_n = len(examples)
        noise_masks = [self._noise_mask_with_index(idx) for idx in range(example_n)]
        inputs = [
            self._unique_sentinel_input_with_index(example, noise_mask, index)
            for index, (example, noise_mask) in enumerate(zip(examples, noise_masks))
        ]
        targets = [
            self._unique_sentinel_target(example, noise_mask) for example, noise_mask in zip(examples, noise_masks)
        ]
        # make labels and input_ids
        batch = {
            "input_ids": _torch_collate_batch(
                inputs,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,  # all samples' length are set to self.max_length by design
            ),
            "labels": _torch_collate_batch(
                targets,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,  # labels' length are all sample by design
            ),
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
