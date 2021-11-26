import random
from typing import Any, Dict, List, Optional
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import _torch_collate_batch


class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of


    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        examples = self.ready_for_sop(examples)

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def ready_for_sop(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            _input_ids = example["input_ids"]
            seq_length = len(_input_ids)
            start, end = seq_length // 3, seq_length // 3 * 2
            split_position = random.randrange(start, end)
            reverse = random.random() < 0.5
            if reverse:
                token_a = _input_ids[split_position:]
                token_b = _input_ids[:split_position]
            else:
                token_a = _input_ids[:split_position]
                token_b = _input_ids[split_position:]
            
            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_a, token_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_a, token_b)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
            sentence_order_label = 1 if reverse else 0

            output_examples.append({
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "special_tokens_mask": special_tokens_mask,
                "sentence_order_label": sentence_order_label,
            })
        return output_examples
