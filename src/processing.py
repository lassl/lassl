from typing import Dict, List
from transformers import AutoTokenizer
import warnings


class RobertaProcessor:
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._buffer = []

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list_of_input_ids: List[List[int]] = self._tokenizer(
                list_of_str,
                padding=False,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_special_tokens_mask=False,
            )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            if len(self._buffer) >= (self._max_length - 2):
                chunk_ids = self._buffer[: self._max_length - 2]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    padding=False,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                special_tokens_mask_of_training_example: List[int] = \
                    self._tokenizer.get_special_tokens_mask(
                    training_example["input_ids"], already_has_special_tokens=True,
                )
                training_example["special_tokens_mask"] = special_tokens_mask_of_training_example
                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])
                self._buffer = self._buffer[self._max_length - 2 :]

        for training_example in list_of_training_examples:
            for key in training_example:
                if key not in dict_of_training_examples:
                    dict_of_training_examples.setdefault(key, [])
                dict_of_training_examples[key].append(training_example[key])

        return dict_of_training_examples


class AlbertProcessor:
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._buffer = []

    def process(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}
        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            padding=False,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]
        
        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            if len(self._buffer) >= (self._max_length - 3):
                chunk_ids = self._buffer[: self._max_length - 3]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._max_length - 3 :]

        return dict_of_training_examples
