from typing import Dict, List

from transformers import RobertaTokenizerFast
from transformers import GPT2TokenizerFast


class BaseProcessor:
    def __init__(self) -> None:
        self._tokenizer = None
        self._max_length = None
        self._chunk_size = None
        self._buffer = []

    def process(self, batch_of_str: List[str]) -> Dict[str, List[int]]:
        list_of_training_examples: List[Dict[str, int]] = []
        dict_of_training_examples: Dict[str, List[int]] = {}
        batch_of_input_ids: List[List[int]] = self._tokenizer(
            batch_of_str,
            padding=False,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]

        for input_ids in batch_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            if len(self._buffer) >= (self._chunk_size):
                chunk_ids = self._buffer[: self._chunk_size]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    padding=False,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )

                training_example[
                    "special_tokens_mask"
                ] = self._tokenizer.get_special_tokens_mask(
                    training_example["input_ids"], already_has_special_tokens=True
                )

                training_example["position_ids"] = [
                    i for i in range(len(training_example["input_ids"]))
                ]

                assert (
                    len(training_example["input_ids"])
                    == len(training_example["position_ids"])
                    == len(training_example["special_tokens_mask"])
                )
                list_of_training_examples.append(training_example)
                self._buffer = self._buffer[self._chunk_size :]

        for training_example in list_of_training_examples:
            for key in training_example:
                if key not in dict_of_training_examples:
                    dict_of_training_examples.setdefault(key, [])
                dict_of_training_examples[key].append(training_example[key])

        return dict_of_training_examples


class RobertaProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__()
        self._tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._chunk_size = max_length - 2


class GPT2Processor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__()
        self._tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._chunk_size = max_length
