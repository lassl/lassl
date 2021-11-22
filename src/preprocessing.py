from typing import Dict, List

from transformers import RobertaTokenizerFast


class RobertaProcessor:
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._buffer = []

    def process(self, batch_of_str: List[str]) -> Dict[List[int]]:
        list_of_examples: List[Dict[int]] = []
        training_examples: Dict[List[int]] = {}
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

            if len(self._buffer) >= (self._max_length - 2):
                chunk_ids = self._buffer[: self._max_length - 2]
                example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    padding=False,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                list_of_examples.append(example)
                self._buffer = self._buffer[self._max_length - 2 :]

        for training_example in list_of_examples:
            for key in training_example:
                if key not in training_examples:
                    training_examples.setdefault(key, [])
                training_examples[key].append(training_example[key])

        return training_examples
