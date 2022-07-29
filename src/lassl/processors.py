from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


class BaseProcessor(ABC):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._chunk_size = max_length
        self._buffer = []

    def save_tokenizer(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)

    @abstractmethod
    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        pass


class BertProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 3

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.sep_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class RobertaProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_token_type_ids=False,
                    return_attention_mask=False,
                )

                training_example["special_tokens_mask"] = self._tokenizer.get_special_tokens_mask(
                    training_example["input_ids"], already_has_special_tokens=True
                )

                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])

                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class GPT2Processor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])

                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class AlbertProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 3

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class BartProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class T5Processor(BaseProcessor):
    def __init__(
        self, model_name_or_path: str, max_length: int, noise_density: float = 0.15, mean_span_length: float = 3.0
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.max_length = max_length
        self._chunk_size = self._compute_chunk_size()[0]

    def _compute_chunk_size(self) -> Tuple[int, int]:
        """
        compute pre-noise length that is mapped to max_length (= input_size)
        target size is usually shorter than input size (when noise_density < 0.5)
        so we compute pre-noise length that makes input size to be max_length
        """

        def _tokens_length_to_inputs_length_targets_length(tokens_length):
            """
            https://github.com/google-research/text-to-text-transfer-transformer/blob/c3be7cf1c20e5f6d83e6de99377b653a3a0bc44a/t5/data/preprocessors.py#L2648
            """
            num_noise_tokens = int(round(tokens_length * self.noise_density))
            num_nonnoise_tokens = tokens_length - num_noise_tokens
            num_noise_spans = int(round(num_noise_tokens / self.mean_span_length))
            # inputs contain all nonnoise tokens, sentinels for all noise spans
            # and one EOS token.
            return (num_nonnoise_tokens + num_noise_spans + 1, num_noise_tokens + num_noise_spans + 1)

        tokens_length = self.max_length - 1
        while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= self.max_length:
            tokens_length += 1
        inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
        return tokens_length, targets_length

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            # no document seperation
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                # slice upto chunk_size - 1 since chunk_size contains eos token already
                # and add eos token at the end of sequence
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class ElectraProcessor(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.sep_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples
