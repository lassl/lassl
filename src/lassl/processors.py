from abc import ABC, abstractmethod
from typing import Dict, List

from transformers import AutoTokenizer

from lassl.utils import compute_indv_chunk_size


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
        self._chunk_size = compute_indv_chunk_size(self.max_length, self.noise_density, self.mean_span_length)[0]

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            add_special_tokens=False,
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


class UL2Processor(BaseProcessor):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        noise_densities: List = [0.15, 0.15, 0.5, 0.5, 0.15, 0.5, 0.25],
        mean_span_lengths: List = [3.0, 8.0, 3.0, 8.0, 64.0, 64.0, None],
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self.noise_densities = noise_densities
        self.mean_span_lengths = mean_span_lengths
        self.max_length = max_length - 2  # eos, denoiser_specific_token
        self._chunk_size = self._compute_chunk_size()

    def _compute_chunk_size(self) -> int:
        """
        compute maximum chunk size for all denoisers
        """
        inp_sizes = []
        for noise_density, mean_span_length in zip(self.noise_densities, self.mean_span_lengths):
            required_inp_size = compute_indv_chunk_size(self.max_length, noise_density, mean_span_length)[0]
            inp_sizes.append(required_inp_size)

        return max(inp_sizes)

    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            # no document seperation
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
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
