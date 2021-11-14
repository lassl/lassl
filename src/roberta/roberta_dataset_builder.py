from typing import Dict, List, Union

import datasets
import transformers

from ..dataset_builder import DatasetBuilder
from ..utils import list_dict_to_dict_list


class RobertaDatasetBuilder(DatasetBuilder):
    """Roberta DatasetBuilder"""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
        max_length: int,
        key: str,
        reset_position_ids=False,
    ) -> None:
        self.features = datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value("int32")),
                "attention_mask": datasets.Sequence(datasets.Value("uint8")),
                "position_ids": datasets.Sequence(datasets.Value("int32")),
                "special_tokens_mask": datasets.Sequence(datasets.Value("uint8")),
            }
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key = key
        self.reset_position_ids = reset_position_ids
        self.buffer = []

    def make_examples(self, batch: Dict) -> Dict:
        examples = []
        docs = self.tokenizer(
            batch[self.key], padding=False, add_special_tokens=False, truncation=False, return_attention_mask=False
        )["input_ids"]

        for doc in docs:
            doc += [self.tokenizer.sep_token_id]
            ex = self._add_document(doc)
            if ex:
                examples.append(ex)
        return list_dict_to_dict_list(examples)

    def _add_document(self, input_ids: List[int]) -> Union[transformers.BatchEncoding, None]:
        self.buffer += input_ids
        if len(self.buffer) >= self.max_length - 1:
            ex = self._create_example()
            return ex
        return None

    def _create_example(self) -> transformers.BatchEncoding:
        chunk = self.buffer[: self.max_length - 1]
        chunk.insert(0, self.tokenizer.cls_token_id)

        example = self.tokenizer.prepare_for_model(
            chunk,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_special_tokens_mask=False,
        )
        example["position_ids"] = [position_id for position_id in range(len(chunk))]
        example["special_tokens_mask"] = self.tokenizer.get_special_tokens_mask(
            example["input_ids"], already_has_special_tokens=True
        )
        assert len(example["input_ids"]) == len(example["position_ids"]) == len(example["special_tokens_mask"])

        # Prepare to start building the next example
        self.buffer = self.buffer[self.max_length :]
        return example
