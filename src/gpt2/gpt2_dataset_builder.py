from typing import Dict, List, Union

import datasets
import transformers

from ..dataset_builder import DatasetBuilder
from ..utils import list_dict_to_dict_list


class GPT2DatasetBuilder(DatasetBuilder):
    """GPT2 DatasetBuilder"""

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
            doc += [
                self.tokenizer.eos_token_id
            ]  # In GPT-2, concatenate eos_token (e.g. <|endoftext|>) to real documents
            ex = self._add_document(doc)
            if ex:
                examples.append(ex)
        return list_dict_to_dict_list(examples)

    def _add_document(self, input_ids: List[int]) -> Union[transformers.BatchEncoding, None]:
        self.buffer += input_ids
        if len(self.buffer) >= self.max_length:
            ex = self._create_example()
            return ex
        return None

    def _create_example(self) -> transformers.BatchEncoding:
        chunk = self.buffer[: self.max_length]

        example = self.tokenizer.prepare_for_model(
            chunk, padding=False, truncation=False, return_attention_mask=True, return_token_type_ids=False
        )
        example["position_ids"] = [position_id for position_id in range(len(chunk))]

        # TODO: 추 후에 reset position ids를 구현하기위해 필요한 code snippet
        # special_tokens_mask_of_example = self.tokenizer.get_special_tokens_mask(
        #     example["input_ids"], already_has_special_tokens=True
        # )
        # To be deleted when issue(https://github.com/huggingface/transformers/issues/9933) is fixed

        # if self.reset_position_ids:
        #     position_id = 0
        #     for i in special_tokens_mask_of_example:
        #         if i != 1:  # 1 denotes eos_token (e.g. <|endoftext|>)
        #             example["position_ids"].append(position_id)
        #             position_id += 1
        #         else:
        #             example["position_ids"].append(position_id)
        #             position_id = 0
        # else:
        #     example["position_ids"] = list(range(len(special_tokens_mask_of_example)))

        # Prepare to start building the next example
        self.buffer = self.buffer[self.max_length :]
        return example
