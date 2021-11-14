from abc import ABC, abstractmethod
from typing import Dict, List, Union

import datasets
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DatasetBuilder(ABC):
    """Abstract builder class compatible with datasets library."""

    @abstractmethod
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Instantiate DatasetBuilder class to be re-implement in subclass

        Args:
            tokenizer: tokenizer for serialization
        """
        self.tokenizer = tokenizer
        self.features = datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value("int32")),
            }
        )
        self.buffer = []

    @abstractmethod
    def make_examples(self, batch: Dict) -> Dict:
        """Build examples (consecutive tokens) from sentneces."""
        pass

    @abstractmethod
    def _add_document(self, input_ids: List[int]) -> Union[transformers.BatchEncoding, None]:
        """Add a document to a current example being built."""
        pass

    @abstractmethod
    def _create_example(self) -> transformers.BatchEncoding:
        """Generate BatchEncoding from a list of sentences."""
        pass
