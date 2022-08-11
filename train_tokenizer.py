from dataclasses import dataclass, field
from typing import List, Optional

from numpy.random import choice
from transformers import AutoTokenizer, HfArgumentParser

from lassl import MODEL_TYPE_TO_PREDEFINED_MODEL
from lassl.utils import batch_iterator, load_corpora


@dataclass
class DataArguments:
    corpora_dirpath: str = field(
        default="corpora",
    )
    corpus_type: str = field(
        default="sent_text",
        metadata={
            "choices": [
                "docu_text",
                "docu_json",
                "sent_text",
                "sent_json",
            ]
        },
    )
    batch_size: int = field(
        default=1000,
    )
    sampling_ratio: float = field(
        default=0.3,
    )
    output_base_dirpath: str = field(default="tokenizers")


@dataclass
class ModelArguments:
    model_type: str = field(
        default="t5",
        metadata={"choices": ["bert", "gpt2", "roberta", "albert", "bart", "t5", "electra"]},
    )
    vocab_size: int = field(
        default=50000,
    )
    min_frequency: int = field(
        default=2,
    )
    additional_special_tokens: Optional[List[str]] = field(
        default=None,
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    corpora = load_corpora(data_args.corpora_dirpath, data_args.corpus_type)

    assert data_args.sampling_ratio > 0, "sampling_ratio must be greater than 0."

    if 0 < data_args.sampling_ratio < 1.0:
        total_size = len(corpora)
        sample_size = int(total_size * data_args.sampling_ratio)
        corpora = corpora.select(indices=choice(total_size, sample_size, replace=False))
    else:
        print("Since sampling_ratio >= 1.0, all corpora will be used.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE_TO_PREDEFINED_MODEL[model_args.model_type])
    data_iterator = batch_iterator(corpora, batch_size=data_args.batch_size)

    if model_args.additional_special_tokens:
        assert len(model_args.additional_special_tokens) == len(
            set(model_args.additional_special_tokens)
        ), "Each additional special tokens must be unique."
        assert not set(tokenizer.all_special_tokens).intersection(
            set(model_args.additional_special_tokens)
        ), "Each additional special tokens are not of default special tokens from tokenizer."

        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=model_args.vocab_size,
            min_frequency=model_args.min_frequency,
            new_special_tokens=model_args.additional_special_tokens,
        )
    else:
        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=model_args.vocab_size,
            min_frequency=model_args.min_frequency,
        )
    tokenizer.save_pretrained(f"{data_args.output_base_dirpath}/{model_args.model_type}")


if __name__ == "__main__":
    main()
