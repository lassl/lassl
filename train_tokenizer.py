from dataclasses import dataclass, field
from typing import List, Optional

from numpy.random import choice
from transformers import AutoTokenizer, HfArgumentParser

from src.utils import batch_iterator, load_corpora

model_type_to_predefined_model = {
    "bert-uncased": "bert-base-uncased",
    "bert-cased": "bert-base-cased",
    "gpt2": "gpt2",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "electra": "google/electra-small-discriminator",
}


@dataclass
class DataArguments:
    corpora_dir: str = field(
        default="corpora/kowiki",
    )
    corpus_type: str = field(
        default="docu_json",
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
        default=0.1,
    )


@dataclass
class ModelArguments:
    model_type: str = field(
        default="roberta",
        metadata={
            "choices": [
                "bert-uncased",
                "bert-cased",
                "gpt2",
                "roberta",
                "albert",
                "electra",
            ]
        },
    )
    vocab_size: int = field(
        default=30000,
    )
    min_frequency: int = field(
        default=2,
    )
    additinal_special_tokens: Optional[List[str]] = field(
        default=None,
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()
    corpora = load_corpora(data_args.corpora_dir, data_args.corpus_type)

    if 0 < data_args.sampling_ratio < 1.0:
        total_size = len(corpora)
        sample_size = int(total_size * data_args.sampling_ratio)
        sampled_corpora = corpora.select(indices=choice(range(total_size), sample_size))

    tokenizer = AutoTokenizer.from_pretrained(model_type_to_predefined_model[model_args.model_type])
    data_iterator = batch_iterator(sampled_corpora, batch_size=data_args.batch_size)

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
    tokenizer.save_pretrained("tokenizers/" + model_args.model_type)


if __name__ == "__main__":
    main()
