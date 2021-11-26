from dataclasses import dataclass, field

from transformers import HfArgumentParser

from src.processing import (
    RobertaProcessor,
    AlbertProcessor,
)
from src.utils import load_corpora

name_to_processor = {
    "roberta": RobertaProcessor,
    "albert": AlbertProcessor,
}


@dataclass
class Arguments:
    model_name: str = field(
        default="roberta",
        metadata={
            "choices": [
                "roberta-base",
                "albert-base-v2",
            ]
        },
    )
    tokenizer_dir: str = field(
        default="tokenizers/roberta",
    )
    corpora_dir: str = field(
        default="corpora/kowiki",
    )
    max_length: int = field(
        default=512,
    )
    num_proc: int = field(
        default=4,
    )
    batch_size: int = field(
        default=1000,
    )
    writer_batch_size: int = field(
        default=1000,
    )


def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    processor = name_to_processor[args.model_name](
        args.tokenizer_dir, args.max_length
    )

    corpora = load_corpora(args.corpora_dir)
    
    #####
    from numpy.random import choice
    total_size = len(corpora)
    sample_size = int(total_size * 0.01)
    corpora = corpora.select(indices=choice(range(total_size), sample_size))
    #####

    dataset = corpora.map(
        lambda examples: processor(examples["text"]),
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        remove_columns=corpora.column_names,
    )
    dataset.save_to_disk("datasets/" + args.model_name)


if __name__ == "__main__":
    main()
