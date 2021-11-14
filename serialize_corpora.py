from dataclasses import dataclass, field
from src.utils import load_corpora
from src.preprocessing import RobertaPreProcesssor
from transformers import HfArgumentParser


name_to_preprocessor = {
    "roberta": RobertaPreProcesssor,
}


@dataclass
class Argument:
    model_name: str = field(
        default= "roberta",
        metadata = {
            "choices": [
                "roberta",
            ]
        }
    )
    tokenizer_dir: str = field(
        default = "tokenizers/roberta",
    )
    corpora_dir: str = field(
        default = "corpora/kowiki",
    )
    max_length: int = field(
        default = 512,
    )
    num_proc: int = field(
        default = 4,
    )
    batch_size: int = field(
        default = 1000,
    )
    writer_batch_size: int = field(
        default = 1000,
    )


def main():
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    preprocessor = name_to_preprocessor[args.model_name](args.tokenizer_dir, args.max_length)

    corpora = load_corpora(args.corpora_dir)
    intermediate_dataset = corpora.map(
        lambda examples: preprocessor.create_training_examples(examples["text"]),
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        remove_columns=corpora.column_names,
    )
    dataset = intermediate_dataset.map(
        lambda examples: preprocessor.concatenate_training_examples(examples),
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
    )

    dataset.save_to_disk("datasets/" + args.model_name)


if __name__ == "__main__":
    main()
