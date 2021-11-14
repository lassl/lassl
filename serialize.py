from dataclasses import dataclass, field
from src.dataset_loader import dataset_loader
from src.gpt2 import GPT2DatasetBuilder
from src.roberta import RobertaDatasetBuilder
from transformers import AutoTokenizer, HfArgumentParser


name_to_builder = {
    "gpt2": GPT2DatasetBuilder,
    "roberta": RobertaDatasetBuilder,
}

@dataclass
class Argument:
    model_name: str = field(
        default = "gpt2",
        metadata = {
            "choices": [
                "gpt2",
                "roberta",
            ]
        }
    )
    data_dir: str = field(
        default = "corpora/wiki",
    )
    tokenizer_dir: str = field(
        default = "tokenizers/bert",
    )
    max_length: int = field(
        default = 1024,
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = dataset_loader(args.data_dir)
    dataset_builder = name_to_builder[args.model_name](tokenizer=tokenizer, max_length=args.max_length, key="text")
    dataset = dataset.map(
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        function=dataset_builder.make_examples,
        remove_columns=dataset.column_names,
        features=dataset_builder.features,
    )
    dataset.save_to_disk("datasets/" + args.model_name)


if __name__ == "__main__":
    main()
