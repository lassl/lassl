from dataclasses import dataclass, field

from transformers import HfArgumentParser

from src.processing import (
    AlbertProcessor,
    BertProcessor,
    GPT2Processor,
    RobertaProcessor,
)
from src.utils import load_corpora

model_type_to_processor = {
    "bert": BertProcessor,
    "roberta": RobertaProcessor,
    "gpt2": GPT2Processor,
    "albert": AlbertProcessor,
}


@dataclass
class Arguments:
    model_type: str = field(
        default="roberta",
        metadata={
            "choices": [
                "bert",
                "roberta",
                "gpt2",
                "albert",
            ]
        },
    )
    tokenizer_dir: str = field(
        default="tokenizers/roberta",
    )
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
    load_from_cache_file: bool = field(
        default=True,
    )


def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    processor = model_type_to_processor[args.model_type](args.tokenizer_dir, args.max_length)

    corpora = load_corpora(args.corpora_dir, corpus_type=args.corpus_type)

    dataset = corpora.map(
        lambda examples: processor(examples["text"]),
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        load_from_cache_file=args.load_from_cache_file,
        remove_columns=corpora.column_names,
    )
    dataset.save_to_disk("datasets/" + args.model_type)
    processor.save_tokenizer("datasets/" + args.model_type)


if __name__ == "__main__":
    main()
