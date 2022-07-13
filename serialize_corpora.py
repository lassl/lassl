from dataclasses import dataclass, field

from transformers import HfArgumentParser

from lassl.processors import (
    AlbertProcessor,
    BertProcessor,
    GPT2Processor,
    RobertaProcessor,
    BartProcessor,
    T5Processor
)

model_type_to_processor = {
    "bert": BertProcessor,
    "roberta": RobertaProcessor,
    "gpt2": GPT2Processor,
    "albert": AlbertProcessor,
    "bart" : BartProcessor,
    "t5" : T5Processor
}

from lassl import MODEL_TYPE_TO_PROCESSOR
from lassl.utils import load_corpora


@dataclass
class Arguments:
    model_type: str = field(
        default="t5",
        metadata={
            "choices": [
                "bert",
                "roberta",
                "gpt2",
                "albert",
                "bart",
                "t5"
            ]
        },
    )
    tokenizer_dir: str = field(
        default = "tokenizers/t5"
        ### pretrained options
        # default="KETI-AIR/ke-t5-small",
        # default="hyunwoongko/kobart"
    )
    corpora_dir: str = field(
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
    max_length: int = field(
        default=512,
    )
    num_proc: int = field(
        default=1,
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
    keep_in_memory: bool = field(
        default=False,
    )


def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    data_processor = MODEL_TYPE_TO_PROCESSOR[args.model_type](args.tokenizer_dir, args.max_length)

    corpora = load_corpora(args.corpora_dir, corpus_type=args.corpus_type)

    dataset = corpora.map(
        lambda examples: data_processor(examples["text"]),
        num_proc=args.num_proc,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        load_from_cache_file=args.load_from_cache_file,
        keep_in_memory=args.keep_in_memory,
        remove_columns=corpora.column_names,
    )

    dataset.save_to_disk("datasets/" + args.model_type)
    data_processor.save_tokenizer("datasets/" + args.model_type)


if __name__ == "__main__":
    main()
