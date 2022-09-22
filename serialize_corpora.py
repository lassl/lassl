from argparse import ArgumentParser
from pathlib import Path

from lassl import MODEL_TYPE_TO_PROCESSOR
from lassl.utils import load_corpora

CACHE_DIR = str(Path(__file__).parent.resolve() / ".cache")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["bert", "roberta", "gpt2", "albert", "bart", "t5", "ul2", "electra"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--corpus_type",
        choices=["docu_text", "docu_json", "sent_text", "sent_json"],
        type=str,
        default="docu_json",
    )
    parser.add_argument("--tokenizer_dirpath", type=str, required=True)
    parser.add_argument("--corpora_dirpath", type=str, default="datasets", required=True)
    parser.add_argument("--output_base_dirpath", type=str, default="datasets")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--writer_batch_size", type=int, default=1000)
    parser.add_argument("--no_load_from_cache_file", action="store_false", dest="load_from_cache_file")
    parser.add_argument("--keep_in_memory", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_processor = MODEL_TYPE_TO_PROCESSOR[args.model_type](args.tokenizer_dirpath, args.max_length)

    corpora = load_corpora(args.corpora_dirpath, corpus_type=args.corpus_type, cache_dir=CACHE_DIR)

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

    dataset.save_to_disk(f"{args.output_base_dirpath}/{args.model_type}")
    data_processor.save_tokenizer(f"{args.output_base_dirpath}/{args.model_type}")


if __name__ == "__main__":
    main()
