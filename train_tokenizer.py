from argparse import ArgumentParser
from pathlib import Path

from numpy import random
from transformers import AutoTokenizer

from lassl import MODEL_TYPE_TO_PREDEFINED_MODEL
from lassl.utils import batch_iterator, load_corpora

CACHE_DIR = str(Path(__file__).parent.resolve() / ".cache")


def get_args():
    parser = ArgumentParser()
    data_arguments = parser.add_argument_group("data")
    data_arguments.add_argument("--corpora_dirpath", type=str, required=True)
    data_arguments.add_argument(
        "--corpus_type", choices=["docu_text", "docu_json", "sent_text", "sent_json"], type=str, default="docu_json"
    )
    data_arguments.add_argument("--batch_size", type=int, default=1000)
    data_arguments.add_argument("--sampling_ratio", type=int, default=0.01)
    data_arguments.add_argument("--seed", type=int, default=42)
    data_arguments.add_argument("--output_base_dirpath", type=str, default="tokenizers")

    model_arguments = parser.add_argument_group("model")
    model_arguments.add_argument(
        "--model_type",
        choices=["bert-cased", "gpt2", "roberta", "albert", "bart", "t5", "ul2"],
        type=str,
        required=True,
    )
    model_arguments.add_argument("--vocab_size", type=int, default=51200)
    model_arguments.add_argument("--min_frequency", type=int, default=2)
    model_arguments.add_argument("--additional_special_tokens", nargs="*", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)

    corpora = load_corpora(dirpath=args.corpora_dirpath, corpus_type=args.corpus_type, cache_dir=CACHE_DIR)

    assert args.sampling_ratio > 0, "sampling_ratio must be greater than 0."

    if 0 < args.sampling_ratio < 1.0:
        total_size = len(corpora)
        sample_size = int(total_size * args.sampling_ratio)
        corpora = corpora.select(indices=random.choice(total_size, sample_size, replace=False))
    else:
        print("Since sampling_ratio >= 1.0, all corpora will be used.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE_TO_PREDEFINED_MODEL[args.model_type])
    data_iterator = batch_iterator(corpora, batch_size=args.batch_size)

    if args.additional_special_tokens:
        print(f"Additional Special Tokens : {args.additional_special_tokens}")
        assert len(args.additional_special_tokens) == len(
            set(args.additional_special_tokens)
        ), "Each additional special tokens must be unique."
        assert not set(tokenizer.all_special_tokens).intersection(
            set(args.additional_special_tokens)
        ), "Each additional special tokens are not of default special tokens from tokenizer."
        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            new_special_tokens=args.additional_special_tokens,
        )
    else:
        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    tokenizer.save_pretrained(f"{args.output_base_dirpath}/{args.model_type}")


if __name__ == "__main__":
    main()
