from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lassl import MODEL_TYPE_TO_COLLATOR, MODEL_TYPE_TO_PROCESSOR
from lassl.utils import load_corpora


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="bert", choices=["bert", "roberta", "gpt2", "albert", "bart", "t5"])
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    test_dirpath = Path(__file__).parent
    test_corpus_dirpath = test_dirpath / "corpus"
    test_tokenizer_dirpath = test_dirpath / f"tokenizers/{args.model_type}"

    test_corpus = load_corpora(test_corpus_dirpath, corpus_type="sent_text")
    reference_tokenizer = AutoTokenizer.from_pretrained(test_tokenizer_dirpath)
    test_processor = MODEL_TYPE_TO_PROCESSOR[args.model_type](
        model_name_or_path=test_tokenizer_dirpath, max_length=128
    )
    test_ds = test_corpus.map(
        lambda examples: test_processor(examples["text"]),
        batched=True,
        remove_columns=test_corpus.column_names,
    )

    list_of_input_ids = test_ds["input_ids"][: args.batch_size]

    print("----------Output of processor----------")
    list_of_input_tokens = [reference_tokenizer.convert_ids_to_tokens(input_ids) for input_ids in list_of_input_ids]
    list_of_input_string = [
        reference_tokenizer.convert_tokens_to_string(input_tokens) for input_tokens in list_of_input_tokens
    ]

    for idx in range(args.batch_size):
        print(f"----------example {idx}----------")
        print(f"input_tokens:\n{list_of_input_tokens[idx]}")
        print(f"input_string:\n{list_of_input_string[idx]}")
        print(f"input_ids:\n{list_of_input_ids[idx]}")

    test_collator = MODEL_TYPE_TO_COLLATOR[args.model_type](reference_tokenizer)
    dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=test_collator)
    training_examples = next(iter(dl))

    print("----------Output of collator----------")
    for idx in range(args.batch_size):
        print(f"----------example {idx}----------")
        training_example = {key: training_examples[key][idx] for key in training_examples.keys()}
        input_tokens = reference_tokenizer.convert_ids_to_tokens(training_example["input_ids"].tolist())
        input_string = reference_tokenizer.convert_tokens_to_string(input_tokens)

        if args.model_type in ["bart", "t5"]:
            decoder_input_tokens = reference_tokenizer.convert_ids_to_tokens(
                training_example["decoder_input_ids"].tolist()
            )
            decoder_input_string = reference_tokenizer.convert_tokens_to_string(decoder_input_tokens)
            decoder_label_tokens = reference_tokenizer.convert_ids_to_tokens(training_example["labels"].tolist())
            decoder_label_string = reference_tokenizer.convert_tokens_to_string(decoder_label_tokens)
            print(f"input_tokens:\n{input_tokens}")
            print(f"decoder_input_tokens:\n{decoder_input_tokens}")
            print(f"decoder_label_tokens:\n{decoder_label_tokens}")
            print(f"input_string:\n{input_string}")
            print(f"decoder_input_string:\n{decoder_input_string}")
            print(f"decoder_label_string:\n{decoder_label_string}")
        else:
            print(f"input_tokens:\n{input_tokens}")
            print(f"input_string:\n{input_string}")

        pprint(training_example)


if __name__ == "__main__":
    main()
