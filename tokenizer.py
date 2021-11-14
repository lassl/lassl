from dataclasses import dataclass, field
from numpy.random import choice
from src.dataset_loader import dataset_loader
from src.utils import batch_iterator
from transformers import AutoTokenizer, HfArgumentParser


name_to_predefined_model = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "electra": "google/electra-small-discriminator",
}

@dataclass
class DataArgument:
    data_dir: str = field(
        default = "corpora/wiki",
    )
    batch_size: int = field(
        default = 1000,
    )
    sampling_ratio: float = field(
        default = 0.1,
    )

@dataclass
class ModelArgument:
    model_name: str = field(
        default = "bert",
        metadata = {
            "choices": [
                "bert",
                "gpt2",
                "roberta",
                "albert",
                "electra",
            ]
        }
    )
    vocab_size: int = field(
        default = 30000,
    )
    model_max_length: int = field(
        default = 512,
    )


def main():
    parser = HfArgumentParser((DataArgument, ModelArgument))
    data_args, model_args = parser.parse_args_into_dataclasses()
    dataset = dataset_loader(data_args.data_dir)

    if 0 < data_args.sampling_ratio < 1.0:
        total_size = len(dataset)
        sample_size = int(total_size * data_args.sampling_ratio)
        dataset = dataset.select(indices=choice(range(total_size), sample_size))

    tokenizer = AutoTokenizer.from_pretrained(name_to_predefined_model[model_args.model_name])
    data_iterator = batch_iterator(dataset, batch_size=data_args.batch_size)
    tokenizer = tokenizer.train_new_from_iterator(data_iterator, vocab_size=model_args.vocab_size)
    tokenizer.save_pretrained("tokenizers/" + model_args.model_name)


if __name__ == "__main__":
    main()
