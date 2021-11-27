from pathlib import Path
from typing import Generator, Union

import datasets
from datasets import load_dataset

SENTENCE_PER_LINE_SCRIPT = str(Path("./src/scripts/sentence_per_line.py").absolute())


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def load_corpora(dir_path, text_type_per_line="docu"):
    corpora_dir = Path(dir_path).absolute()
    all_file_paths = []

    if text_type_per_line == "docu":
        list_of_file_paths = [
            str(file_path) for file_path in corpora_dir.rglob("*.json")
        ]

        if not list_of_file_paths:
            raise Exception("source files must have 'json' extension.")

        all_file_paths.extend(list_of_file_paths)
        return load_dataset("json", data_files=all_file_paths, split="train")
    elif text_type_per_line == "sent":
        # sent
        list_of_file_paths = [
            str(file_path) for file_path in corpora_dir.rglob("*.txt")
        ]
        if not list_of_file_paths:
            raise Exception("source files must have 'txt' extension.")

        all_file_paths.extend(list_of_file_paths)
        return load_dataset(
            SENTENCE_PER_LINE_SCRIPT, data_files=all_file_paths, split="train"
        )
    else:
        raise NotImplementedError("Implementing loading scripts along your text type.")


def get_params_without_weight_decay_ln(
    named_params: Union[list, Generator], weight_decay: float = 0.1
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
