import glob
import os
import datasets

import datetime
import re
from typing import Generator, Union

from datasets import load_dataset


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def find_files(data_dir, data_files):
    if data_dir is None and data_files is None:
        raise ValueError("At least one of data_files or data_dir must be specified")
    elif data_dir is not None and data_files is None:
        pathname = data_dir + "/**/*.*"
        data_files = glob.glob(pathname, recursive=True)
        data_files = [p for p in data_files if p.endswith(".txt") or p.endswith(".json")]
    elif data_dir is not None and data_files is not None:
        if isinstance(data_files, str):
            data_files = [data_files]
        data_files = [os.path.join(data_dir, d) for d in data_files]
    return data_files


def load_corpora(data_dir, cache_dir=None, split="train", data_files=None):
    data_files = find_files(data_dir, data_files)
    if data_files[0].endswith(".txt"):
        return load_dataset(
            "text",
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=cache_dir,
            split=split,
        )
    elif data_files[0].endswith(".json"):
        return load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=cache_dir,
            split=split,
        )
    raise ValueError("File formats other than '.txt' and '.json' are not supported yet")


def secs_to_str(secs: float):
    """From Google Electra(https://github.com/google-research/electra/blob/master/util/training_utils.py#L68-L84)"""
    s = str(datetime.timedelta(seconds=int(round(secs))))
    s = re.sub("^0:", "", s)
    s = re.sub("^0", "", s)
    s = re.sub("^0:", "", s)
    s = re.sub("^0", "", s)
    return s


def get_params_without_weight_decay_ln(named_params: Union[list, Generator], weight_decay: float = 0.1):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
