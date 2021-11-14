import datetime
import re
from typing import Generator, Union

import datasets


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


def list_dict_to_dict_list(list_of_dict: list):
    if len(list_of_dict) == 0:
        return {}
    return {k: [dic[k] for dic in list_of_dict] for k in list_of_dict[0]}


def secs_to_str(secs: float):
    """From Google Electra(https://github.com/google-research/electra/blob/master/util/training_utils.py#L68-L84)"""
    s = str(datetime.timedelta(seconds=int(round(secs))))
    s = re.sub("^0:", "", s)
    s = re.sub("^0", "", s)
    s = re.sub("^0:", "", s)
    s = re.sub("^0", "", s)
    return s


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]
