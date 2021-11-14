import glob
import os

from datasets import load_dataset


def find_data_files(data_dir, data_files):
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


def dataset_loader(data_dir, cache_dir=None, split="train", data_files=None):
    data_files = find_data_files(data_dir, data_files)
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
