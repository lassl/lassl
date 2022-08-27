from collections import Counter

import pytest
from datasets import load_dataset

from lassl.blender import DatasetBlender


def test_blending():
    try:
        from langid import classify
    except ImportError as _:
        raise ImportError(
            "To test dataset blending, you need to install langid. "
            "Please install langid using `pip install langid`."
        )

    en = load_dataset("squad").data["train"]["context"]
    ko = load_dataset("oscar", "unshuffled_deduplicated_ko").data["train"]["text"]
    ja = load_dataset("amazon_reviews_multi", "ja").data["train"]["review_body"]

    weights = {"en": 0.2, "ko": 0.5, "ja": 0.3}
    datasets = {"en": en, "ko": ko, "ja": ja}

    blend = DatasetBlender(
        datasets=list(datasets.values()),
        weights=list(weights.values()),
    )

    langs = [classify(str(blend[i]))[0] for i in range(10)]
    counts = Counter(langs)

    assert int(counts["ko"]) == int(weights["ko"] * 10)
    assert int(counts["en"]) == int(weights["en"] * 10)
    assert int(counts["ja"]) == int(weights["ja"] * 10)
    print("All tests are passed ;)")
