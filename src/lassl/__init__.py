from lassl.callbacks import TokenizerSaveCallback
from lassl.collators import (
    DataCollatorForAlbert,
    DataCollatorForBart,
    DataCollatorForBert,
    DataCollatorForGpt2,
    DataCollatorForRoberta,
)
from lassl.processors import (
    AlbertProcessor,
    BartProcessor,
    BertProcessor,
    GPT2Processor,
    RobertaProcessor,
)

# in train_tokenizer.py
MODEL_TYPE_TO_PREDEFINED_MODEL = {
    "bert-cased": "bert-base-cased",
    "gpt2": "gpt2",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
}

# in serialize_corpora.py
MODEL_TYPE_TO_PROCESSOR = {
    "bert": BertProcessor,
    "roberta": RobertaProcessor,
    "gpt2": GPT2Processor,
    "albert": AlbertProcessor,
    "bart": BartProcessor,
}

# in pretrain_language_model.py
MODEL_TYPE_TO_COLLATOR = {
    "bert": DataCollatorForBert,
    "albert": DataCollatorForAlbert,
    "roberta": DataCollatorForRoberta,
    "gpt2": DataCollatorForGpt2,
    "bart": DataCollatorForBart,
}
