from lassl.callbacks import TokenizerSaveCallback
from lassl.collators import (
    DataCollatorForAlbert,
    DataCollatorForBart,
    DataCollatorForBert,
    DataCollatorForElectra,
    DataCollatorForGpt2,
    DataCollatorForRoberta,
    DataCollatorForT5,
    DataCollatorForUL2
)
from lassl.processors import (
    AlbertProcessor,
    BartProcessor,
    BertProcessor,
    ElectraProcessor,
    GPT2Processor,
    RobertaProcessor,
    T5Processor,
    UL2Processor
)

# in train_tokenizer.py
MODEL_TYPE_TO_PREDEFINED_MODEL = {
    "bert": "bert-base-cased",
    "gpt2": "gpt2",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "t5" : "t5-base",
    "ul2" : "google/ul2",
    "electra": "google/electra-base-discriminator",
}

# in serialize_corpora.py
MODEL_TYPE_TO_PROCESSOR = {
    "bert": BertProcessor,
    "roberta": RobertaProcessor,
    "gpt2": GPT2Processor,
    "albert": AlbertProcessor,
    "bart": BartProcessor,
    "ul2" : UL2Processor,
    "t5": T5Processor,
    "electra": ElectraProcessor,
}

# in pretrain_language_model.py
MODEL_TYPE_TO_COLLATOR = {
    "bert": DataCollatorForBert,
    "albert": DataCollatorForAlbert,
    "roberta": DataCollatorForRoberta,
    "gpt2": DataCollatorForGpt2,
    "bart": DataCollatorForBart,
    "t5" : DataCollatorForT5,
    "ul2" : DataCollatorForUL2,
    "electra": DataCollatorForElectra,

}
