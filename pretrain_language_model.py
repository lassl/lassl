import logging
import os
from dataclasses import dataclass, field
from src.collator import DataCollatorForSOP
from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    data_dir: str = field(default="datasets/roberta")
    mlm_probability: float = field(
        default=0.15,
    )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="roberta-base",
        metadata={
            "choices": [
                "roberta-base",
                "albert-base-v2",
            ]
        },
    )
    tokenizer_dir: str = field(
        default="tokenizers/roberta",
    )


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    dataset = Dataset.load_from_disk(data_args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_dir)
    config = AutoConfig.from_pretrained(model_args.model_name)
    model = AutoModelForPreTraining.from_config(config)
    model.resize_token_embeddings(tokenizer.vocab_size)

    if model_args.model_name in ["roberta-base"]:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=data_args.mlm_probability,
        )
    elif model_args.model_name in ["gpt2"]:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
            )
    elif model_args.model_name in ["albert-base-v2"]:
        data_collator = DataCollatorForSOP(
            tokenizer=tokenizer, mlm_probability=data_args.mlm_probability,
        )
    else:
        raise NotImplementedError
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    max_train_samples = len(dataset)
    metrics["train_samples"] = min(max_train_samples, len(dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
