import logging
import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
from transformers import (
    CONFIG_MAPPING,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from datasets import Dataset
from src.collator import DataCollatorForBertWithSOP, DataCollatorForSOP

logger = logging.getLogger(__name__)


def get_main_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_main_args()
    nested_args = OmegaConf.load(args.config_path)
    model_args = nested_args.model
    data_args = nested_args.data
    training_args = TrainingArguments(**nested_args.training)

    dataset = Dataset.load_from_disk(data_args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(data_args.data_dir)

    assert (
        model_args.model_type in CONFIG_MAPPING.keys()
    ), f"model_args.model_type must be one of {CONFIG_MAPPING.keys()}"
    model_config = CONFIG_MAPPING[model_args.model_type](**model_args)
    model = AutoModelForPreTraining.from_config(model_config)
    model.resize_token_embeddings(tokenizer.vocab_size)

    if model_args.model_type in ["bert"]:
        data_collator = DataCollatorForBertWithSOP(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        )

    elif model_args.model_type in ["roberta"]:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=data_args.mlm_probability,
        )
    elif model_args.model_type in ["gpt2"]:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    elif model_args.model_type in ["albert"]:
        data_collator = DataCollatorForSOP(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
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
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
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


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
