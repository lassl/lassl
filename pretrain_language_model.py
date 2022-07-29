import logging
import os
from argparse import ArgumentParser

from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (
    CONFIG_MAPPING,
    AutoModelForPreTraining,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from lassl import MODEL_TYPE_TO_COLLATOR, TokenizerSaveCallback


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
    collator_args = nested_args.collator
    training_args = TrainingArguments(**nested_args.training)

    train_dataset = Dataset.load_from_disk(data_args.data_dir)
    eval_dataset = None
    tokenizer = AutoTokenizer.from_pretrained(data_args.data_dir)

    assert (
        model_args.model_type in CONFIG_MAPPING.keys()
    ), f"model_args.model_type must be one of {CONFIG_MAPPING.keys()}"

    model_config = CONFIG_MAPPING[model_args.model_type](**model_args)
    model = AutoModelForPreTraining.from_config(model_config)
    model.resize_token_embeddings(tokenizer.vocab_size)

    data_collator = MODEL_TYPE_TO_COLLATOR[model_args.model_type](tokenizer=tokenizer, **collator_args)

    if training_args.do_eval and data_args.test_size:
        train_dataset, eval_dataset = (
            _ for _ in train_dataset.train_test_split(test_size=data_args.test_size).values()
        )
        logger.info(
            f"eval_dataset is set to {len(eval_dataset)} samples. pre-training is run by consuming training dataset whose number of samples are {len(train_dataset)}."
        )
    else:
        logger.info(
            f"eval_dataset is not set. pre-training is run by consuming training dataset whose number of samples are {len(train_dataset)}."
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[TokenizerSaveCallback()],
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

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
