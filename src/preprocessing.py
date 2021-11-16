from transformers.utils.logging import get_logger
from transformers.testing_utils import CaptureLogger
from transformers import RobertaTokenizerFast


class RobertaPreProcesssor:
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
        self._max_length = max_length

    def create_training_examples(self, examples):
        tok_logger = get_logger("transformers.tokenization_utils_base")
        with CaptureLogger(tok_logger) as cl:
            training_examples = self._tokenizer(examples, padding=False, add_special_tokens=True, truncation=False, return_attention_mask=True, return_special_tokens_mask=True)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return training_examples

    def concatenate_training_examples(self, training_examples):
        # Concatenate all training examples.
        concatenated_training_examples = {k: sum(training_examples[k], []) for k in training_examples.keys()}
        total_length = len(concatenated_training_examples[list(training_examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self._max_length) * self._max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self._max_length] for i in range(0, total_length, self._max_length)]
            for k, t in concatenated_training_examples.items()
        }
        return result
