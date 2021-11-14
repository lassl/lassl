from transformers import RobertaTokenizerFast


class RobertaPreProcesssor:
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
        self._max_length = max_length

    def create_training_examples(self, examples):
        training_examples = self._tokenizer(examples, padding=False, add_special_tokens=True, truncation=False, return_attention_mask=True, return_special_tokens_mask=True)
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
