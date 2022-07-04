from transformers.trainer_callback import TrainerCallback


class TokenizerSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            kwargs["tokenizer"].save_pretrained(args.output_dir)
