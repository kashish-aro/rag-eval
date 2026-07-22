from typing import Any


class Generator:
    """Qwen 2.5 Instruct wrapped for greedy, deterministic generation.

    Greedy decoding (do_sample=False) plus a fixed torch seed at model load
    is the reproducibility guarantee the config-comparison harness depends
    on: same input twice must produce identical output, otherwise the
    faithfulness axis carries noise the harness cannot correct for.

    Model + tokenizer are injectable so unit tests never load transformers.
    The transformers import itself is deferred to first .generate() call.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        seed: int = 42,
        max_new_tokens: int = 200,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.seed)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()

    def generate(self, messages: list[dict]) -> str:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Strip the echoed prompt tokens; return only the newly generated ones.
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
