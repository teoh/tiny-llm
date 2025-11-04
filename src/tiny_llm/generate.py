from typing import Callable

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        # shape: (N.., L) -> (N.., L, num_vocab)
        output_logits = model(y)
        # shape: (N.., L, num_vocab) -> (N.., num_vocab)
        logits_for_next_token = output_logits[:, -1, :]
        # shape: (N.., num_vocab) -> (N..)
        next_tokens = mx.argmax(logits_for_next_token, axis=-1)
        # since for now (nov 3rd 2025) batch_size=1
        return next_tokens[0].item()

    tokens = tokenizer.encode(prompt)
    while True:
        next_token = _step(model=model, y=mx.expand_dims(mx.array(tokens), axis=0))
        tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        tokenizer.detokenizer.add_token(next_token)
        print(tokenizer.detokenizer.last_segment, end="", flush=True)
    return ""


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
