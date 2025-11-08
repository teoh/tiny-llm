import copy

import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            scaled = logprobs / temp
            return mx.random.categorical(scaled)

    return sample
