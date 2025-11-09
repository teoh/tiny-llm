import copy

import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs_raw: mx.array):
        if top_k is not None:
            idxs = mx.argpartition(-logprobs_raw, kth=top_k - 1, axis=-1)
            mask = idxs[..., top_k:]
            logprobs = copy.deepcopy(logprobs_raw)
            logprobs[:, mask] = -mx.inf
        elif top_p is not None:
            assert 0, "not implemented yet"
        else:
            logprobs = logprobs_raw

        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            scaled = logprobs / temp
            return mx.random.categorical(scaled)

    return sample
