import copy

import mlx.core as mx


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs_raw: mx.array):
        if top_k is not None:
            idxs = mx.argpartition(-logprobs_raw, kth=top_k - 1, axis=-1)
            mask = idxs[..., top_k:]
            logprobs = copy.deepcopy(logprobs_raw)
            # TODO: i dont think this works for arbitrary dimensions
            logprobs[:, mask] = -mx.inf
        elif top_p is not None:
            # for each row, get the indices that would sort that row in descending order, then sort
            sorted_idxs = mx.argsort(-logprobs_raw, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs_raw, sorted_idxs, axis=-1)
            # get cumsum since we'll do top p from here
            cumul_logprobs = mx.cumsum(sorted_logprobs, axis=-1)
            # get the idxs of the sorted array where cumul values are outside the top p
            sorted_mask = cumul_logprobs > top_p
            # map idx back to original
            inv_idx = mx.argsort(sorted_idxs, axis=-1)
            # "convert" this mask to back to original array
            original_mask = mx.take_along_axis(sorted_mask, inv_idx, axis=-1)
            # any prob not part of cumul top p gets -inf so that we ignore it in distributoin
            logprobs = mx.where(original_mask, -mx.inf, logprobs_raw)
        else:
            logprobs = logprobs_raw

        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            scaled = logprobs / temp
            return mx.random.categorical(scaled)

    return sample
