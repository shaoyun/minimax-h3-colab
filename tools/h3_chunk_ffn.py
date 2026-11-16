"""Lossless feed-forward chunking node for MiniMax H3.

Splits the token sequence in each DiT block's SwiGLU MLP along dim=-2 when the
token count exceeds `seq_threshold`. Because MLP layers are point-wise across
tokens, the output is mathematically identical to unchunked execution while
cutting peak activation VRAM by ~chunks x on high-resolution (e.g. 1080P) runs.
"""
import functools
import torch


def _make_chunked_forward(orig_forward, chunks, seq_threshold):
    base_forward = getattr(orig_forward, "_orig_forward", orig_forward)

    @functools.wraps(base_forward)
    def _chunked_forward(*args, **kwargs):
        if args:
            x = args[0]
            rest = args[1:]
            use_kw = False
        elif "x" in kwargs:
            x = kwargs["x"]
            rest = ()
            use_kw = True
        else:
            return base_forward(*args, **kwargs)

        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            return base_forward(*args, **kwargs)

        seq_len = x.shape[-2]
        if chunks <= 1 or seq_len <= seq_threshold:
            return base_forward(*args, **kwargs)

        step = ((seq_len + chunks - 1) // chunks + 63) // 64 * 64
        step = max(64, step)
        if step >= seq_len:
            return base_forward(*args, **kwargs)

        def call_slice(x_slice):
            if use_kw:
                kw = dict(kwargs)
                kw["x"] = x_slice
                return base_forward(**kw)
            return base_forward(x_slice, *rest, **kwargs)

        first_len = min(step, seq_len)
        y0 = call_slice(x.narrow(-2, 0, first_len).contiguous())
        out_shape = list(y0.shape)
        out_shape[-2] = seq_len
        out = torch.empty(out_shape, dtype=y0.dtype, device=y0.device)
        out.narrow(-2, 0, first_len).copy_(y0)
        del y0

        for start in range(first_len, seq_len, step):
            length = min(step, seq_len - start)
            yi = call_slice(x.narrow(-2, start, length).contiguous())
            out.narrow(-2, start, length).copy_(yi)
            del yi
        return out

    _chunked_forward._is_h3_chunk_ffn = True
    _chunked_forward._orig_forward = base_forward
    return _chunked_forward


def _should_enable_chunk_ffn(max_vram_gib=28.0):
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            if total >= max_vram_gib * (1024 ** 3):
                return False
    except Exception:
        pass
    return True


class MiniMaxChunkFeedForward:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "chunks": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "seq_threshold": ("INT", {"default": 24576, "min": 0, "max": 1000000, "step": 64}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "minimax"

    def patch(self, model, chunks=2, seq_threshold=24576):
        chunks = int(chunks)
        seq_threshold = int(seq_threshold)
        if chunks <= 1 or not _should_enable_chunk_ffn():
            return (model,)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            return (m,)
        for i, block in enumerate(blocks):
            mlp = getattr(block, "mlp", None)
            orig_forward = getattr(mlp, "forward", None)
            if orig_forward is None:
                continue
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.mlp.forward",
                _make_chunked_forward(orig_forward, chunks, seq_threshold),
            )
        return (m,)


_already_registered = False
try:
    import nodes  # type: ignore
    _already_registered = "MiniMaxChunkFeedForward" in getattr(nodes, "NODE_CLASS_MAPPINGS", {})
except Exception:
    pass

if _already_registered:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    NODE_CLASS_MAPPINGS = {
        "MiniMaxChunkFeedForward": MiniMaxChunkFeedForward,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "MiniMaxChunkFeedForward": "MiniMax H3 Chunk FeedForward",
    }
