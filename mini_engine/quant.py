"""Week 10: group-wise absmax int8 weight quantization, pure PyTorch.

Idea: weights within a small group (128 output values along the input dim)
share one fp16 scale; each value stores as int8. 4 bytes -> ~1.03 bytes per
weight. Group-wise (not per-tensor) because outlier weights would otherwise
stretch the scale and crush everyone else's precision — the same observation
that motivates AWQ/GPTQ, minus their calibration step.

Weight-only (not weight+activation): at small batch, decode is memory-
bandwidth-bound, so shrinking the weights you must stream from DRAM is the
whole win; activations stay fp32 and the matmul runs in fp32 after
dequantization. KV-cache quantization (fp8 KV = 2x more conversations per
GPU) is the same math applied to Week 3's 12 KB/token.
"""

import torch


def quantize_groupwise(w: torch.Tensor, group_size: int = 128):
    """w: [out, in] -> (int8 [out, in], scales fp32 [out, in/group_size])."""
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} % {group_size} != 0"
    g = w.reshape(out_dim, in_dim // group_size, group_size)
    scales = g.abs().amax(dim=-1).clamp(min=1e-8) / 127.0  # [out, groups]
    q = torch.round(g / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q.reshape(out_dim, in_dim), scales


def dequantize_groupwise(q: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    out_dim, in_dim = q.shape
    g = q.reshape(out_dim, in_dim // group_size, group_size).float()
    return (g * scales.unsqueeze(-1)).reshape(out_dim, in_dim)


MLP_KEYS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def quantize_mlp_weights(weights: dict[str, torch.Tensor], group_size: int = 128):
    """Quantize->dequantize all MLP weights in place (simulated int8 storage).

    Returns (bytes_before, bytes_after, max_abs_err) over quantized tensors.
    Simulated because we keep fp32 compute; the memory numbers are what real
    int8 storage would use.
    """
    before = after = 0
    max_err = 0.0
    for name in list(weights):
        if not name.endswith(MLP_KEYS):
            continue
        w = weights[name]
        q, scales = quantize_groupwise(w.float(), group_size)
        deq = dequantize_groupwise(q, scales, group_size)
        max_err = max(max_err, (w.float() - deq).abs().max().item())
        before += w.numel() * w.element_size()
        after += q.numel() + scales.numel() * 2  # int8 + fp16 scales
        weights[name] = deq.to(w.dtype)
    return before, after, max_err
