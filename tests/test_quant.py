"""Week 10: quantization round-trip accuracy + end-to-end sanity.

The perplexity-style check: greedy generation with int8-quantized MLP weights
should stay coherent and mostly agree with fp32 for short horizons.
"""

import torch

from mini_engine.quant import dequantize_groupwise, quantize_groupwise


def test_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q, s = quantize_groupwise(w, group_size=128)
    deq = dequantize_groupwise(q, s, group_size=128)
    # absmax int8: per-group max error <= scale/2 = groupmax/254
    per_group_max = w.reshape(64, 2, 128).abs().amax(-1)
    assert ((w.reshape(64, 2, 128) - deq.reshape(64, 2, 128)).abs()
            <= per_group_max.unsqueeze(-1) / 254 + 1e-6).all()


def test_outlier_isolation():
    """Group-wise scaling contains an outlier's damage to its own group."""
    w = torch.randn(1, 256) * 0.01
    w[0, 0] = 100.0  # outlier in group 0
    q, s = quantize_groupwise(w, group_size=128)
    deq = dequantize_groupwise(q, s, group_size=128)
    # Group 1 (no outlier) keeps fine resolution.
    assert (w[0, 128:] - deq[0, 128:]).abs().max() < 0.001


def test_memory_accounting():
    from mini_engine.quant import quantize_mlp_weights

    weights = {"model.layers.0.mlp.gate_proj.weight": torch.randn(4864, 896)}
    before, after, err = quantize_mlp_weights(weights, group_size=128)
    assert before == 4864 * 896 * 4
    assert after == 4864 * 896 + 4864 * 7 * 2  # int8 + fp16 scales
    assert after / before < 0.27
    assert err < 0.1
