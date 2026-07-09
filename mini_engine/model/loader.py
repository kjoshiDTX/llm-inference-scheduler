"""Load Qwen2.5 safetensors into a plain dict of torch tensors.

No transformers modeling code — huggingface_hub is used only to locate the
snapshot on disk. The weights land as raw tensors keyed by their original
names; qwen2.py knows the naming scheme.
"""

import json
import os

import torch
from safetensors.torch import load_file

from .config import Qwen2Config

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def snapshot_path(model_id: str = MODEL_ID) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def load_weights(
    model_id: str = MODEL_ID,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Qwen2Config, dict[str, torch.Tensor]]:
    path = snapshot_path(model_id)
    config = Qwen2Config.from_hf_config(os.path.join(path, "config.json"))

    weights: dict[str, torch.Tensor] = {}
    index_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    for shard in shards:
        for name, tensor in load_file(os.path.join(path, shard)).items():
            weights[name] = tensor.to(device=device, dtype=dtype)

    return config, weights
