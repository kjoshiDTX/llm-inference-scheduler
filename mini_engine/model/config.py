"""Model configuration for Qwen2.5-0.5B-Instruct.

Values from the HF config.json / Qwen2 tech report architecture table.
The GQA layout (14 query heads, 2 KV heads) is what makes the KV cache
7x smaller than full MHA would be — see notes/architecture.md for the math.
"""

from dataclasses import dataclass


@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64  # hidden_size / num_attention_heads
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    # Special token ids (from tokenizer_config / generation_config)
    eos_token_ids: tuple = (151645, 151643)  # <|im_end|>, <|endoftext|>

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache bytes per token: 2 (K and V) x layers x kv_heads x head_dim x dtype_bytes.

        For fp16: 2 * 24 * 2 * 64 * 2 = 12,288 bytes = 12 KB/token.
        """
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * 2

    @classmethod
    def from_hf_config(cls, config_path: str) -> "Qwen2Config":
        import json

        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["hidden_size"] // cfg["num_attention_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        )
