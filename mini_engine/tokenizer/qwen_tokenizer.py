"""Qwen2.5's real tokenizer, re-implemented: GPT-2-style byte-level BPE with
Qwen's vocab/merges loaded from tokenizer.json. The `tokenizers` library is
used only in tests, to verify this implementation matches exactly.

Also implements the two production concerns from Week 1/4:
- chat templating (Qwen's <|im_start|> format), and
- incremental detokenization: a token can end mid-UTF-8-codepoint, so a
  streaming server must buffer bytes until they form valid UTF-8 (this is
  why TGI streams *detokenization* incrementally, not per-token decode()).
"""

import json
import os

import regex  # supports \p{L} etc., required by the GPT-2 split pattern

# Pre-tokenizer split pattern from Qwen's tokenizer.json (GPT-4 style).
SPLIT_PATTERN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte->printable-char map, so merges operate on
    strings without whitespace/control-char ambiguity."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class QwenTokenizer:
    def __init__(self, tokenizer_json_path: str):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            tk = json.load(f)

        self.vocab: dict[str, int] = tk["model"]["vocab"]
        merges = tk["model"]["merges"]
        # merges may be ["a b", ...] or [["a","b"], ...] depending on version
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            pair = tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
            self.merge_ranks[pair] = rank

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.special_tokens: dict[str, int] = {
            t["content"]: t["id"] for t in tk.get("added_tokens", [])
        }
        self.special_to_content = {v: k for k, v in self.special_tokens.items()}
        self.im_start_id = self.special_tokens[IM_START]
        self.im_end_id = self.special_tokens[IM_END]
        self.eos_ids = {self.im_end_id, self.special_tokens["<|endoftext|>"]}

        self._special_split = regex.compile(
            "(" + "|".join(regex.escape(t) for t in self.special_tokens) + ")"
        )
        self._bpe_cache: dict[str, list[int]] = {}

    @classmethod
    def from_model(cls, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> "QwenTokenizer":
        from huggingface_hub import snapshot_download

        return cls(os.path.join(snapshot_download(model_id), "tokenizer.json"))

    # --- encoding -----------------------------------------------------------

    def _bpe(self, piece: str) -> list[int]:
        """Merge loop over one pre-token, lowest-rank pair first."""
        if piece in self._bpe_cache:
            return self._bpe_cache[piece]
        parts = list(piece)
        while len(parts) >= 2:
            pairs = set(zip(parts, parts[1:]))
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            _, (a, b) = min(ranked)
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        ids = [self.vocab[p] for p in parts]
        self._bpe_cache[piece] = ids
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        ids: list[int] = []
        chunks = self._special_split.split(text) if allow_special else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if allow_special and chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
                continue
            for piece in SPLIT_PATTERN.findall(chunk):
                mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                ids.extend(self._bpe(mapped))
        return ids

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """Qwen2.5 chat format; appends the assistant generation prompt."""
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system",
                         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + messages
        text = ""
        for m in messages:
            text += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
        text += f"{IM_START}assistant\n"
        return self.encode(text, allow_special=True)

    # --- decoding -----------------------------------------------------------

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self.special_to_content:
            return self.special_to_content[token_id].encode("utf-8")
        s = self.id_to_token[token_id]
        return bytes(self.byte_decoder[c] for c in s)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")


class IncrementalDetokenizer:
    """Streams text token-by-token without ever emitting a broken codepoint.

    Bytes accumulate in a buffer; we emit the longest prefix that is valid
    UTF-8 and keep the (max 3-byte) incomplete tail for the next token.
    """

    def __init__(self, tokenizer: QwenTokenizer):
        self.tok = tokenizer
        self._pending = b""

    def push(self, token_id: int) -> str:
        buf = self._pending + self.tok.token_bytes(token_id)
        try:
            text = buf.decode("utf-8")
            self._pending = b""
            return text
        except UnicodeDecodeError as e:
            # e.start is where the (possibly incomplete) trailing sequence begins.
            text = buf[: e.start].decode("utf-8")
            self._pending = buf[e.start :]
            # A tail longer than 4 bytes can never complete — flush as replacement.
            if len(self._pending) > 4:
                text += self._pending.decode("utf-8", errors="replace")
                self._pending = b""
            return text

    def flush(self) -> str:
        text = self._pending.decode("utf-8", errors="replace")
        self._pending = b""
        return text
