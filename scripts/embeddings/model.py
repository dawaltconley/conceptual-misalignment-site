"""GujiRoBERTa embedding: final-layer, per-occurrence mean over subword tokens.

For each passage we take the model's final hidden states and, for every
word-of-interest span in that passage, average the hidden vectors of the tokens
overlapping the span. A single hanzi is normally one token, but the overlap +
mean logic keeps this correct for multi-char neighbors or any subword splitting.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embeddings.occurrences import Segment

DEFAULT_MODEL = "hsc748NLP/GujiRoBERTa_fan"


def pick_device() -> str:
    """Return the compute device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_label(device: str) -> str:
    """Human-readable device label, e.g. ``cuda:0 (AMD Radeon Graphics)``."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return device


class Embedder:
    """Wraps a HF encoder and extracts per-occurrence span vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.device = device or pick_device()
        self.device_label = device_label(self.device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"{model_name} lacks a fast tokenizer; offset mapping "
                "(needed for span extraction) is unavailable."
            )
        self.model = AutoModel.from_pretrained(
            model_name).to(self.device).eval()
        self.hidden_size = self.model.config.hidden_size

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Model token count per text (special tokens excluded), for packing."""
        enc = self.tokenizer(texts, add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    def unk_words(self, words: Iterable[str]) -> dict[str, list[str]]:
        """Words that tokenize to at least one ``[UNK]`` → their token list.

        Spot-check for out-of-vocab characters among the terms we embed: any
        word here loses a character to ``[UNK]``, so its vector is degraded.
        """
        words = list(words)
        unk_id = self.tokenizer.unk_token_id
        encs = self.tokenizer(words, add_special_tokens=False)["input_ids"]
        return {
            w: self.tokenizer.convert_ids_to_tokens(ids)
            for w, ids in zip(words, encs)
            if unk_id in ids
        }

    @torch.no_grad()
    def embed(
        self,
        segments: list[Segment],
        batch_size: int = 32,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Stream ``(word, occurrence_vector)`` for every word-of-interest span.

        Each occurrence vector is the mean of the final-layer hidden states of the
        tokens overlapping that occurrence's character span. Only one batch's
        hidden states live in memory at a time — this yields the occurrences and
        holds nothing, so the caller decides how to fold them (e.g. a streaming
        max-pool) rather than accumulating every vector here.
        """
        for start in range(0, len(segments), batch_size):
            yield from self._embed_batch(segments[start: start + batch_size])

    def _embed_batch(
        self,
        batch: list[Segment],
    ) -> Iterator[tuple[str, np.ndarray]]:
        enc = self.tokenizer(
            [r.text for r in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")  # (B, T, 2), stays on CPU
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state.cpu().numpy()  # (B, T, H)

        for i, rec in enumerate(batch):
            offs = offsets[i].tolist()
            for span in rec.spans:
                tok_idx = [
                    j
                    for j, (a, b) in enumerate(offs)
                    if b > a and a < span.end and b > span.start  # real, overlapping
                ]
                if not tok_idx:
                    continue  # span fell outside the 512-token truncation window
                yield span.word, hidden[i, tok_idx].mean(axis=0)
