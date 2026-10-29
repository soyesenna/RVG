"""Byte-bounded passage spans. The corpus snapshot store of the SciFact retrieval research stayed
in the research repository."""

from __future__ import annotations

from ..errors import ContractError


def split_spans(text: str, max_bytes: int, overlap: int = 128):
    if max_bytes < 512:
        raise ContractError("chunk_byte_budget_too_small")
    start = 0
    while start < len(text):
        piece = text[start:].encode()[:max_bytes].decode("utf-8", errors="ignore")
        end = start + len(piece)
        if end < len(text):
            boundary = max(piece.rfind(". "), piece.rfind("\n"))
            if boundary > len(piece) // 2:
                end = start + boundary + 1
        yield start, end
        if end == len(text):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 4))
