"""Sliding-window token buffer for per-sentence-chunk policy evaluation.

Accumulates LLM output tokens into complete sentence-chunks delimited by
sentence-ending punctuation (period, exclamation mark, or question mark)
followed by whitespace. Emitted chunks are passed to the output validator
and GroundednessAuditor before being forwarded to the SSE client.

All buffer operations are synchronous pure-string ops (< 1 ms); any
CPU-bound policy checks on assembled chunks are dispatched via
asyncio.to_thread in the calling layer.

Requirements: 4.5, 4.12
"""

from __future__ import annotations

import re

# Sentence boundary: lookbehind for .  !  ? followed by one or more whitespace chars.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SlidingWindow:
    """Accumulates LLM tokens and emits complete sentence-chunks.

    Usage::

        window = SlidingWindow()
        for token in token_stream:
            for chunk in window.push(token):
                await validate_and_emit(chunk)
        for chunk in window.flush_remaining():
            await validate_and_emit(chunk)
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, token: str) -> list[str]:
        """Append *token* to the internal buffer and return any complete sentence-chunks.

        A chunk is considered complete when a sentence-ending punctuation mark
        (.  !  ?) is followed by whitespace.  The returned list may be empty
        when no boundary is detected yet.

        Args:
            token: A single token string emitted by the LLM.

        Returns:
            A (possibly empty) list of stripped sentence-chunk strings.
        """
        self._buffer += token
        return self._flush()

    def flush_remaining(self) -> list[str]:
        """Flush any buffered content as a single final chunk.

        Called at end-of-stream to emit whatever remains in the buffer
        that did not end with a recognised sentence boundary.

        Returns:
            A list containing the remaining content (stripped) if non-empty,
            otherwise an empty list.
        """
        if self._buffer.strip():
            chunk = self._buffer
            self._buffer = ""
            return [chunk.strip()]
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> list[str]:
        """Split off all complete sentence-chunks from the front of the buffer.

        Iterates until no more sentence boundaries remain, stripping each
        chunk and leaving any trailing incomplete sentence in ``self._buffer``.

        Returns:
            Ordered list of complete sentence-chunk strings.
        """
        chunks: list[str] = []
        while True:
            match = _SENTENCE_BOUNDARY.search(self._buffer)
            if not match:
                break
            end = match.end()
            chunks.append(self._buffer[:end].strip())
            self._buffer = self._buffer[end:]
        return chunks
