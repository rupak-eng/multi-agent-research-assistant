"""Token counting.

Prefers tiktoken (cl100k_base) when importable; otherwise falls back to a
char-based estimator (~4 chars/token for English). The active method is
recorded on every run snapshot so numbers are never mislabeled.
"""

from __future__ import annotations


class TokenCounter:
    def __init__(self) -> None:
        self.method = "char-estimate"
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding("cl100k_base")
            self.method = "tiktoken"
        except Exception:  # noqa: BLE001 - tiktoken optional
            self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        # ~4 chars per token is the standard rough estimator for English.
        return max(1, len(text) // 4)

    def count_messages(self, system: str, user: str) -> int:
        # +8 accounts for chat framing overhead (conservative, documented).
        return self.count(system) + self.count(user) + 8
