from __future__ import annotations

import re
from dataclasses import dataclass


_SPACE_RE = re.compile(r"\s+")
_ENGLISH_RE = re.compile(r"[^a-z' ]+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _ENGLISH_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class CharacterVocabulary:
    alphabet: str = "abcdefghijklmnopqrstuvwxyz '"
    blank: str = "<blank>"

    def __post_init__(self) -> None:
        symbols = [self.blank, *list(self.alphabet)]
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "stoi", {symbol: idx for idx, symbol in enumerate(symbols)})
        object.__setattr__(self, "itos", {idx: symbol for idx, symbol in enumerate(symbols)})

    @property
    def blank_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        return [self.stoi[ch] for ch in normalized if ch in self.stoi and ch != self.blank]

    def decode_ctc(self, token_ids: list[int]) -> str:
        output: list[str] = []
        previous = self.blank_id
        for token_id in token_ids:
            if token_id != self.blank_id and token_id != previous:
                output.append(self.itos[token_id])
            previous = token_id
        return "".join(output).strip()
