from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sanday.text import CharacterVocabulary, normalize_text


class CommonVoiceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        vocab: CharacterVocabulary,
        sample_rate: int = 16_000,
        audio_column: str = "path",
        text_column: str = "sentence",
        split: str | None = None,
        split_seed: int = 42,
        train_ratio: float = 0.9,
        valid_ratio: float = 0.05,
        max_items: int | None = None,
    ) -> None:
        self.root = Path(root)
        requested_manifest = Path(manifest)
        self.manifest_path = self._resolve_manifest(requested_manifest)
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.text_column = text_column
        self.table = pd.read_csv(self.manifest_path, sep="\t")
        self.table = self.table.dropna(subset=[self.audio_column, self.text_column]).copy()
        self.table["_normalized_text"] = self.table[self.text_column].map(lambda value: normalize_text(str(value)))
        self.table = self.table[self.table["_normalized_text"].str.len() > 0].reset_index(drop=True)
        inferred_split = split or self._infer_split(requested_manifest)
        if requested_manifest.name != self.manifest_path.name and self.manifest_path.name == "validated.tsv":
            self.table = self._split_table(inferred_split, split_seed, train_ratio, valid_ratio)
        if max_items is not None:
            self.table = self.table.head(max_items).reset_index(drop=True)
        print(
            f"CommonVoiceDataset split={inferred_split} manifest={self.manifest_path} "
            f"rows={len(self.table)}"
        )

    def _resolve_manifest(self, manifest: Path) -> Path:
        candidate = self.root / manifest
        if candidate.exists():
            return candidate
        matches = list(self.root.rglob(manifest.name))
        if not matches and manifest.name in {"train.tsv", "dev.tsv", "valid.tsv", "test.tsv"}:
            matches = self._validated_manifest_matches()
        if not matches:
            available = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*.tsv"))[:30]
            raise FileNotFoundError(
                f"Could not find manifest {manifest} under {self.root}. "
                f"Available TSV examples: {available}"
            )
        return matches[0]

    def _validated_manifest_matches(self) -> list[Path]:
        matches = [
            path
            for path in self.root.rglob("validated.tsv")
            if "/English/" in path.as_posix() or "/en/" in path.as_posix()
        ]
        return sorted(matches, key=lambda path: (len(path.parts), str(path)))

    @staticmethod
    def _infer_split(manifest: Path) -> str:
        name = manifest.name
        if name == "dev.tsv" or name == "valid.tsv":
            return "valid"
        if name == "test.tsv":
            return "test"
        return "train"

    def _split_table(
        self,
        split: str,
        split_seed: int,
        train_ratio: float,
        valid_ratio: float,
    ) -> pd.DataFrame:
        table = self.table.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)
        train_end = int(len(table) * train_ratio)
        valid_end = train_end + int(len(table) * valid_ratio)
        if split == "train":
            return table.iloc[:train_end].reset_index(drop=True)
        if split == "valid":
            return table.iloc[train_end:valid_end].reset_index(drop=True)
        if split == "test":
            return table.iloc[valid_end:].reset_index(drop=True)
        raise ValueError(f"Unknown split: {split}")

    def _resolve_audio(self, value: str) -> Path:
        relative = Path(value)
        candidates = [
            self.root / relative,
            self.manifest_path.parent / relative,
            self.manifest_path.parent / "clips" / relative.name,
            self.root / "clips" / relative.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = list(self.root.rglob(relative.name))
        if not matches:
            raise FileNotFoundError(f"Could not find audio file {value} under {self.root}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        audio_path = self._resolve_audio(str(row[self.audio_column]))
        waveform, source_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)

        transcript = str(row["_normalized_text"])
        labels = torch.tensor(self.vocab.encode(transcript), dtype=torch.long)
        return {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "labels": labels,
            "label_length": torch.tensor(labels.numel(), dtype=torch.long),
            "text": transcript,
        }


def collate_common_voice(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = pad_sequence([item["waveform"] for item in batch], batch_first=True)
    labels = torch.cat([item["labels"] for item in batch])
    return {
        "waveforms": waveforms,
        "waveform_lengths": torch.stack([item["waveform_length"] for item in batch]),
        "labels": labels,
        "label_lengths": torch.stack([item["label_length"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }
