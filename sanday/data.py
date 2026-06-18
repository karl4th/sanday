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
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self._resolve_manifest(Path(manifest))
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.text_column = text_column
        self.table = pd.read_csv(self.manifest_path, sep="\t")
        self.table = self.table.dropna(subset=[self.audio_column, self.text_column]).copy()
        self.table["_normalized_text"] = self.table[self.text_column].map(lambda value: normalize_text(str(value)))
        self.table = self.table[self.table["_normalized_text"].str.len() > 0].reset_index(drop=True)

    def _resolve_manifest(self, manifest: Path) -> Path:
        candidate = self.root / manifest
        if candidate.exists():
            return candidate
        matches = list(self.root.rglob(manifest.name))
        if not matches:
            raise FileNotFoundError(f"Could not find manifest {manifest} under {self.root}")
        return matches[0]

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
