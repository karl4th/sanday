from __future__ import annotations

import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    """Mel spectrogram frontend used by Sanday."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = 8_000.0,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.hop_length = hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        features = self.mel(waveform)
        features = self.amplitude_to_db(features)
        features = features.transpose(-1, -2)
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (features - mean) / std

    def output_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(waveform_lengths, self.hop_length, rounding_mode="floor") + 1


class SpecAugment(torch.nn.Module):
    def __init__(self, time_masks: int = 2, time_width: int = 40, freq_masks: int = 2, freq_width: int = 12) -> None:
        super().__init__()
        self.time_masks = torch.nn.ModuleList(
            [torchaudio.transforms.TimeMasking(time_mask_param=time_width) for _ in range(time_masks)]
        )
        self.freq_masks = torch.nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_width) for _ in range(freq_masks)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # torchaudio masking expects [batch, freq, time].
        x = features.transpose(1, 2)
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x.transpose(1, 2)
