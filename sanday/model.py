from __future__ import annotations

import torch
from ncps.torch import CfC


class SandayCfCCTC(torch.nn.Module):
    """Compact CfC acoustic model trained with CTC."""

    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        conv_channels: int = 96,
        conv_stride: int = 2,
        cfc_hidden_size: int = 256,
        cfc_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, conv_channels, kernel_size=5, stride=conv_stride, padding=2),
            torch.nn.GroupNorm(8, conv_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.encoder_layers = torch.nn.ModuleList()
        input_size = conv_channels
        for _ in range(cfc_layers):
            self.encoder_layers.append(
                CfC(input_size, cfc_hidden_size, return_sequences=True, batch_first=True)
            )
            input_size = cfc_hidden_size
        self.norm = torch.nn.LayerNorm(cfc_hidden_size)
        self.classifier = torch.nn.Linear(cfc_hidden_size, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # features: [batch, time, n_mels]
        x = features.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)

        if input_lengths is not None:
            stride = self.frontend[0].stride[0]
            input_lengths = torch.div(input_lengths + stride - 1, stride, rounding_mode="floor")

        for layer in self.encoder_layers:
            output = layer(x)
            x = output[0] if isinstance(output, tuple) else output

        x = self.norm(x)
        logits = self.classifier(x)
        return logits, input_lengths


class DepthwiseSeparableConv1d(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            torch.nn.Conv1d(channels, channels, kernel_size=1),
            torch.nn.GroupNorm(8, channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleConvFrontend(torch.nn.Module):
    """Conformer-style multi-scale temporal convolution frontend."""

    def __init__(self, n_mels: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, channels),
            torch.nn.SiLU(),
        )
        self.branches = torch.nn.ModuleList(
            [
                DepthwiseSeparableConv1d(channels, kernel_size=3, dropout=dropout),
                DepthwiseSeparableConv1d(channels, kernel_size=7, dropout=dropout),
                DepthwiseSeparableConv1d(channels, kernel_size=15, dropout=dropout),
            ]
        )
        self.mix = torch.nn.Sequential(
            torch.nn.Conv1d(channels * len(self.branches), channels, kernel_size=1),
            torch.nn.GroupNorm(8, channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.input_projection(x)
        branches = [branch(x) for branch in self.branches]
        return self.mix(torch.cat(branches, dim=1)).transpose(1, 2)


class AdaptiveTimeDownsampler(torch.nn.Module):
    """Gated temporal downsampling with learned keep/merge behavior."""

    def __init__(self, channels: int, stride: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.content = torch.nn.Conv1d(channels, channels, kernel_size=5, stride=stride, padding=2)
        self.gate = torch.nn.Conv1d(channels, channels, kernel_size=5, stride=stride, padding=2)
        self.norm = torch.nn.LayerNorm(channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.stride == 1:
            return x, lengths

        x_t = x.transpose(1, 2)
        content = self.content(x_t)
        gate = torch.sigmoid(self.gate(x_t))
        y = (content * gate).transpose(1, 2)
        y = self.dropout(self.norm(y))

        if lengths is not None:
            lengths = torch.div(lengths + self.stride - 1, self.stride, rounding_mode="floor")
        return y, lengths


class SandayHybridCfCTransformerCTC(torch.nn.Module):
    """Hybrid Sanday ASR encoder: multi-scale conv, adaptive downsampling, CfC, transformer, CTC."""

    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        conv_channels: int = 160,
        downsample_stride: int = 2,
        cfc_hidden_size: int = 320,
        cfc_layers: int = 3,
        transformer_heads: int = 4,
        transformer_ffn: int = 768,
        transformer_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = MultiScaleConvFrontend(n_mels, conv_channels, dropout)
        self.downsampler = AdaptiveTimeDownsampler(conv_channels, downsample_stride, dropout)

        self.cfc_layers = torch.nn.ModuleList()
        input_size = conv_channels
        for _ in range(cfc_layers):
            self.cfc_layers.append(
                CfC(input_size, cfc_hidden_size, return_sequences=True, batch_first=True)
            )
            input_size = cfc_hidden_size

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=cfc_hidden_size,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = torch.nn.LayerNorm(cfc_hidden_size)
        self.classifier = torch.nn.Linear(cfc_hidden_size, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.frontend(features)
        x, input_lengths = self.downsampler(x, input_lengths)

        for layer in self.cfc_layers:
            output = layer(x)
            x = output[0] if isinstance(output, tuple) else output

        padding_mask = None
        if input_lengths is not None:
            steps = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            padding_mask = steps >= input_lengths.unsqueeze(1)

        x = self.context(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.classifier(x), input_lengths


def build_sanday_model(config: dict, vocab_size: int) -> torch.nn.Module:
    model_config = {
        key: value
        for key, value in config["model"].items()
        if key not in {"target_parameters", "variant"}
    }
    variant = config["model"].get("variant", "cfc")
    if variant == "cfc":
        return SandayCfCCTC(
            n_mels=config["features"]["n_mels"],
            vocab_size=vocab_size,
            **model_config,
        )
    if variant == "hybrid_v2":
        return SandayHybridCfCTransformerCTC(
            n_mels=config["features"]["n_mels"],
            vocab_size=vocab_size,
            **model_config,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
