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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
