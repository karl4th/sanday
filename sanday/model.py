from __future__ import annotations

import math

import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP, FullyConnected


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


class SlidingWindowCNN(torch.nn.Module):
    """Apply a small 2D CNN to overlapping log-mel windows."""

    def __init__(
        self,
        n_mels: int = 80,
        window_frames: int = 32,
        window_stride: int = 8,
        out_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_frames = window_frames
        self.window_stride = window_stride

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
        )

        h_out = math.ceil(window_frames / 4)
        w_out = math.ceil(n_mels / 4)
        flat_dim = 128 * h_out * w_out
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(flat_dim, out_dim, bias=False),
            torch.nn.LayerNorm(out_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, time, n_mels = features.shape
        if time < self.window_frames:
            features = torch.nn.functional.pad(features, (0, 0, 0, self.window_frames - time))
            time = self.window_frames

        windows = features.unfold(1, self.window_frames, self.window_stride)
        steps = windows.size(1)
        x = windows.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch * steps, 1, self.window_frames, n_mels)
        x = self.cnn(x)
        x = self.proj(x.flatten(1))
        return x.view(batch, steps, -1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.clamp(min=self.window_frames)
        lengths = torch.div(lengths - self.window_frames, self.window_stride, rounding_mode="floor") + 1
        return lengths.clamp(min=1)


class CfCResidualLayer(torch.nn.Module):
    """CfC layer with optional AutoNCP wiring, residual projection, norm, and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        mode: str = "default",
        use_ncp: bool = True,
        ncp_ratio: int = 2,
        mixed_memory: bool = False,
    ) -> None:
        super().__init__()
        if use_ncp:
            wiring = AutoNCP(hidden_dim * ncp_ratio, hidden_dim)
        else:
            wiring = FullyConnected(hidden_dim, hidden_dim)

        self.cfc = CfC(
            in_dim,
            wiring,
            mode=mode,
            mixed_memory=mixed_memory,
            batch_first=True,
            return_sequences=True,
        )
        self.skip = torch.nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.skip(x)
        output = self.cfc(x, hx=hx)
        if isinstance(output, tuple):
            out, hx = output
        else:
            out, hx = output, None
        out = self.dropout(out)
        return self.norm(out + residual), hx


class CfCResidualEncoder(torch.nn.Module):
    """Stack of residual CfC layers. Hidden states can be reused for streaming."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        mode: str = "default",
        use_ncp: bool = True,
        ncp_ratio: int = 2,
        mixed_memory: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim, *([hidden_dim] * num_layers)]
        self.layers = torch.nn.ModuleList(
            [
                CfCResidualLayer(
                    in_dim=dims[index],
                    hidden_dim=dims[index + 1],
                    dropout=dropout,
                    mode=mode,
                    use_ncp=use_ncp,
                    ncp_ratio=ncp_ratio,
                    mixed_memory=mixed_memory,
                )
                for index in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hx_list: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        if hx_list is None:
            hx_list = [None] * len(self.layers)

        new_hx = []
        for layer, hx in zip(self.layers, hx_list):
            x, next_hx = layer(x, hx)
            new_hx.append(next_hx)
        return x, new_hx


class SandaySlidingNCPCTC(torch.nn.Module):
    """Sliding-window CNN plus residual AutoNCP CfC encoder for CTC ASR."""

    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        window_frames: int = 32,
        window_stride: int = 8,
        cnn_dim: int = 256,
        hidden_dim: int = 256,
        cfc_layers: int = 4,
        dropout: float = 0.1,
        ctc_dropout: float = 0.1,
        mode: str = "default",
        use_ncp: bool = True,
        ncp_ratio: int = 2,
        mixed_memory: bool = False,
    ) -> None:
        super().__init__()
        self.window = SlidingWindowCNN(
            n_mels=n_mels,
            window_frames=window_frames,
            window_stride=window_stride,
            out_dim=cnn_dim,
            dropout=dropout,
        )
        self.encoder = CfCResidualEncoder(
            input_dim=cnn_dim,
            hidden_dim=hidden_dim,
            num_layers=cfc_layers,
            dropout=dropout,
            mode=mode,
            use_ncp=use_ncp,
            ncp_ratio=ncp_ratio,
            mixed_memory=mixed_memory,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(ctc_dropout),
            torch.nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        hx_list: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.window(features)
        if input_lengths is not None:
            input_lengths = self.window.output_lengths(input_lengths)

        x, _ = self.encoder(x, hx_list)
        return self.head(x), input_lengths


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
    if variant == "sliding_ncp":
        return SandaySlidingNCPCTC(
            n_mels=config["features"]["n_mels"],
            vocab_size=vocab_size,
            **model_config,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
