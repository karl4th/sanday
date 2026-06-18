from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
import torch
import yaml
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm

from sanday.data import CommonVoiceDataset, collate_common_voice
from sanday.features import LogMelSpectrogram
from sanday.model import SandayCfCCTC
from sanday.model_v2 import SandayHybridCfCTransformerCTC
from sanday.reproducibility import seed_everything
from sanday.text import CharacterVocabulary, normalize_text


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict, vocab_size: int) -> torch.nn.Module:
    model_config = {
        key: value
        for key, value in config["model"].items()
        if key not in {"target_parameters", "variant"}
    }
    variant = config["model"].get("variant", "cfc")
    if variant == "hybrid_v2":
        return SandayHybridCfCTransformerCTC(
            n_mels=config["features"]["n_mels"],
            vocab_size=vocab_size,
            **model_config,
        )
    if variant == "cfc":
        return SandayCfCCTC(
            n_mels=config["features"]["n_mels"],
            vocab_size=vocab_size,
            **model_config,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sanday CfC CTC ASR model")
    parser.add_argument("--config", default="configs/sanday_cfc_2m.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["project"].get("seed", 42), config["project"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = kagglehub.dataset_download(config["data"]["dataset"])
    vocab = CharacterVocabulary(config["vocab"]["alphabet"])
    features = LogMelSpectrogram(**config["features"], sample_rate=config["data"]["sample_rate"]).to(device)
    model = build_model(config, len(vocab)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_dataset = CommonVoiceDataset(
        root=dataset_root,
        manifest=config["data"]["test_manifest"],
        vocab=vocab,
        sample_rate=config["data"]["sample_rate"],
        audio_column=config["data"]["audio_column"],
        text_column=config["data"]["text_column"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_common_voice,
    )

    references: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluate"):
            mel = features(batch["waveforms"].to(device))
            input_lengths = features.output_lengths(batch["waveform_lengths"].to(device))
            logits, output_lengths = model(mel, input_lengths)
            token_ids = logits.argmax(dim=-1).cpu().tolist()
            lengths = output_lengths.cpu().tolist()
            predictions.extend(vocab.decode_ctc(ids[:length]) for ids, length in zip(token_ids, lengths))
            references.extend(normalize_text(text) for text in batch["texts"])

    print(f"WER: {wer(references, predictions):.4f}")
    print(f"CER: {cer(references, predictions):.4f}")


if __name__ == "__main__":
    main()
