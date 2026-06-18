from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

import kagglehub
import torch
import yaml
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm

from sanday.data import CommonVoiceDataset, collate_common_voice
from sanday.features import LogMelSpectrogram
from sanday.model import build_sanday_model
from sanday.reporting import collect_environment, default_run_dir, write_config_snapshot, write_json
from sanday.reproducibility import seed_everything
from sanday.text import CharacterVocabulary, normalize_text


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Sanday CfC CTC ASR model")
    parser.add_argument("--config", default="configs/sanday_cfc_2m.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def prepare_run(args: argparse.Namespace) -> tuple[dict, Path, str]:
    config = load_config(args.config)
    variant = config["model"].get("variant", "cfc")
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.config, f"{variant}_eval")
    run_dir.mkdir(parents=True, exist_ok=True)
    return config, run_dir, variant


def main() -> None:
    args = parse_args()
    config, run_dir, variant = prepare_run(args)
    seed_everything(config["project"].get("seed", 42), config["project"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_config_snapshot(run_dir / "config.yaml", config)
    write_json(run_dir / "environment.json", collect_environment())

    dataset_root = kagglehub.dataset_download(config["data"]["dataset"])
    vocab = CharacterVocabulary(config["vocab"]["alphabet"])
    features = LogMelSpectrogram(**config["features"], sample_rate=config["data"]["sample_rate"]).to(device)
    model = build_sanday_model(config, len(vocab)).to(device)
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
        split="test",
        split_seed=config["project"].get("seed", 42),
        train_ratio=config["data"].get("train_ratio", 0.9),
        valid_ratio=config["data"].get("valid_ratio", 0.05),
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
    prediction_rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluate"):
            mel = features(batch["waveforms"].to(device))
            input_lengths = features.output_lengths(batch["waveform_lengths"].to(device))
            logits, output_lengths = model(mel, input_lengths)
            token_ids = logits.argmax(dim=-1).cpu().tolist()
            lengths = output_lengths.cpu().tolist()
            batch_predictions = [vocab.decode_ctc(ids[:length]) for ids, length in zip(token_ids, lengths)]
            batch_references = [normalize_text(text) for text in batch["texts"]]
            predictions.extend(batch_predictions)
            references.extend(batch_references)
            for reference, prediction in zip(batch_references, batch_predictions):
                prediction_rows.append({"reference": reference, "prediction": prediction})

    final_wer = wer(references, predictions)
    final_cer = cer(references, predictions)
    print(f"Run dir: {run_dir}")
    print(f"WER: {final_wer:.4f}")
    print(f"CER: {final_cer:.4f}")

    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference", "prediction"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    write_json(
        run_dir / "evaluation.json",
        {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "variant": variant,
            "seed": config["project"].get("seed", 42),
            "wer": final_wer,
            "cer": final_cer,
            "num_examples": len(references),
            "predictions_csv": str(run_dir / "predictions.csv"),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        args = parse_args()
        try:
            _, run_dir, _ = prepare_run(args)
            error = traceback.format_exc()
            (run_dir / "error.log").write_text(error, encoding="utf-8")
            print(f"\nEvaluation failed. Full traceback written to: {run_dir / 'error.log'}")
            print(error)
        finally:
            raise
