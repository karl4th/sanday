from __future__ import annotations

import argparse
import time
from pathlib import Path

import kagglehub
import torch
import yaml
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm

from sanday.data import CommonVoiceDataset, collate_common_voice
from sanday.features import LogMelSpectrogram, SpecAugment
from sanday.model import build_sanday_model, count_parameters
from sanday.reporting import (
    append_csv,
    append_jsonl,
    collect_environment,
    default_run_dir,
    write_config_snapshot,
    write_json,
)
from sanday.reproducibility import seed_everything
from sanday.text import CharacterVocabulary
from sanday.text import normalize_text


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def decode_batch(model, features, vocab, waveforms, waveform_lengths, device):
    mel = features(waveforms.to(device))
    input_lengths = features.output_lengths(waveform_lengths.to(device))
    logits, output_lengths = model(mel, input_lengths)
    token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    lengths = output_lengths.detach().cpu().tolist()
    return [vocab.decode_ctc(ids[:length]) for ids, length in zip(token_ids, lengths)]


def evaluate(model, feature_extractor, loader, vocab, device) -> tuple[float, float]:
    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False):
            predictions.extend(
                decode_batch(
                    model,
                    feature_extractor,
                    vocab,
                    batch["waveforms"],
                    batch["waveform_lengths"],
                    device,
                )
            )
            references.extend(normalize_text(text) for text in batch["texts"])
    model.train()
    return wer(references, predictions), cer(references, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sanday CfC CTC ASR model")
    parser.add_argument("--config", default="configs/sanday_cfc_2m.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["project"].get("seed", 42), config["project"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = config["model"].get("variant", "cfc")
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.config, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(run_dir / "config.yaml", config)
    write_json(run_dir / "environment.json", collect_environment())

    dataset_root = kagglehub.dataset_download(config["data"]["dataset"])
    vocab = CharacterVocabulary(config["vocab"]["alphabet"])
    features = LogMelSpectrogram(**config["features"], sample_rate=config["data"]["sample_rate"]).to(device)
    augment = None
    if config.get("augmentation", {}).get("specaugment", False):
        augment = SpecAugment(
            time_masks=config["augmentation"]["time_masks"],
            time_width=config["augmentation"]["time_width"],
            freq_masks=config["augmentation"]["freq_masks"],
            freq_width=config["augmentation"]["freq_width"],
        ).to(device)
    model = build_sanday_model(config, len(vocab)).to(device)

    print(f"Dataset root: {dataset_root}")
    print(f"Run dir: {run_dir}")
    print(f"Seed: {config['project'].get('seed', 42)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    train_dataset = CommonVoiceDataset(
        root=dataset_root,
        manifest=config["data"]["train_manifest"],
        vocab=vocab,
        sample_rate=config["data"]["sample_rate"],
        audio_column=config["data"]["audio_column"],
        text_column=config["data"]["text_column"],
        split="train",
        split_seed=config["project"].get("seed", 42),
        train_ratio=config["data"].get("train_ratio", 0.9),
        valid_ratio=config["data"].get("valid_ratio", 0.05),
    )
    valid_dataset = CommonVoiceDataset(
        root=dataset_root,
        manifest=config["data"]["valid_manifest"],
        vocab=vocab,
        sample_rate=config["data"]["sample_rate"],
        audio_column=config["data"]["audio_column"],
        text_column=config["data"]["text_column"],
        split="valid",
        split_seed=config["project"].get("seed", 42),
        train_ratio=config["data"].get("train_ratio", 0.9),
        valid_ratio=config["data"].get("valid_ratio", 0.05),
    )
    generator = torch.Generator()
    generator.manual_seed(config["project"].get("seed", 42))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_common_voice,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_common_voice,
    )

    criterion = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = None
    if config["training"].get("scheduler") == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["training"]["learning_rate"],
            epochs=config["training"]["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=config["training"].get("pct_start", 0.1),
        )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("mixed_precision", False) and device.type == "cuda")
    best_wer = float("inf")
    best_epoch = 0
    started_at = time.time()
    model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_started_at = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        train_loss_sum = 0.0
        train_steps = 0
        for step, batch in enumerate(progress, start=1):
            waveforms = batch["waveforms"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                mel = features(waveforms)
                input_lengths = features.output_lengths(waveform_lengths)
                if augment is not None:
                    mel = augment(mel)
                logits, output_lengths = model(mel, input_lengths)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, labels, output_lengths, label_lengths)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += float(loss.item())
            train_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_wer, valid_cer = evaluate(model, features, valid_loader, vocab, device)
        train_loss = train_loss_sum / max(train_steps, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_wer": valid_wer,
            "valid_cer": valid_cer,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started_at,
        }
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"valid_wer={valid_wer:.4f} valid_cer={valid_cer:.4f}"
        )
        append_jsonl(run_dir / "metrics.jsonl", epoch_metrics)
        append_csv(run_dir / "metrics.csv", epoch_metrics)
        checkpoint_path = checkpoint_dir / f"sanday_epoch_{epoch + 1}.pt"
        payload = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "valid_wer": valid_wer,
            "valid_cer": valid_cer,
            "parameters": count_parameters(model),
            "run_dir": str(run_dir),
        }
        torch.save(payload, checkpoint_path)
        if valid_wer < best_wer:
            best_wer = valid_wer
            best_epoch = epoch + 1
            torch.save(payload, checkpoint_dir / "sanday_best.pt")

        write_json(
            run_dir / "summary.json",
            {
                "config": str(args.config),
                "variant": variant,
                "seed": config["project"].get("seed", 42),
                "parameters": count_parameters(model),
                "best_epoch": best_epoch,
                "best_valid_wer": best_wer,
                "last_valid_wer": valid_wer,
                "last_valid_cer": valid_cer,
                "epochs_completed": epoch + 1,
                "checkpoint_dir": str(checkpoint_dir),
                "best_checkpoint": str(checkpoint_dir / "sanday_best.pt"),
                "elapsed_seconds": time.time() - started_at,
            },
        )


if __name__ == "__main__":
    main()
