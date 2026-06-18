from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kagglehub
import torch
import yaml
from jiwer import cer, wer
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sanday CfC CTC ASR model")
    parser.add_argument("--config", default="configs/sanday_cfc_2m.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-valid-items", type=int, default=None)
    return parser.parse_args()


def prepare_run(args: argparse.Namespace) -> tuple[dict, Path, str]:
    from sanday.reporting import default_run_dir

    config = load_config(args.config)
    variant = config["model"].get("variant", "cfc")
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.config, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    return config, run_dir, variant


def decode_batch(model, features, vocab, waveforms, waveform_lengths, device):
    mel = features(waveforms.to(device))
    input_lengths = features.output_lengths(waveform_lengths.to(device))
    logits, output_lengths = model(mel, input_lengths)
    token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    lengths = output_lengths.detach().cpu().tolist()
    return [vocab.decode_ctc(ids[:length]) for ids, length in zip(token_ids, lengths)]


def select_ctc_targets(
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pieces = torch.split(labels, label_lengths.detach().cpu().tolist())
    kept_pieces = [piece for piece, keep in zip(pieces, keep_mask.detach().cpu().tolist()) if keep]
    if not kept_pieces:
        return labels.new_empty((0,)), label_lengths.new_empty((0,))
    return torch.cat(kept_pieces), label_lengths[keep_mask]


def evaluate(model, feature_extractor, loader, vocab, device, max_batches: int | None = None) -> tuple[float, float]:
    from sanday.text import normalize_text

    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="valid", leave=False), start=1):
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
            if max_batches is not None and batch_index >= max_batches:
                break
    model.train()
    return wer(references, predictions), cer(references, predictions)


def main() -> None:
    from sanday.data import CommonVoiceDataset, collate_common_voice
    from sanday.features import LogMelSpectrogram, SpecAugment
    from sanday.model import build_sanday_model, count_parameters
    from sanday.reporting import append_csv, append_jsonl, collect_environment, write_config_snapshot, write_json
    from sanday.reproducibility import seed_everything
    from sanday.text import CharacterVocabulary

    args = parse_args()
    config, run_dir, variant = prepare_run(args)
    seed_everything(config["project"].get("seed", 42), config["project"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        max_items=args.max_train_items,
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
        max_items=args.max_valid_items,
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
    epochs = args.epochs or config["training"]["epochs"]
    if config["training"].get("scheduler") == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["training"]["learning_rate"],
            epochs=epochs,
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
    for epoch in range(epochs):
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
                if not torch.isfinite(logits).all():
                    print(
                        "Skipping batch: non-finite logits "
                        f"finite_ratio={torch.isfinite(logits).float().mean().item():.4f} "
                        f"min={torch.nan_to_num(logits.detach()).min().item():.4f} "
                        f"max={torch.nan_to_num(logits.detach()).max().item():.4f}"
                    )
                    if args.max_train_batches is not None and step >= args.max_train_batches:
                        break
                    continue
                keep_mask = output_lengths >= label_lengths
                if not bool(keep_mask.any()):
                    print(
                        "Skipping batch: all CTC targets are longer than model outputs "
                        f"output_lengths={output_lengths.detach().cpu().tolist()} "
                        f"label_lengths={label_lengths.detach().cpu().tolist()}"
                    )
                    if args.max_train_batches is not None and step >= args.max_train_batches:
                        break
                    continue
                if not bool(keep_mask.all()):
                    logits = logits[keep_mask]
                    output_lengths = output_lengths[keep_mask]
                    labels, label_lengths = select_ctc_targets(labels, label_lengths, keep_mask)
                log_probs = logits.float().log_softmax(dim=-1).clamp(min=-30).transpose(0, 1)
                loss = criterion(log_probs, labels, output_lengths, label_lengths)

            if not torch.isfinite(loss):
                print(
                    "Skipping batch: non-finite CTC loss "
                    f"loss={loss.item()} "
                    f"output_lengths={output_lengths.detach().cpu().tolist()} "
                    f"label_lengths={label_lengths.detach().cpu().tolist()}"
                )
                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer_step_was_skipped = scaler.is_enabled() and scale_after < scale_before
            if scheduler is not None and not optimizer_step_was_skipped:
                scheduler.step()

            train_loss_sum += float(loss.item())
            train_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

        valid_wer, valid_cer = evaluate(model, features, valid_loader, vocab, device, args.max_valid_batches)
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
    try:
        main()
    except Exception:
        args = parse_args()
        try:
            _, run_dir, _ = prepare_run(args)
            error = traceback.format_exc()
            (run_dir / "error.log").write_text(error, encoding="utf-8")
            print(f"\nTraining failed. Full traceback written to: {run_dir / 'error.log'}")
            print(error)
        finally:
            raise
