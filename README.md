# Sanday

Sanday is an experimental automatic speech recognition research project by Manifestro.
The project studies whether liquid neural networks, specifically Closed-form
Continuous-time networks (CfC), can model speech as a continuous dynamical
system and compete with transformer-based acoustic encoders while using orders
of magnitude fewer parameters.

The core hypothesis is:

> Liquid/CfC networks can provide a competitive acoustic ASR encoder with
> roughly two million parameters, without relying on an external language model.

Sanday intentionally does not include an LM in the decoding path. The goal is to
measure the acoustic model itself, rather than hide weaknesses behind language
model rescoring.

## Dataset

Experiments are run in Google Colab against the Common Voice Kaggle dataset:

```python
import kagglehub

path = kagglehub.dataset_download("prateeknarain/common-voice")
print("Path to dataset files:", path)
```

## Architecture

The baseline follows this shape:

```text
audio waveform
  -> mel spectrogram
  -> lightweight convolutional frontend
  -> CfC recurrent/liquid temporal encoder
  -> linear projection
  -> CTC loss / CTC decoding
```

The intended model scale is approximately 2M trainable parameters.

The v2 experimental model keeps CfC as the main temporal component but adds a
stronger acoustic stack:

- Conformer-style multi-scale depthwise temporal convolutions
- adaptive gated time downsampling
- CfC recurrent/liquid encoder layers
- one shallow transformer context layer after CfC
- CTC projection head

## Repository Layout

```text
configs/
  sanday_cfc_2m.yaml       Experiment configuration
  sanday_hybrid_v2.yaml    Hybrid CfC/Transformer experiment configuration
  sanday_sliding_ncp.yaml  Sliding-window CNN and AutoNCP CfC configuration
docs/
  abstract.md              Research abstract
  architecture.md          Model design notes
  methodology.md           Dataset and training methodology
  experiments.md           Experiment plan and comparison targets
references/
  project_notes.md         Project facts and open questions
sanday/
  features.py              Mel spectrogram extraction
  model.py                 ASR model variants and model factory
  text.py                  Character vocabulary and CTC helpers
  data.py                  Common Voice dataset loader skeleton
scripts/
  train.py                 Colab-oriented training entry point
  evaluate.py              Evaluation entry point skeleton
notebooks/
  sanday_colab_experiment.ipynb
```

## Install

The target environment is Google Colab. A minimal setup is:

```bash
pip install torch torchaudio ncps kagglehub pandas jiwer pyyaml tqdm
```

## Training

Recommended Colab entry point:

```text
notebooks/sanday_colab_experiment.ipynb
```

The notebook runs a short smoke test first, then launches the full training run
and prints `error.log` automatically if a subprocess fails.

```bash
python scripts/train.py --config configs/sanday_cfc_2m.yaml
```

Hybrid v2:

```bash
python scripts/train.py --config configs/sanday_hybrid_v2.yaml
```

Sliding-window AutoNCP:

```bash
python scripts/train.py --config configs/sanday_sliding_ncp.yaml
```

The default configuration uses seed `42` and saves both per-epoch
checkpoints and the best validation-WER checkpoint:

```text
results/<run-id>/checkpoints/sanday_best.pt
```

Each run also writes small readable artifacts:

```text
results/<run-id>/
  config.yaml
  environment.json
  metrics.csv
  metrics.jsonl
  summary.json
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/sanday_cfc_2m.yaml --checkpoint results/<run-id>/checkpoints/sanday_best.pt
```

Evaluation writes:

```text
results/<eval-run-id>/
  config.yaml
  environment.json
  evaluation.json
  predictions.csv
```

## Research Positioning

Sanday should be compared against compact acoustic ASR baselines and larger
transformer/conformer systems using:

- parameter count
- WER and CER
- real-time factor / decoding latency
- training stability
- performance without LM rescoring

The claim should be stated conservatively: Sanday investigates whether a very
small liquid/CfC acoustic model can remain competitive in constrained ASR
settings, not that it universally replaces transformer ASR systems.

## Baseline and Target

The baseline reference is approximately 68% WER at around 2M parameters. The
next target is a reproducible seed-42 experiment reaching the 40-50% WER range
without an external language model.
