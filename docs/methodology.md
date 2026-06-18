# Methodology

## Dataset

Experiments use Common Voice from Kaggle:

```python
import kagglehub

path = kagglehub.dataset_download("prateeknarain/common-voice")
print("Path to dataset files:", path)
```

The experiment targets English Common Voice. The code assumes a Common
Voice-style metadata table containing audio paths and transcripts.

## Preprocessing

1. Load audio.
2. Resample to 16 kHz.
3. Convert to mono.
4. Compute mel spectrogram features.
5. Normalize or log-compress mel values.
6. Encode transcripts as character labels.

## Training Objective

Sanday uses CTC loss. This avoids requiring frame-level alignments and keeps the
experiment focused on acoustic sequence modeling.

## Decoding

The primary decoder is greedy CTC decoding:

1. Take argmax over vocabulary at each timestep.
2. Collapse repeated symbols.
3. Remove blank tokens.
4. Convert character IDs back to text.

Beam search and language model rescoring are intentionally excluded from the
main comparison.

## Metrics

The intended metrics are:

- word error rate (WER)
- character error rate (CER)
- trainable parameter count
- real-time factor or inference latency
- memory usage during inference

WER and CER are the primary quality metrics. Parameter count and latency are
used to evaluate the efficiency claim.
