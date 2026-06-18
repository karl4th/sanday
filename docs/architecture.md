# Architecture

Sanday is a CTC acoustic model built around a CfC temporal encoder.

## Signal Path

```text
waveform
  -> mel spectrogram
  -> log compression
  -> convolutional subsampling frontend
  -> CfC encoder
  -> classifier head
  -> CTC logits
```

## Components

### Mel Features

The input representation is a mel spectrogram. The expected configuration is:

- sample rate: 16 kHz
- mel bins: 80
- FFT window: 25 ms
- hop length: 10 ms

### Convolutional Frontend

The frontend reduces local acoustic variation before the recurrent/liquid
encoder. It should remain small enough that the CfC remains the main modeling
component.

### CfC Encoder

The CfC layer models the mel feature stream as continuous-time dynamics. This
is the main distinction from transformer ASR encoders: Sanday does not rely on
global self-attention as its core temporal operation.

### CTC Head

The classifier projects encoder states into a character vocabulary plus the CTC
blank symbol. The training objective is CTC loss, allowing alignment-free
training from utterance-level transcripts.

## Language Model Exclusion

No external language model is used in the main experiment. This is deliberate:
Sanday is intended to test the acoustic model itself.
