# Experiments

## Main Experiment

Train a roughly 2M-parameter CfC CTC acoustic model on Common Voice mel
spectrograms and evaluate recognition quality without a language model.

## Hypothesis

CfC dynamics can capture enough temporal structure in speech to produce a
competitive compact acoustic model, especially under parameter and latency
constraints.

## Baselines

Potential comparison targets:

- compact recurrent CTC model
- small Conformer/Transformer encoder
- wav2vec-style or Whisper-style model reported parameter counts

The comparison should be careful: full transformer ASR systems often include
pretraining, larger datasets, and stronger decoders. Sanday's clean comparison
is parameter efficiency under a no-LM acoustic setup.

## Ablations

- with and without convolutional frontend
- CfC hidden size
- number of CfC layers
- mel bin count
- greedy CTC vs beam CTC without LM
- parameter count vs WER/CER

## Expected Reporting

Each experiment should report:

- config name
- number of parameters
- dataset split
- training duration
- WER
- CER
- decoding mode
- whether an external LM was used

## Current Target

The baseline reference is approximately 68% WER with roughly 2M parameters. The
next target is a reproducible seed-42 run in the 40-50% WER range without adding
an external language model. Improvements should come from the acoustic pipeline
itself: correct CTC lengths, transcript normalization, SpecAugment,
learning-rate scheduling, and stable checkpoint selection.
