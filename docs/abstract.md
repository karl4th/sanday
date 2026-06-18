# Abstract

Sanday investigates automatic speech recognition with liquid neural networks.
Instead of treating speech as a long discrete token sequence for a transformer
encoder, Sanday models the acoustic stream as a continuous dynamical system.
The project uses Closed-form Continuous-time networks (CfC) from the `ncps`
library as the core temporal encoder.

The central hypothesis is that a compact CfC acoustic model can compete with
transformer-based speech encoders in constrained ASR settings while using
approximately three orders of magnitude fewer parameters. The target model
contains roughly two million trainable parameters.

To isolate acoustic modeling capability, Sanday does not use an external
language model during decoding. The model consumes mel spectrogram features and
is trained with a CTC objective on Common Voice data obtained through Kaggle.

This design makes Sanday a study of parameter efficiency, continuous-time
sequence modeling, and low-resource ASR deployment rather than a full production
speech recognition stack.
