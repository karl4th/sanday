# Project Notes

Known project facts:

- Project name: Sanday
- Team: Manifestro
- Domain: automatic speech recognition
- Core idea: ASR with liquid neural networks
- Library: `ncps`
- Cell type: CfC
- Input representation: mel spectrogram
- Loss/decoding direction: CTC acoustic model
- Parameter target: approximately 2M parameters
- Dataset: Common Voice from Kaggle via `kagglehub`
- Language: English
- Environment: Google Colab notebooks
- External language model: intentionally excluded
- Metrics: CER and WER

Open questions:

- Exact vocabulary: characters, BPE, or another unit set
- Exact WER/CER results
- Exact training hyperparameters
- Exact model dimensions that produced the 2M-parameter target
- Whether the workflow should stay notebook-only or use separate modules
