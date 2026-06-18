# Results Template

Do not report a number here until it has been produced by an actual run.

## Main Run

| Run | Dataset | Language | Params | LM | Decoder | WER | CER | Notes |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| sanday-cfc-2m | Common Voice | English | TBD | no | greedy CTC | TBD | TBD | baseline run |

## Experiment Log

| Date | Config | Change | Params | WER | CER | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| TBD | `sanday_cfc_2m.yaml` | initial CfC CTC run | TBD | TBD | TBD | TBD |

## Reporting Rule

Sanday can claim a result only after recording:

- exact config file
- checkpoint path
- dataset split
- parameter count
- decoding mode
- WER
- CER
- whether an external LM was used

The no-LM setting must stay explicit in every table.
