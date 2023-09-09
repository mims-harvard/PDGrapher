# PDGrapher: TODO

This file summarizes things that still need to be done and some observations.

## Bugs:
- There is some CUDA memory leak, but it is only present when we train a lot of models in one go, as in train them with one `python ...` call. An example is [examples/hyperparameter_tuning.py](examples/hyperparameter_tuning.py) script, which crashed due to OOM error in CUDA after training >130 models for 1 epoch. I suspect it is related to lightning.Fabric, which handles moving tensort to and from GPU.

## TODO:
- [data](data/) folder has no data currently, and the folder structure with 'raw' folders is not consistent with code in scripts folders. Maybe we can rename them to '2022-03-PPI', '2022-02-LINCS_Level3' and so on. There are two such PPI folders, so we need to be careful there.
- [main README.md](README.md) needs some updating, like add authors, project description.