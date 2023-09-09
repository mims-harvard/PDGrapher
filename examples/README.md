# PDGrapher: Examples

Please refer to the next Python scripts for examples:
- [single_fold.py](single_fold.py) demonstrates how to train one model on data with only one train-test split,
- [multiple_folds.py](multiple_folds.py) demonstrates how to train models on data that has k-fold splits. The number of splits is unlimited,
- [test_folds.py](test_folds.py) demonstrates how we can test an already trained model using a trick where we train the model for 0 epochs (no training), and then the trainer performs testing for us.
- [hyperparameter_tuning.py](hyperparameter_tuning.py) demonstrates how to use [Ray](https://docs.ray.io/en/latest/tune/index.html) library with [Optuna](https://optuna.org/) to perform hyperparameter search. There is some manual variable setting that would otherwise be taken care of by the pdgrapher.Trainer, but here we just copied that part of the code,
- [extract_demo.py](extract_demo.py) is an example of the code we used to generate data for the demo page that is hosted on HuggingFace Spaces.

All of the experiments are designed to run from the root of this repository. The [PDGrapher](PDGrapher/) folder here is just a place that collects all of the output of all of the scripts above. Apparently, HuggingFace has no way of adding collaborators to private spaces