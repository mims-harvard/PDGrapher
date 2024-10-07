from copy import deepcopy
import string
from time import perf_counter
from typing import Callable

from lightning.fabric.wrappers import _FabricModule
import numpy as np
import torch
import torch.nn as nn


def _test_condition(condition: bool, text: str):
    if not condition:
        raise ValueError(text)


# SAMPLE WEIGHTS
""" def cal_weights_model_1_forward(dataset):
    #predicting 'diseased'
    labels = []
    for data in dataset:
        labels += data.diseased.tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return torch.tensor([n_full / (2 * n_negative), n_full / (2 * n_positive)])

def cal_weights_model_1_backward(dataset):
    #predicting 'treated'
    labels = []
    for data in dataset:
        labels += data.treated.tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return torch.tensor([n_full / (2 * n_negative), n_full / (2 * n_positive)])

def cal_weights_model_2_backward(dataset):
    #predicting 'intervention'
    labels = []
    for data in dataset:
        labels += data.intervention.tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return torch.tensor([n_full / (2 * n_negative), n_full / (2 * n_positive)]) """


def calculate_loss_sample_weights(dataset, kind: str) -> torch.Tensor:
    _test_condition(kind in {"diseased", "treated", "intervention"}, "`kind` should be one of (diseased, treated, intervention)")
    labels = []
    for data in dataset:
        labels += getattr(data, kind).tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return torch.tensor([n_full/(2*n_negative), n_full/(2*n_positive)])


""" def get_threshold_healthy(dataset):
    all_healthy_values = []
    for data in dataset:
        all_healthy_values.append(data.healthy.cpu())
    percentiles = torch.Tensor(np.percentile(torch.stack(all_healthy_values).flatten(), [e for e in np.arange(0,100,0.2)] + [100]))
    return percentiles

def get_threshold_diseased(dataset):
    all_diseased_values = []
    for data in dataset:
        all_diseased_values.append(data.diseased.cpu())
    percentiles = torch.Tensor(np.percentile(torch.stack(all_diseased_values).flatten(), [e for e in np.arange(0,100,0.2)] + [100]))
    return percentiles

def get_threshold_treated(dataset):
    all_treated_values = []
    for data in dataset:
        all_treated_values.append(data.treated.cpu())
    percentiles = torch.Tensor(np.percentile(torch.stack(all_treated_values).flatten(), np.arange(0, 100.2, 0.2)))
    return percentiles """


def _get_thresholds(dataset, kind: str):
    _test_condition(kind in {"healthy", "diseased", "treated"}, "`kind` should be one of (diseased, treated, healthy)")
    all_values = [getattr(data, kind).cpu() for data in dataset]
    percentiles = torch.tensor(np.percentile(torch.stack(all_values).flatten(), [e for e in np.arange(0, 100, 0.2)] + [100]))
    return percentiles


def get_thresholds(dataset):
    return {
        'healthy': _get_thresholds(dataset.train_dataset_forward, "healthy") if hasattr(dataset, 'train_dataset_forward') else None,
        'diseased': _get_thresholds(dataset.train_dataset_backward, "diseased"),
        'treated': _get_thresholds(dataset.train_dataset_backward, "treated")
    }


class EarlyStopping:

    def __init__(self, patience: int = 15, skip: int = 0, minmax: str = "min", rope: float = 1e-5,
                 model: _FabricModule = None, save_path: str = None):

        self.skip = skip
        self.patience = patience
        self.rope = abs(rope)

        self.minmax = minmax
        self.comparison_f = (lambda x, y: x < y-self.rope) if self.minmax == "min" else (lambda x, y: x > y+self.rope)

        self.reset()

        self.model = model
        self.save_path = save_path

        self.successful_comparison = (self._save_model if (self.save_path and self.model) else lambda: None)
        self.load_model = (self._load_model if (self.save_path and self.model) else lambda: None)

    def _save_model(self):
        tmp_model = deepcopy(self.model.module)
        torch.save({"epoch": self.skip_counter, "model_state_dict": tmp_model.cpu().state_dict()}, self.save_path)

    def _load_model(self) -> nn.Module:
        checkpoint = torch.load(self.save_path)
        tmp_model = deepcopy(self.model.module)
        tmp_model.load_state_dict(checkpoint["model_state_dict"])
        return tmp_model

    def reset(self):
        self.counter = 0
        self.skip_counter = 0
        self.is_stopped = False
        self.value = float("inf") if self.minmax == "min" else -float("inf")

    def __call__(self, value):
        self.skip_counter += 1
        if self.skip_counter < self.skip:
            if self.comparison_f(value, self.value): # even when skipping, save best value
                self.value = value
                self.successful_comparison()
            return False

        if self.comparison_f(value, self.value):
            self.value = value
            self.counter = 0
            self.successful_comparison()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.is_stopped = True
                return True

        return False


class DummyEarlyStopping(EarlyStopping):

    def __init__(self, patience: int = 15, skip: int = 0, minmax: str = "min", rope: float = 1e-5,
                 model: _FabricModule = None, save_path: str = None):
        super().__init__(patience, skip, minmax, rope, model, save_path)

        self.successful_comparison = lambda: None
        self.load_model = lambda: None

    def __call__(self, value):
        return False


def tictoc(*args):
    # https://stackoverflow.com/questions/3931627/how-to-build-a-decorator-with-optional-parameters
    def wrap(function: Callable):
        def wrapped_f(*args, **kwargs):
            tic = perf_counter()
            result = function(*args, **kwargs)
            toc = perf_counter()
            print(text_to_format.format(toc-tic))
            return result
        return wrapped_f
    if len(args) >= 1 and callable(args[0]):
        text_to_format: str = args[1] if len(args) >= 2 else "{}secs"
        to_return = wrap(args[0])
    else:
        text_to_format = args[0] if args else "{}secs"
        to_return = wrap
    matches = [tup[1] for tup in string.Formatter().parse(text_to_format) if tup[1] is not None]
    if len(matches) != 1:
        raise ValueError(r"tictoc decorator requires string with one {}!")
    return to_return


class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
