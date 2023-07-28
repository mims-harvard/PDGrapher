import numpy as np
import os
from time import perf_counter
from typing import Tuple, Callable

import torch
import torch.nn as nn

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


def calculate_loss_sample_weights(dataset, kind: str) -> Tuple[int, int]:
    _test_condition(kind in {"diseased", "treated", "intervention"}, "`kind` should be one of (diseased, treated, intervention)")
    labels = []
    for data in dataset:
        labels += getattr(data, kind).tolist()
    labels_tensor = torch.tensor(labels).squeeze()
    n_positive = labels_tensor.nonzero().size(0)
    n_negative = labels_tensor.size(0) - n_positive
    n_full = labels_tensor.size(0)
    return n_full/(2*n_negative), n_full/(2*n_positive)


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
    all_values = [getattr(data, kind).cpu() for data in dataset]
    percentiles = torch.Tensor(np.percentile(torch.stack(all_values).flatten(), [e for e in np.arange(0, 100, 0.2)] + [100]))
    return percentiles


def get_thresholds(dataset):
    return {
        'healthy': _get_thresholds(dataset.train_dataset_forward, "healthy"),
        'diseased': _get_thresholds(dataset.train_dataset_backward, "diseased"),
        'treated': _get_thresholds(dataset.train_dataset_backward, "treated")
    }


def save_best_model(model, file, outdir, optimizer, scheduler, epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },
        os.path.join(outdir, file))


def load_best_model(model, file, outdir):
    checkpoint = torch.load(os.path.join(outdir, file))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class EarlyStopping:

    def __init__(self, patience: int = 15, skip: int = 0, minmax: str = "min", rope: float = 1e-5,
                 model: nn.Module = None, save_path: str = None):

        self.skip = skip
        self.patience = patience
        self.rope = abs(rope)
        self.counter = 0
        self.skip_counter = 0 # also counts epochs
        self.is_stopped = False

        self.minmax = minmax
        self.comparison_f = (lambda x, y: x < y-self.rope) if self.minmax == "min" else (lambda x, y: x > y+self.rope)
        self.value = float("inf") if self.minmax == "min" else -float("inf")

        self.model: nn.Module = model
        self.save_path: str = save_path

        self.successful_comparison = (
            lambda: torch.save({"epoch": self.skip_counter, "model_state_dict": self.model.state_dict()}, self.save_path)
            if (self.save_path and self.model) else lambda: None
        )

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def reset(self):
        self.counter = 0
        self.skip_counter = 0
        self.is_stopped = False
        self.value = float("inf") if self.minmax == "min" else -float("inf")

    def __call__(self, value):
        self.skip_counter += 1
        if self.skip_counter < self.skip:
            self.counter = 0
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


def tictoc(text_to_format: str = "{}secs"):
    def wrap(function: Callable):
        def wrapped_f(*args, **kwargs):
            tic = perf_counter()
            result = function(*args, **kwargs)
            toc = perf_counter()
            print(text_to_format.format(toc-tic))
            return result
        return wrapped_f
    return wrap


class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass


def _test_condition(condition: bool, text: str):
    if not condition:
        raise ValueError(text)
