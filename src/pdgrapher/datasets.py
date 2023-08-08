from typing import List

import torch
from torch_geometric.loader import DataLoader

from pdgrapher._utils import _test_condition

__all__ = ["Dataset"]

_TVT_IDX = ["train_index_forward", "train_index_backward",
            "val_index_forward", "val_index_backward",
            "test_index_forward", "test_index_backward"
            ]


class Dataset:

    def __init__(self, forward_path: str, backward_path: str, splits_path: str,
                 test_indices: bool = True):
        self.dataset_forward = torch.load(forward_path)
        self.dataset_backward = torch.load(backward_path)

        self.splits = torch.load(splits_path) # TODO this is single_fold, for k-fold self.splits is a list
        self.train_index_forward = self.splits["train_index_forward"]
        self.train_index_backward = self.splits["train_index_backward"]
        self.val_index_forward = self.splits["val_index_forward"]
        self.val_index_backward = self.splits["val_index_backward"]
        self.test_index_forward = self.splits["test_index_forward"]
        self.test_index_backward = self.splits["test_index_backward"]

        # for name in _TVT_IDX:
        #    setattr(self, name, self.splits[name])

        # Test for index overlap
        if test_indices:
            self._test_indices()

        self.train_dataset_forward = [self.dataset_forward[i] for i in self.train_index_forward]
        self.val_dataset_forward = [self.dataset_forward[i] for i in self.val_index_forward]
        self.test_dataset_forward = [self.dataset_forward[i] for i in self.test_index_forward]
        self.train_dataset_backward = [self.dataset_backward[i] for i in self.train_index_backward]
        self.val_dataset_backward = [self.dataset_backward[i] for i in self.val_index_backward]
        self.test_dataset_backward = [self.dataset_backward[i] for i in self.test_index_backward]


    def _test_indices(self):
        set_trif = set(self.train_index_forward)
        set_vif = set(self.val_index_forward)
        set_teif = set(self.test_index_forward)
        set_trib = set(self.train_index_backward)
        set_vib = set(self.val_index_backward)
        set_teib = set(self.test_index_backward)
        _test_condition(not any(x in set_vif for x in set_trif), "Overlap between train and validation indices should be zero (forward)")
        _test_condition(not any(x in set_teif for x in set_trif), "Overlap between train and test indices should be zero (forward)")
        _test_condition(not any(x in set_teif for x in set_vif), "Overlap between validation and test indices should be zero (forward)")
        _test_condition(not any(x in set_vib for x in set_trib), "Overlap between train and validation indices should be zero (backward)")
        _test_condition(not any(x in set_teib for x in set_trib), "Overlap between train and test indices should be zero (backward)")
        _test_condition(not any(x in set_teib for x in set_vib), "Overlap between validation and test indices should be zero (backward)")


    def get_dataloaders(self, batch_size: int = 64, **kwargs) -> List[DataLoader]:
        return [
            DataLoader(self.train_dataset_forward, batch_size=batch_size, **kwargs),
            DataLoader(self.train_dataset_backward, batch_size=batch_size, **kwargs),
            DataLoader(self.val_dataset_forward, batch_size=batch_size, **kwargs),
            DataLoader(self.val_dataset_backward, batch_size=batch_size, **kwargs),
            DataLoader(self.test_dataset_forward, batch_size=batch_size, **kwargs),
            DataLoader(self.test_dataset_backward, batch_size=batch_size, **kwargs)
        ]

    def get_train_forward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset_forward, batch_size=batch_size, **kwargs)

    def get_train_backward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset_backward, batch_size=batch_size, **kwargs)

    def get_val_forward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset_forward, batch_size=batch_size, **kwargs)

    def get_val_backward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset_backward, batch_size=batch_size, **kwargs)

    def get_test_forward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset_forward, batch_size=batch_size, **kwargs)

    def get_test_backward_dataloader(self, batch_size: int = 64, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset_backward, batch_size=batch_size, **kwargs)
