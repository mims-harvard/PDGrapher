from typing import List

import torch
from torch_geometric.loader import DataLoader

from ._utils import _test_condition


__all__ = ["Dataset"]


class Dataset:

    def __init__(self, forward_path: str, backward_path: str, splits_path: str,
                 test_indices: bool = True):
        if forward_path is not None:
            self.dataset_forward = torch.load(forward_path)
        else:
            self.dataset_forward = []
            
        self.dataset_backward = torch.load(backward_path)

        self.splits = torch.load(splits_path)
        if 1 in self.splits.keys(): # we are in a multiple fold regime
            self.num_of_folds = len(self.splits)
        else:
            self.num_of_folds = 1

        # Test for index overlap
        if test_indices:
            if self.num_of_folds > 1:
                for idx in range(1, self.num_of_folds+ 1):
                    self._test_indices(self.splits[idx])
            else:
                self._test_indices(self.splits)

        self.prepare_fold() # loads first fold by default

    def prepare_fold(self, fold_idx: int = 1):
        if self.num_of_folds > 1:
            _test_condition(isinstance(fold_idx, int), "'fold_idx' must be an integer")
            _test_condition(1 <= fold_idx < self.num_of_folds + 1, f"'fold_idx' must be between 1 and {self.num_of_folds+1}")
            fold_splits = self.splits[fold_idx]
        else:
            fold_splits = self.splits

        self.train_index_forward = fold_splits["train_index_forward"]
        self.train_index_backward = fold_splits["train_index_backward"]
        self.val_index_forward = fold_splits["val_index_forward"]
        self.val_index_backward = fold_splits["val_index_backward"]
        self.test_index_forward = fold_splits["test_index_forward"]
        self.test_index_backward = fold_splits["test_index_backward"]

        if self.train_index_forward is not None:
            self.train_dataset_forward = [self.dataset_forward[i] for i in self.train_index_forward]
            self.val_dataset_forward = [self.dataset_forward[i] for i in self.val_index_forward]
            self.test_dataset_forward = [self.dataset_forward[i] for i in self.test_index_forward]
        self.train_dataset_backward = [self.dataset_backward[i] for i in self.train_index_backward]
        self.val_dataset_backward = [self.dataset_backward[i] for i in self.val_index_backward]
        self.test_dataset_backward = [self.dataset_backward[i] for i in self.test_index_backward]

    def get_num_vars(self) -> int:
        return self.dataset_backward[0].num_nodes

    def _test_indices(self, fold_splits):
        #Forward indices - if exists
        if fold_splits["train_index_forward"] is not None:
            set_trif = set(fold_splits["train_index_forward"])
            set_vif = set(fold_splits["val_index_forward"])
            set_teif = set(fold_splits["test_index_forward"])
            _test_condition(not any(x in set_teif for x in set_trif), "Overlap between train and test indices should be zero (forward)")
            _test_condition(not any(x in set_vif for x in set_trif), "Overlap between train and validation indices should be zero (forward)")
            _test_condition(not any(x in set_teif for x in set_vif), "Overlap between validation and test indices should be zero (forward)")


        #Backward indices
        set_trib = set(fold_splits["train_index_backward"])
        set_vib = set(fold_splits["val_index_backward"])
        set_teib = set(fold_splits["test_index_backward"])
        
        _test_condition(not any(x in set_vib for x in set_trib), "Overlap between train and validation indices should be zero (backward)")
        _test_condition(not any(x in set_teib for x in set_trib), "Overlap between train and test indices should be zero (backward)")
        _test_condition(not any(x in set_teib for x in set_vib), "Overlap between validation and test indices should be zero (backward)")

    def get_dataloaders(self, batch_size: int = 64, shuffle = True, **kwargs) -> List[DataLoader]:
        kwargs = {**{"shuffle": shuffle, "drop_last": True}, **kwargs} # default kwargs
        return [
            DataLoader(self.train_dataset_forward, batch_size=batch_size, **kwargs) if hasattr(self, 'train_dataset_forward') else None,
            DataLoader(self.train_dataset_backward, batch_size=batch_size, **kwargs),
            DataLoader(self.val_dataset_forward, batch_size=batch_size, **kwargs) if hasattr(self, 'val_dataset_forward') else None,
            DataLoader(self.val_dataset_backward, batch_size=batch_size, **kwargs),
            DataLoader(self.test_dataset_forward, batch_size=batch_size, **kwargs) if hasattr(self, 'test_dataset_forward') else None,
            DataLoader(self.test_dataset_backward, batch_size=batch_size, **kwargs)
        ]
