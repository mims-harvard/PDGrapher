import unittest

import torch
torch.set_num_threads(4)

from pdgrapher import Dataset


class TestDataset(unittest.TestCase):

    def test_single_fold(self):
        # test creation
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
        )

        # test dataloaders
        dataloaders = dataset.get_dataloaders()
        self.assertEqual(len(dataloaders), 6)

    def test_multiple_folds(self):
        # test creation
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/5fold/splits.pt",
        )

        # test dataloaders for all folds
        for fold_idx in range(dataset.num_of_folds):
            dataset.prepare_fold(fold_idx)
            dataloaders = dataset.get_dataloaders()
            self.assertEqual(len(dataloaders), 6)

        # test invalid fold_idx
        with self.assertRaises(ValueError):
            dataset.prepare_fold(-1)
        with self.assertRaises(ValueError):
            dataset.prepare_fold(1.2)
        with self.assertRaises(ValueError):
            dataset.prepare_fold(dataset.num_of_folds+1)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
