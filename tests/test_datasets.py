import unittest

from pdgrapher import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset(self):
        # test creation
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/1fold/splits.pt",
            test_indices=False
        )

        # test indices -> data dependant
        dataset._test_indices()

        # test dataloaders
        dataloaders = dataset.get_dataloaders()
        self.assertEqual(len(dataloaders), 6)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
