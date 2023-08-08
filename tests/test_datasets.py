import unittest

from pdgrapher import Dataset


class TestDataset(unittest.TestCase):

    dataset = None

    def test_creation(self):
        self.__class__.dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
        )

    def test_indices(self):
        self.__class__.dataset._test_indices()
    
    def test_dataloaders(self):
        dataloaders = self.__class__.dataset.get_dataloaders()
        self.assertEqual(len(dataloaders), 6)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
