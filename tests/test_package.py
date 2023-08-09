import unittest

from torch import load as torch_load

from pdgrapher import PDGrapher, Dataset, Trainer


class TestPackage(unittest.TestCase):

    def test_package(self):
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
        )
        edge_index = torch_load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
        model = PDGrapher(edge_index)
        trainer = Trainer(log=True, logging_dir="tests/PDGrapher_test")

        model_performance = trainer.train(model, dataset, 5)




if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()