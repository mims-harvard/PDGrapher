import unittest

from torch import load as torch_load
from torch.nn import Module as nn_Module

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
        trainer = Trainer(
            fabric_kwargs={"accelerator": "gpu"}, log=True, logging_dir="tests/PDGrapher_test",
        )

        model_performance = trainer.train(model, dataset, 2)

        # Check model types, should not be _Fabric_Module
        self.assertIsInstance(model.response_prediction, nn_Module)
        self.assertIsInstance(model.perturbation_discovery, nn_Module)
        
        # Check return value
        self.assertIn("train", model_performance)
        self.assertIn("test", model_performance)
        for key in [
            "forward_spearman", "forward_mae", "forward_mse", "forward_r2",
            "forward_r2_scgen", "backward_spearman", "backward_mae",
            "backward_mse", "backward_r2", "backward_r2_scgen", "backward_avg_topk"
        ]:
            self.assertIn(key, model_performance["train"])
            self.assertIn(key, model_performance["test"])


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
