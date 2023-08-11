import unittest
import os.path as osp

from torch import load as torch_load
from torch.nn import Module as nn_Module

from pdgrapher import PDGrapher, Dataset, Trainer


class TestPackage(unittest.TestCase):

    def test_single_fold(self):
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
        )
        edge_index = torch_load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
        model = PDGrapher(edge_index)
        trainer = Trainer(
            fabric_kwargs={"accelerator": "gpu"}, log=True, logging_dir="tests/PDGrapher_test"
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
        
        # Check if all files exist
        self.assertTrue(osp.isfile(osp.abspath("tests/PDGrapher_test/params.txt")))
        self.assertTrue(osp.isfile(osp.abspath("tests/PDGrapher_test/metrics.txt")))
        self.assertTrue(osp.isfile(osp.abspath("tests/PDGrapher_test/response_prediction.pt")))
        self.assertTrue(osp.isfile(osp.abspath("tests/PDGrapher_test/perturbation_discovery.pt")))

    def test_multiple_folds(self):
        dataset = Dataset(
            forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
            backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
            splits_path="data/splits/genetic/A549/random/5fold/splits.pt"
        )
        edge_index = torch_load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
        trainer = Trainer(
            fabric_kwargs={"accelerator": "gpu"}, log=True, logging_dir="tests/PDGrapher_test"
        )

        for fold_idx in range(dataset.num_of_folds):
            dataset.prepare_fold(fold_idx)
            trainer.logging_paths(name=f"fold_{fold_idx}_")

            model = PDGrapher(edge_index)

            train_metrics = trainer.train(model, dataset, 1)
            print(train_metrics)

        # Check if all fold files have been created
        for fold_idx in range(dataset.num_of_folds):
            self.assertTrue(osp.isfile(osp.abspath(f"tests/PDGrapher_test/fold_{fold_idx}_params.txt")))
            self.assertTrue(osp.isfile(osp.abspath(f"tests/PDGrapher_test/fold_{fold_idx}_metrics.txt")))
            self.assertTrue(osp.isfile(osp.abspath(f"tests/PDGrapher_test/fold_{fold_idx}_response_prediction.pt")))
            self.assertTrue(osp.isfile(osp.abspath(f"tests/PDGrapher_test/fold_{fold_idx}_perturbation_discovery.pt")))


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
