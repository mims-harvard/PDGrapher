import random
import unittest

from torch import load as torch_load

from pdgrapher._models import GCNArgs, GCNBase, ResponsePredictionModel, PerturbationDiscoveryModel
import torch
import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestGCNArgs(unittest.TestCase):

    DEFAULT_GCNARGS_DICT = {
            "positional_features_dims": 16,
            "embedding_layer_dim": 16,
            "dim_gnn": 16,
            "num_vars": 1,
            "n_layers_gnn": 1,
            "n_layers_nn": 2,
        }

    def test_empty_args(self):
        gcnargs_obj = GCNArgs()

    def test_from_to_dict(self):
        gcnargs_dict = {}
        gcnargs_obj = GCNArgs()
        self.assertEqual(GCNArgs.from_dict(gcnargs_dict), gcnargs_obj)

        gcnargs_dict = {"n_layers_gnn": 3, "embedding_layer_dim": 4, "n_layers_nn": 1}
        gcnargs_obj = GCNArgs(n_layers_gnn=3, embedding_layer_dim=4, n_layers_nn=1)
        obj = GCNArgs.from_dict(gcnargs_dict)
        self.assertEqual(obj, gcnargs_obj)
        gcnargs_dict = {**self.DEFAULT_GCNARGS_DICT, **gcnargs_dict}
        self.assertDictEqual(obj.to_dict(), gcnargs_dict)


class TestGCNBase(unittest.TestCase):

    edge_index = torch_load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")

    def test_empty_args(self):
        gcnargs = GCNArgs()
        gcnbase = GCNBase(gcnargs, "response", self.edge_index)
        with self.assertRaises(NotImplementedError):
            gcnbase.forward(None)

    def test_some_args(self):
        gcnargs = GCNArgs(n_layers_gnn=4, n_layers_nn=4)
        gcnbase = GCNBase(gcnargs, "response", self.edge_index)

        # test if out_fun is identity
        for i in range(100):
            x = random.randint(0, 100)
            self.assertEqual(gcnbase.out_fun(x), x)

        # test number of GNN layers
        self.assertEqual(len(gcnbase.convs), 4)
        # test number of NN layers
        self.assertEqual(len(gcnbase.mlp), 1+4+1) # input + specified + output layers


class TestResponsePredictionModel(unittest.TestCase):

    edge_index = torch_load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")

    def test_empty_args(self):
        gcnargs = GCNArgs()
        model = ResponsePredictionModel(gcnargs, self.edge_index)

    def test_some_args(self):
        gcnargs = GCNArgs(n_layers_gnn=2, n_layers_nn=2)
        model = ResponsePredictionModel(gcnargs, self.edge_index)

        # test if out_fun is identity
        for i in range(100):
            x = random.randint(0, 100)
            self.assertEqual(model.out_fun(x), x)

        # test number of GNN layers
        self.assertEqual(len(model.convs), 2)
        # test number of NN layers
        self.assertEqual(len(model.mlp), 1+2+1) # input + specified + output layers


class TestPerturbationDiscoveryModel(unittest.TestCase):

    edge_index = torch_load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")

    def test_empty_args(self):
        gcnargs = GCNArgs()
        model = PerturbationDiscoveryModel(gcnargs, self.edge_index)

    def test_some_args(self):
        gcnargs = GCNArgs(n_layers_gnn=2, n_layers_nn=2)
        model = PerturbationDiscoveryModel(gcnargs, self.edge_index)

        # test if out_fun is identity
        for i in range(100):
            x = random.randint(0, 100)
            self.assertEqual(model.out_fun(x), x)

        # test number of GNN layers
        self.assertEqual(len(model.convs), 2)
        # test number of NN layers
        self.assertEqual(len(model.mlp), 1+2+1) # input + specified + output layers


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
