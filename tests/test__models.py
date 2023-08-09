import random
import unittest

from torch import load as torch_load

from pdgrapher._models import GCNArgs, GCNBase, ResponsePredictionModel, PerturbationDiscoveryModel


class TestGCNArgs(unittest.TestCase):
    
    def test_empty_args(self):
        obj = GCNArgs()

    def test_from_dict(self):
        gcnargs_dict = {}
        gcnargs_obj = GCNArgs()
        self.assertEqual(GCNArgs.from_dict(gcnargs_dict), gcnargs_obj)

        gcnargs_dict = {"n_layers_gnn": 3, "embedding_layer_dim": 4}
        gcnargs_obj = GCNArgs(n_layers_gnn=3, embedding_layer_dim=4)
        self.assertEqual(GCNArgs.from_dict(gcnargs_dict), gcnargs_obj)


class TestGCNBase(unittest.TestCase):

    edge_index = torch_load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
    
    def test_empty_args(self):
        gcnargs = GCNArgs()
        gcnbase = GCNBase(gcnargs, "response", self.edge_index)
    
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
        self.assertEqual(len(gcnbase.mlp), 4+1) # specified + output layer


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
