import unittest

from torch import load as torch_load
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from pdgrapher import PDGrapher

import torch
import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestPDGrapher(unittest.TestCase):

    edge_index = torch_load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")

    def test_pdgrapher(self):
        model = PDGrapher(self.edge_index)

    def test_pdgrapher_args(self):
        model = PDGrapher(
            self.edge_index, response_kwargs={"n_layers_gnn": 2, "n_layers_nn": 2, "train": False},
            perturbation_kwargs={"n_layers_gnn": 2, "n_layers_nn": 2, "train": True})

        # test response_args
        self.assertEqual(len(model.response_prediction.convs), 2)
        self.assertEqual(len(model.response_prediction.mlp), 1+2+1)
        self.assertFalse(model._train_response_prediction)

        # test perturbation_args
        self.assertEqual(len(model.perturbation_discovery.convs), 2)
        self.assertEqual(len(model.perturbation_discovery.mlp), 1+2+1)
        self.assertTrue(model._train_perturbation_discovery)

    def test_missing_optimizers(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, None])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([None, o2])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[], []])

    def test_single_optimizer(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)

        model.set_optimizers_and_schedulers([o1, o2])
        model.set_optimizers_and_schedulers([[o1], o2])
        model.set_optimizers_and_schedulers([o1, [o2]])
        model.set_optimizers_and_schedulers([[o1], [o2]])

    def test_single_optimizer_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m1.parameters(), 1) # for m1

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, o2])

    def test_multiple_optimizers(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m2.parameters(), 1)

        model.set_optimizers_and_schedulers([[o11, o12], o21])
        model.set_optimizers_and_schedulers([o11, [o21, o22]])
        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]])

    def test_multiple_optimizers_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m1.parameters(), 1) # for m1

        # model.set_optimizers_and_schedulers([[o11, o12], o21])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[o11, o12], o22])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]])

    def test_single_optimizer_single_scheduler(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)
        s1 = lr_scheduler.LinearLR(o1)
        s2 = lr_scheduler.LinearLR(o2)

        model.set_optimizers_and_schedulers([o1, o2], [s1, None])
        model.set_optimizers_and_schedulers([o1, o2], [None, s2])
        model.set_optimizers_and_schedulers([o1, o2], [s1, s2])

    def test_single_optimizer_single_scheduler_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)
        s1 = lr_scheduler.LinearLR(o1)
        s2 = lr_scheduler.LinearLR(o1) # for o1, m1

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, o2], [None, s2])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, o2], [s1, s2])

    def test_single_optimizer_multiple_schedulers(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)
        s11 = lr_scheduler.LinearLR(o1)
        s12 = lr_scheduler.LinearLR(o1)
        s21 = lr_scheduler.LinearLR(o2)
        s22 = lr_scheduler.LinearLR(o2)

        model.set_optimizers_and_schedulers([o1, o2], [[s11, s12], s21])
        model.set_optimizers_and_schedulers([o1, o2], [s11, [s21, s22]])
        model.set_optimizers_and_schedulers([o1, o2], [[s11, s12], [s21, s22]])

    def test_single_optimizer_multiple_schedulers_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o1 = optim.SGD(m1.parameters(), 1)
        o2 = optim.SGD(m2.parameters(), 1)
        s11 = lr_scheduler.LinearLR(o1)
        s12 = lr_scheduler.LinearLR(o1)
        s21 = lr_scheduler.LinearLR(o2)
        s22 = lr_scheduler.LinearLR(o1) # for o1, m1

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, o2], [[s11, s12], s22])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([o1, o2], [[s11, s12], [s21, s22]])

    def test_multiple_optimizers_single_scheduler(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m2.parameters(), 1)
        s1 = lr_scheduler.LinearLR(o11)
        s2 = lr_scheduler.LinearLR(o21)

        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [s1, None])
        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [None, s2])
        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [s1, s2])

    def test_multiple_optimizers_single_scheduler_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m2.parameters(), 1)
        s1 = lr_scheduler.LinearLR(o11)
        s2 = lr_scheduler.LinearLR(o12) # for o12, m1

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [None, s2])
        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [s1, s2])

    def test_multiple_optimizers_multiple_schedulers(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m2.parameters(), 1)
        s11 = lr_scheduler.LinearLR(o11)
        s12 = lr_scheduler.LinearLR(o12)
        s21 = lr_scheduler.LinearLR(o21)
        s22 = lr_scheduler.LinearLR(o22)

        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [s11, [s21, s22]])
        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [[s11, s12], s21])
        model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [[s11, s12], [s21, s22]])

    def test_multiple_optimizers_multiple_schedulers_wrong(self):
        model = PDGrapher(self.edge_index)

        m1 = model.response_prediction
        m2 = model.perturbation_discovery
        o11 = optim.SGD(m1.parameters(), 1)
        o12 = optim.SGD(m1.parameters(), 1)
        o21 = optim.SGD(m2.parameters(), 1)
        o22 = optim.SGD(m2.parameters(), 1)
        s11 = lr_scheduler.LinearLR(o11)
        s12 = lr_scheduler.LinearLR(o12)
        s21 = lr_scheduler.LinearLR(o21)
        s22 = lr_scheduler.LinearLR(o12) # for o12, m1

        with self.assertRaises(ValueError):
            model.set_optimizers_and_schedulers([[o11, o12], [o21, o22]], [[s11, s12], [s21, s22]])


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
