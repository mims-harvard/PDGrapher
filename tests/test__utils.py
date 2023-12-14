import io
import random
import sys
import unittest

from lightning.fabric import Fabric
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from pdgrapher._utils import (
    _test_condition,
    calculate_loss_sample_weights,
    _get_thresholds, get_thresholds,
    tictoc,
    EarlyStopping,
    DummyWriter
)
import torch
import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestTestCondition(unittest.TestCase):

    def test_assertion_false(self):
        with self.assertRaises(ValueError):
            _test_condition(False, "This is False")

    def test_assertion_true(self):
        _test_condition(True, "This is True")


class TestCalculateLossSampleWeights(unittest.TestCase):

    def setUp(self):
        """
        calculate_loss_sample_weights expects the next data structure:
        dataset: Iterable[Object1]
        Object1 has attributes diseased, treated, intervention: Object2
        Object2: has method .tolist()
        """

        # TODO generate and calculate this by hand

        random.seed(0)

        class InnerDataset:

            def __init__(self, N):
                self._data = [random.randint(0, 5) for _ in range(N)]
                self._nonzero = sum(1 for x in self._data if x != 0)
                self._size = N

            def tolist(self):
                return self._data

        class OuterDataset:

            def __init__(self):
                self.diseased = InnerDataset(random.randint(3, 5))
                self.treated = InnerDataset(random.randint(3, 5))
                self.intervention = InnerDataset(random.randint(3, 5))

        self.dataset = [OuterDataset() for _ in range(10)]

    def true_ab(self, kind: str):
        positive = sum(getattr(data, kind)._nonzero for data in self.dataset)
        full = sum(getattr(data, kind)._size for data in self.dataset)
        negative = full-positive
        return full/(2*negative), full/(2*positive)

    def test_wrong_kind(self):
        with self.assertRaises(ValueError):
            calculate_loss_sample_weights(self.dataset, "unknown")

    def test_output(self):
        output = calculate_loss_sample_weights(self.dataset, "diseased")
        self.assertEqual(len(output), 2)

    def test_diseased(self):
        a, b = calculate_loss_sample_weights(self.dataset, "diseased")
        true_a, true_b = self.true_ab("diseased")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)

    def test_treated(self):
        a, b = calculate_loss_sample_weights(self.dataset, "treated")
        true_a, true_b = self.true_ab("treated")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)

    def test_intervention(self):
        a, b = calculate_loss_sample_weights(self.dataset, "intervention")
        true_a, true_b = self.true_ab("intervention")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)


class TestGetThresholds(unittest.TestCase):

    def setUp(self):
        random.seed(0)

        class Dataset1:

            def __init__(self, N):
                self._data = torch.tensor([random.randint(0, 5) for _ in range(N)])

            def cpu(self):
                return self._data

        class Dataset2:

            def __init__(self):
                self.healthy = Dataset1(10)
                self.diseased = Dataset1(10)
                self.treated = Dataset1(10)

        class Dataset3:

            def __init__(self):
                self.train_dataset_forward = [Dataset2() for _ in range(10)]
                self.train_dataset_backward = [Dataset2() for _ in range(10)]

        self.dataset = Dataset3()

    def test_wrong_kind(self):
        with self.assertRaises(ValueError):
            _get_thresholds(self.dataset.train_dataset_forward, "unknown")

    def test_output(self):
        percentiles = _get_thresholds(self.dataset.train_dataset_forward, "healthy")
        self.assertEqual(percentiles.shape[0], int(100/0.2 + 1))

    def test_output_all(self):
        thresholds = get_thresholds(self.dataset)
        self.assertIn("healthy", thresholds)
        self.assertIn("diseased", thresholds)
        self.assertIn("treated", thresholds)
        self.assertEqual(thresholds["healthy"].shape[0], int(100/0.2 + 1))
        self.assertEqual(thresholds["diseased"].shape[0], int(100/0.2 + 1))
        self.assertEqual(thresholds["treated"].shape[0], int(100/0.2 + 1))


class TestTicToc(unittest.TestCase):

    def get_captured_stdout(self, fn):
        captured_stdout = io.StringIO()
        sys.stdout = captured_stdout
        fn()
        sys.stdout = sys.__stdout__
        return captured_stdout.getvalue()

    def test_output_basic_1(self):
        @tictoc
        def fn():
            return 0

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.endswith("secs\n"))

    def test_output_basic_2(self):
        @tictoc()
        def fn():
            return 0

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.endswith("secs\n"))

    def test_output_custom(self):
        @tictoc("Custom {}secs")
        def fn():
            return 0

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.startswith("Custom "))
        self.assertTrue(printed_stdout.endswith("secs\n"))

    def test_output_custom_format(self):
        @tictoc("Custom {:1.2f}secs")
        def fn():
            return 0

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.startswith("Custom "))
        self.assertTrue(printed_stdout.endswith("secs\n"))
        self.assertEqual(len(printed_stdout), 15+1) # output + \n

    def test_wrong_format(self):
        with self.assertRaises(ValueError):
            @tictoc("Not formattable")
            def fn():
                return 0

    def test_output_wrapped(self):
        def fn():
            return 0
        fn = tictoc(fn)

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.endswith("secs\n"))

    def test_output_wrapped_custom(self):
        def fn():
            return 0
        fn = tictoc(fn, "Custom {}secs")

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.startswith("Custom "))
        self.assertTrue(printed_stdout.endswith("secs\n"))

    def test_output_wrapped_custom_format(self):
        def fn():
            return 0
        fn = tictoc(fn, "Custom {:1.2f}secs")

        printed_stdout = self.get_captured_stdout(fn)
        self.assertTrue(printed_stdout.startswith("Custom "))
        self.assertTrue(printed_stdout.endswith("secs\n"))
        self.assertEqual(len(printed_stdout), 15+1) # output + \n

    def test_wrapped_wrong_format(self):
        with self.assertRaises(ValueError):
            def fn():
                return 0
            fn = tictoc(fn, "Not formattable")


class TestEarlyStopping(unittest.TestCase):

    def test_saving_loading(self):
        test_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        optimizer = optim.SGD(test_model.parameters(), lr=1e-3)
        fabric = Fabric(accelerator="cpu")
        fabric_model, fabric_optimizer = fabric.setup(test_model, optimizer)

        criterion = nn.MSELoss()

        es = EarlyStopping(model=fabric_model, save_path="tests/PDGrapher_test/test_model.pt")
        es(10) # initial save of the model
        tmp_model = es.load_model() # check if model is the same
        self.assertListEqual(list(test_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        self.assertListEqual(list(fabric_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        for key, val in test_model.state_dict().items():
            val_2 = tmp_model.state_dict()[key]
            if not torch.equal(val, val_2):
                raise AssertionError()

        x = np.random.rand(200, 10).astype("float32")
        y = (x[:, 2] + 2*x[:, 3] + (x[:, 5] - x[:, 7])**2 + 5).reshape(200, 1)

        dataloader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=10)
        fabric_dataloader = fabric.setup_dataloaders(dataloader)

        for _ in range(10): # train for 10 epochs
            for x, y in fabric_dataloader:
                fabric_optimizer.zero_grad()
                out = fabric_model(x)
                loss = criterion(out, y)
                fabric.backward(loss)
                fabric_optimizer.step()

        # check if the model has changed
        tmp_model = es.load_model()
        self.assertListEqual(list(test_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        self.assertListEqual(list(fabric_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        for key, val in test_model.state_dict().items():
            val_2 = tmp_model.state_dict()[key]
            if torch.equal(val, val_2):
                raise AssertionError()

        # check if the model is saved
        es(1)
        tmp_model = es.load_model()
        self.assertListEqual(list(test_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        self.assertListEqual(list(fabric_model.state_dict().keys()), list(tmp_model.state_dict().keys()))
        for key, val in test_model.state_dict().items():
            val_2 = tmp_model.state_dict()[key]
            if not torch.equal(val, val_2):
                raise AssertionError()


class TestDummyWriter(unittest.TestCase):

    def test_create(self):
        writer = DummyWriter()

    def test_writing(self):
        writer = DummyWriter()
        writer.add_scalar("some/key", 10)


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
