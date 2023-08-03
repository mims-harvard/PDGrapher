import io
import random
import sys
import unittest

from torch import Tensor

from pdgrapher._utils import (
    _test_condition,
    calculate_loss_sample_weights,
    _get_thresholds, get_thresholds,
    save_best_model, load_best_model,
    tictoc,
    EarlyStopping,
    DummyWriter
)


class Test_test_condition(unittest.TestCase):

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
    
    def _true_ab(self, kind: str):
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
        true_a, true_b = self._true_ab("diseased")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)
    
    def test_treated(self):
        a, b = calculate_loss_sample_weights(self.dataset, "treated")
        true_a, true_b = self._true_ab("treated")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)

    def test_intervention(self):
        a, b = calculate_loss_sample_weights(self.dataset, "intervention")
        true_a, true_b = self._true_ab("intervention")
        self.assertAlmostEqual(a, true_a)
        self.assertAlmostEqual(b, true_b)


class TestGetThresholds(unittest.TestCase):
    
    def setUp(self):
        random.seed(0)

        class Dataset1:
            def __init__(self, N):
                self._data = Tensor([random.randint(0, 5) for _ in range(N)])
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


class TestSaveLoadModel(unittest.TestCase):
    pass


class TestTicToc(unittest.TestCase):

    def test_output_basic(self):
        @tictoc
        def function_basic():
            pass
        
        # redirect stdout to object
        captured_stdout = io.StringIO()
        sys.stdout = captured_stdout
        function_basic()
        # restore stdout
        sys.stdout = sys.__stdout__
        printed_stdout = captured_stdout.getvalue()

        self.assertTrue(printed_stdout.endswith("secs"))



if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
