import unittest

from pdgrapher import Trainer
import torch
import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        trainer = Trainer()

    def test_trainer_args(self):
        trainer = Trainer(
            fabric_kwargs={"accelerator": "cpu"},
            log_train=False, use_backward_data=True, supervision_multiplier=1.3
        )

        self.assertFalse(trainer.log_train)
        self.assertTrue(trainer.use_backward_data)
        self.assertEqual(trainer.supervision_multiplier, 1.3)

        self.assertEqual(trainer.fabric.device.type, "cpu")

    def test_trainer_unknown_kwargs(self):
        with self.assertWarns(Warning):
            trainer = Trainer(
                unknown_arg=1, another_unknown=True
            )


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
