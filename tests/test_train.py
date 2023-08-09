import unittest

from pdgrapher import Trainer


class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        trainer = Trainer()

    def test_trainer_args(self):
        trainer = Trainer(
            log_train=False, use_backward_data=True, supervision_multiplier=1.3,
            accelerator="cpu" # Fabric args
        )

        self.assertFalse(trainer.log_train)
        self.assertTrue(trainer.use_backward_data)
        self.assertEqual(trainer.supervision_multiplier, 1.3)
        
        self.assertEqual(trainer.fabric.device.type, "cpu")


if __name__ == "__main__":
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
