import numpy as np
import torch

from pdgrapher import Dataset, PDGrapher, Trainer

def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = Dataset(
        forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
        backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
        splits_path="data/splits/genetic/A549/random/5fold/splits.pt"
    )

    edge_index = torch.load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
    model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars()})

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=False, log_test=True
    )

    model_performances = trainer.train_kfold(model, dataset, 50)

    print(model_performances)
    with open(f"examples/PDGrapher/multifold_final.txt", "w") as f:
        f.write(str(model_performances))


if __name__ == "__main__":
    main()