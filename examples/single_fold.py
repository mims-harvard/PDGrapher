import torch
import torch.optim as optim

from pdgrapher import Dataset, PDGrapher, Trainer

import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    dataset = Dataset(
        forward_path="data/processed/torch_data/real_lognorm/data_forward_A549.pt",
        backward_path="data/processed/torch_data/real_lognorm/data_backward_A549.pt",
        splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
    )

    edge_index = torch.load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
        "dim_gnn": 8, "num_vars": dataset.get_num_vars()
        })
    model.set_optimizers_and_schedulers([optim.Adam(model.response_prediction.parameters(), lr=0.0075),
        optim.Adam(model.perturbation_discovery.parameters(), lr=0.0033)])

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        devices = 1,
        log=True, logging_name="tuned",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )

    model_performance = trainer.train(model, dataset, n_epochs = 2)

    print(model_performance)
    with open(f"examples/PDGrapher/tuned_final.txt", "w") as f:
        f.write(str(model_performance))


if __name__ == "__main__":
    main()
