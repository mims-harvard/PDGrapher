import torch

from pdgrapher import Dataset, PDGrapher, Trainer

def main():
    dataset = Dataset(
        forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
        backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
        splits_path="data/splits/genetic/A549/random/5fold/splits.pt"
    )
    edge_index = torch.load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
    model = PDGrapher(edge_index, model_args={"n_layers_nn": 1, "n_layers_gnn": 1})
    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=False, log_test=True, logging_name="test_"
    )

    train_metrics = trainer.train(model, dataset, 50)
    print(train_metrics)


if __name__ == "__main__":
    main()
