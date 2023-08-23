from argparse import ArgumentParser

import torch

from pdgrapher import Dataset, PDGrapher, Trainer

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--fold", default=0, type=int, dest="fold")
    args = parser.parse_args()

    dataset = Dataset(
        forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
        backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
        splits_path="data/splits/genetic/A549/random/5fold/splits.pt"
    )
    dataset.prepare_fold(args.fold)

    edge_index = torch.load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")
    model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars()})

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=False, log_test=True, logging_name=f"fold_{args.fold}_"
    )

    model_performance = trainer.train(model, dataset, 50)

    print(model_performance)
    with open(f"examples/PDGrapher/fold_{args.fold}_final.txt", "w") as f:
        f.write(str(model_performance))


if __name__ == "__main__":
    main()
