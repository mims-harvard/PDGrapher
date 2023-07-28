import torch

from pdgrapher import Dataset, PDGrapher, Trainer

def main():
    dataset = Dataset() # TODO add paths
    edge_index = torch.load("edge_index_path")
    model = PDGrapher(edge_index)
    trainer = Trainer()

    train_metrics = trainer.train(model, dataset, 10)
    print(train_metrics)


if __name__ == "__main__":
    main()