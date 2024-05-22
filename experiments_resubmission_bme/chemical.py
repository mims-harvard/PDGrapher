import numpy as np
import torch

import os
import os.path as osp
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pdgrapher import Dataset, PDGrapher, Trainer



cell_line = 'A549'
outdir = f'./{cell_line}'
os.makedirs(outdir, exist_ok=True)

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """

    dataset = Dataset(
        forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line}.pt",
        backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line}.pt",
        splits_path=f"../data/processed/splits/chemical/{cell_line}/random/5fold/splits.pt"
    )

    edge_index = torch.load(f"../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line}.pt")
    model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars(),
                                                })

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=True, log_test=True,
        logging_dir = outdir
    )

    # Iterate over all of the folds and train on each one
    model_performances = trainer.train_kfold(model, dataset, n_epochs = 150)

    print(model_performances)
    with open(osp.join(outdir, "multifold.txt"), "w") as f:
        f.write(str(model_performances))


if __name__ == "__main__":
    main()