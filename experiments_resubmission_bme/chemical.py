
import scipy
import scipy.spatial
from pdgrapher import Dataset, PDGrapher, Trainer
import numpy as np
import torch

import os
import os.path as osp
torch.set_num_threads(20)
from datetime import datetime
from pdgrapher import Dataset, PDGrapher, Trainer
import sys

#PARAMS
cell_line = sys.argv[1]  # Get the cell line from command line arguments
n_layers_nn = 1
global use_forward_data
use_forward_data = False
if cell_line in ['HA1E', 'HT29', 'A375', 'HELA']:
    use_forward_data = False

use_backward_data = True
use_supervision = True #whether to use supervision loss
use_intervention_data = True #whether to use cycle loss
current_date = datetime.now()
os.makedirs('./results/chemical/', exist_ok=True)

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """
    outdir = f'./results/chemical/{cell_line}_corrected_pos_emb_no_forward'
    os.makedirs(outdir, exist_ok=True)



    for n_layers_gnn in [1, 2, 3]:
        outdir_i = osp.join(outdir, 'n_gnn_{}_n_nn_{}_usef_{}_useb_{}_usecycle_{}_usesup_{}_split_random___{}_{}_{}___{}_{}_{}'.format(n_layers_gnn,
                                                                                                                                    n_layers_nn,
                                                                                                                                    use_forward_data,
                                                                                                                                    use_backward_data,
                                                                                                                                    use_intervention_data,
                                                                                                                                    use_supervision,
                                                                                                                                    current_date.year, 
                                                                                                                                    current_date.month, 
                                                                                                                                    current_date.day, 
                                                                                                                                    current_date.hour, 
                                                                                                                                    current_date.minute, 
                                                                                                                                    current_date.second))
        os.makedirs(outdir_i, exist_ok = False)
        dataset = Dataset(
            forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line}.pt",
            backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line}.pt",
            splits_path=f"../data/processed/splits/chemical/{cell_line}/random/5fold/splits.pt"
        )

        edge_index = torch.load(f"../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line}.pt")
        model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": n_layers_gnn, "num_vars": dataset.get_num_vars(),
                                                    },
                                    response_kwargs={'train': True},
                                    perturbation_kwargs={'train':True})

        trainer = Trainer(
            fabric_kwargs={"accelerator": "cuda"},
            log=True, use_forward_data=use_forward_data, use_backward_data=use_backward_data, use_supervision=use_supervision,
            use_intervention_data=use_intervention_data, supervision_multiplier=0.01,
            log_train=True, log_test=True,
            logging_dir = outdir_i
        )

        # Iterate over all of the folds and train on each one
        model_performances = trainer.train_kfold(model, dataset, n_epochs = 50)

        print(model_performances)
        with open(osp.join(outdir_i, "multifold.txt"), "w") as f:
            f.write(str(model_performances))


if __name__ == "__main__":
    main()