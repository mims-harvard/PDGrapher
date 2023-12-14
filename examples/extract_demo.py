"""
This file generates the data that is used by the HuggingFace Space for the
PDGrapher.

At the moment it just loads the first backward test batch with the size of 5,
and runs it through each model, first seperately, and then together to simulate
the prediction of perturbagen and the verification of its effects.
"""

import torch
from pdgrapher import Dataset, PDGrapher
from pdgrapher._utils import get_thresholds, calculate_loss_sample_weights

dataset = Dataset(
    forward_path="data/processed/torch_data/real_lognorm/data_forward_A549.pt",
    backward_path="data/processed/torch_data/real_lognorm/data_backward_A549.pt",
    splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
)

edge_index = torch.load("data/processed/torch_data/real_lognorm/edge_index_A549.pt")
model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars()})

save_path = f"examples/PDGrapher/fold_1_response_prediction.pt"
checkpoint = torch.load(save_path)
model.response_prediction.load_state_dict(checkpoint["model_state_dict"])
save_path = f"examples/PDGrapher/fold_1_perturbation_discovery.pt"
checkpoint = torch.load(save_path)
model.perturbation_discovery.load_state_dict(checkpoint["model_state_dict"])

sample_weights_model_2_backward = calculate_loss_sample_weights(dataset.train_dataset_backward, "intervention")
pos_weight = sample_weights_model_2_backward[1] / sample_weights_model_2_backward[0]
thresholds = get_thresholds(dataset)

_, _, _, _, _, test_loader_backward = dataset.get_dataloaders(batch_size=5)

for db in test_loader_backward:
    break
data = db

model_1 = model.response_prediction
model_2 = model.perturbation_discovery


out_rp_alone, _ = model_1(torch.concat([data.diseased.view(-1, 1), data.intervention.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, binarize_intervention=False, threshold_input=thresholds["diseased"])
out_pd_alone = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
out_rp_together, out_pd_together = model_1(torch.concat([data.diseased.view(-1, 1), out_pd_alone], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)

torch.save(data, "dataloader_backward.pt")
torch.save(out_rp_alone, "out_response_alone.pt")
torch.save(out_pd_alone, "out_perturbation_alone.pt")
torch.save(out_rp_together, "out_response_together.pt")
torch.save(out_pd_together, "out_perturbation_together.pt")