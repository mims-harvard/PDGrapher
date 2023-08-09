from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdgrapher._embed import EmbedLayer
from pdgrapher._torch_geometric import GCNConv

__all__ = ["GCNArgs", "ResponsePredictionModel", "PerturbationDiscoveryModel"]


@dataclass
class GCNArgs():
    positional_features_dims: int = 16
    embedding_layer_dim: int = 16
    dim_gnn: int = 16
    out_channels: int = 1
    num_vars: int = 1
    n_layers_gnn: int = 1
    n_layers_nn: int = 2

    @classmethod
    def from_dict(cls, args: Dict[str, int]) -> "GCNArgs":
        instance = cls()
        instance.positional_features_dims = args.get("positional_features_dims", 16)
        instance.embedding_layer_dim = args.get("embedding_layer_dim", 16)
        instance.dim_gnn = args.get("dim_gnn", 16)
        instance.out_channels = args.get("out_channels", 1)
        instance.num_vars = args.get("num_vars", 1)
        instance.n_layers_gnn = args.get("n_layers_gnn", 1)
        instance.n_layers_nn = args.get("n_layers_nn", 2)
        return instance


class GCNBase(nn.Module):

    def __init__(self, args: GCNArgs, out_fun: str, edge_index: torch.Tensor):
        super().__init__()

        self.positional_features_dims = args.positional_features_dims
        self.edge_index = edge_index

        # Conv layers
        self.convs = nn.ModuleList()
        if args.n_layers_gnn > 0:
            self.convs.append(
                GCNConv(2*args.embedding_layer_dim + args.positional_features_dims, args.dim_gnn, add_self_loops=False)
            )
        if args.n_layers_gnn > 1:
            for _ in range(args.n_layers_gnn-1):
                self.convs.append(
                    GCNConv(args.dim_gnn + 2*args.embedding_layer_dim, args.dim_gnn, add_self_loops=False)
                )

        # Batchnorm GNN
        self.bns = nn.ModuleList()
        for i in range(args.n_layers_gnn):
            self.bns.append(nn.BatchNorm1d(args.dim_gnn + 2*args.embedding_layer_dim))

        # NN layers
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(2*args.embedding_layer_dim + args.dim_gnn, args.dim_gnn))
        for _ in range(args.n_layers_nn-1):
            self.mlp.append(nn.Linear(args.dim_gnn, args.dim_gnn))
        self.mlp.append(nn.Linear(args.dim_gnn, int(args.dim_gnn/2)))
        self.mlp.append(nn.Linear(int(args.dim_gnn/2), args.out_channels))

        # Batchnorm MLP
        self.bns_mlp = nn.ModuleList()
        for _ in range(args.n_layers_nn):
            self.bns_mlp.append(nn.BatchNorm1d(args.dim_gnn))
        self.bns_mlp.append(nn.BatchNorm1d(int(args.dim_gnn/2)))

        # Output function
        out_fun_selector = {'response': lambda x: x, 'perturbation': lambda x: x}
        self.out_fun = out_fun_selector.get(out_fun, lambda x: x)

        # dictionary storing, for each node, its place in edge_index where there
        # is an edge incoming to it (excluding self loops)
        self.dictionary_node_to_edge_index_position = defaultdict(list)
        self.build_dictionary_node_to_edge_index_position()

        self._mutilate_graph = True


    def forward(self, x):
        raise NotImplementedError()


    def mutilate_graph(self, batch, uprime=0, mutilate_mutations=None):
        if mutilate_mutations is not None:
            uprime = mutilate_mutations + uprime
        # buildsx_j_mask sequentially for each sample in the batch
        x_j_mask = []
        for i in range(len(torch.unique(batch))):
            x_j_mask_ind = torch.ones(self.edge_index.size(1), dtype=torch.float, device=self.edge_index.device)
            # Get nodes to intervene on in each batch
            to_intervene_on = torch.where(uprime[batch == i])[0] # - (i * (torch.max(edge_index).item() + 1)) #the subtraction is to correct for gene indices in batche ind > 1

            for node in to_intervene_on:
                mask = self.dictionary_node_to_edge_index_position[node.item()]
                x_j_mask_ind[mask] = 0
            x_j_mask.append(x_j_mask_ind)
        x_j_mask = torch.cat(x_j_mask)
        return x_j_mask


    def from_node_to_out(self, x1, x2, batch, random_dims, x_j_mask=None):
        # Initial node embedding
        x = torch.cat([x1, x2, random_dims], 1)

        # Convs
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(conv(x, self.edge_index, x_j_mask=x_j_mask, batch_size=len(torch.unique(batch))))
            x = torch.cat([x1, x2, x], 1)
            x = bn(x)

        # MLP
        if len(self.mlp) > 1:
            for layer, bn in zip(self.mlp[:-2], self.bns_mlp[:-1]):
                x = bn(F.elu(layer(x)))
            x = self.bns_mlp[-1](F.elu(self.mlp[-2](x)))

        return x


    def build_dictionary_node_to_edge_index_position(self):
        # dictionary storing, for each node, its place in edge_index where
        # there is an edge incoming to it (excluding self loops)
        # will be used to build x_j_mask in get_embeddings()
        for i in range(self.edge_index.size(1)):
            node = self.edge_index[1, i].item()
            if node == self.edge_index[0, i].item():
                continue
            self.dictionary_node_to_edge_index_position[node].append(i)


class ResponsePredictionModel(GCNBase): # GCNModel

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "response", edge_index)

        self.embed_layer_pert = EmbedLayer(args.num_vars, num_features=1, num_categs=2, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_ge = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)

    # TODO def forward(self, x_d, perturbagen) -> x_t
    def forward(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x, in_x_binarized = self.get_embeddings(x, batch, topK, binarize_intervention, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x), in_x_binarized


    def get_embeddings(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
        # Random node initialization
        random_dims = torch.empty(x.shape[0], self.positional_features_dims).to(x.device)
        nn.init.normal_(random_dims)

        # Feature embedding
        x_ge, _ = self.embed_layer_ge(x[:, 0].view(-1, 1), topK=None, binarize_intervention=False, binarize_input=True, threshold_input=threshold_input)
        x_pert, in_x_binarized = self.embed_layer_pert(x[:, 1].view(-1, 1), topK=topK, binarize_intervention=binarize_intervention, binarize_input=False, threshold_input=None)

        if binarize_intervention:
            uprime = in_x_binarized.view(-1).clone()
        else:
            uprime = x[:, 1].clone()

        x_j_mask = None
        if self._mutilate_graph:
            x_j_mask = self.mutilate_graph(batch, uprime, mutilate_mutations)

        x = self.from_node_to_out(x_ge, x_pert, batch, random_dims, x_j_mask)

        return x, in_x_binarized


class PerturbationDiscoveryModel(GCNBase): # GCNModelInterventionDiscovery

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "perturbation", edge_index)

        self.embed_layer_diseased = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_treated = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)


    def forward(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x = self.get_embeddings(x, batch, topK, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x)


    def get_embeddings(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        if self._mutilate_graph and mutilate_mutations is None:
            raise ValueError("Mutations should not be None in intervention discovery model")

        # Random node initialization
        random_dims = torch.empty(x.shape[0], self.positional_features_dims).to(x.device)
        nn.init.normal_(random_dims)

        # Feature embedding
        x_diseased, _ = self.embed_layer_diseased(x[:, 0].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input['diseased'])
        x_treated, _ = self.embed_layer_treated(x[:, 1].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input['treated'])

        x_j_mask = None
        if self._mutilate_graph:
            x_j_mask = self.mutilate_graph(batch, mutilate_mutations=mutilate_mutations)

        x = self.from_node_to_out(x_diseased, x_treated, batch, random_dims, x_j_mask)

        return x
