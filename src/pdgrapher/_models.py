from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._embed import EmbedLayer
from ._torch_geometric import GCNConv

__all__ = ["GCNArgs", "ResponsePredictionModel", "PerturbationDiscoveryModel"]


@dataclass
class GCNArgs():
    """
    Base class for the GCN architecture design.

    Args:
        positional_features_dims (int, optional): Dimensionality of the features. Defaults to 16.
        embedding_layer_dim (int, optional): Dimensionality of the embedding layer. Defaults to 16.
        dim_gnn (int, optional): Size of GNN layer. Defaults to 16.
        num_vars (int, optional): Number of variables in the underlying graph. Defaults to 1.
        n_layers_gnn (int, optional): Number of GNN layers. Defaults to 1.
        n_layers_nn (int, optional): Number of NN layers. Defaults to 2.
    """
    positional_features_dims: int = 16
    embedding_layer_dim: int = 16
    dim_gnn: int = 16
    num_vars: int = 1
    n_layers_gnn: int = 1
    n_layers_nn: int = 2

    # TODO add support for layers of different sizes -> also change in GCNBase class
    # neurons_gnn: list = list
    # neurons_nn: list = list

    @classmethod
    def from_dict(cls, args: Dict[str, int]) -> "GCNArgs":
        """
        Creates a GCNArgs class from provided dictionary. Extra arguments are
        ignored. If some keys are not presented, default values are used.

        Args:
            args (dict[str, int]) Dictionary from which to create this class.

        Returns:
            GCNArgs: instance of self, created from provided dictionary.
        """
        instance = cls(
            positional_features_dims = args.get("positional_features_dims", 16),
            embedding_layer_dim = args.get("embedding_layer_dim", 16),
            dim_gnn = args.get("dim_gnn", 16),
            num_vars = args.get("num_vars", 1),
            n_layers_gnn = args.get("n_layers_gnn", 1),
            n_layers_nn = args.get("n_layers_nn", 2)
        )
        return instance

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class GCNBase(nn.Module):
    """
    Base class for both Perturbation Discovery model and Response Prediction
    model. Contains the underlying structure of each model and some common
    methods. It is not designed to be used directly.
    """

    def __init__(self, args: GCNArgs, out_fun: str, edge_index: torch.Tensor):
        super().__init__()

        self.edge_index = edge_index
        self.positional_features_dims = args.positional_features_dims

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
        for _ in range(args.n_layers_gnn):
            self.bns.append(nn.BatchNorm1d(args.dim_gnn + 2*args.embedding_layer_dim))

        # NN layers
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(args.dim_gnn + 2*args.embedding_layer_dim, args.dim_gnn))
        for _ in range(args.n_layers_nn-1):
            self.mlp.append(nn.Linear(args.dim_gnn, args.dim_gnn))

        self.mlp.append(nn.Linear(args.dim_gnn, args.dim_gnn//2)) # int(args.dim_gnn/2)
        self.mlp.append(nn.Linear(args.dim_gnn//2, 1)) # int(args.dim_gnn/2), args.out_channels

        # Batchnorm MLP
        self.bns_mlp = nn.ModuleList()
        for _ in range(args.n_layers_nn):
            self.bns_mlp.append(nn.BatchNorm1d(args.dim_gnn))
            
        self.bns_mlp.append(nn.BatchNorm1d(args.dim_gnn//2)) # int(args.dim_gnn/2)

        # Output function
        out_fun_selector = {'response': lambda x: x, 'perturbation': lambda x: x}
        self.out_fun = out_fun_selector.get(out_fun, lambda x: x)

        # dictionary storing, for each node, its place in edge_index where there
        # is an edge incoming to it (excluding self loops)
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
        self.dictionary_node_to_edge_index_position = defaultdict(list)
        for i in range(self.edge_index.size(1)):
            node = self.edge_index[1, i].item()
            if node == self.edge_index[0, i].item():
                continue
            self.dictionary_node_to_edge_index_position[node].append(i)



class ResponsePredictionModel(GCNBase):
    """
    Class that represents Response Prediction model.
    """

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "response", edge_index)

        self.num_nodes = args.num_vars
        self.embed_layer_pert = EmbedLayer(args.num_vars, num_features=1, num_categs=2, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_ge = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)
        self.positional_embeddings = nn.Embedding(args.num_vars, self.positional_features_dims)
        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=1.0)


    def forward(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x, in_x_binarized = self._get_embeddings(x, batch, topK, binarize_intervention, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x), in_x_binarized


    def _get_embeddings(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
    

        # Positional encodings
        pos_embeddings = self.positional_embeddings(torch.arange(self.num_nodes).to(x.device))
        random_dims = pos_embeddings.repeat(int(x.shape[0] / self.num_nodes), 1)


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


class PerturbationDiscoveryModel(GCNBase):
    """
    Class that represents Perturbation Discovery model.
    """

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "perturbation", edge_index)
        
        self.num_nodes = args.num_vars
        self.embed_layer_diseased = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_treated = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)
        self.positional_embeddings = nn.Embedding(args.num_vars, self.positional_features_dims)
        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=1.0)

    def forward(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x = self._get_embeddings(x, batch, topK, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x)


    def _get_embeddings(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        if self._mutilate_graph and mutilate_mutations is None:
            raise ValueError("Mutations should not be None in intervention discovery model")


        # Positional encodings
        pos_embeddings = self.positional_embeddings(torch.arange(self.num_nodes).to(x.device))
        random_dims = pos_embeddings.repeat(int(x.shape[0] / self.num_nodes), 1)


        # Feature embedding
        x_diseased, _ = self.embed_layer_diseased(x[:, 0].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input["diseased"])
        x_treated, _ = self.embed_layer_treated(x[:, 1].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input["treated"])

        x_j_mask = None
        if self._mutilate_graph:
            x_j_mask = self.mutilate_graph(batch, mutilate_mutations=mutilate_mutations)

        x = self.from_node_to_out(x_diseased, x_treated, batch, random_dims, x_j_mask)

        return x











##################
#OLD VERSION




class ResponsePredictionModelOld(GCNBase):
    """
    Class that represents Response Prediction model.
    """

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "response", edge_index)

        self.embed_layer_pert = EmbedLayer(args.num_vars, num_features=1, num_categs=2, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_ge = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)


    def forward(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x, in_x_binarized = self._get_embeddings(x, batch, topK, binarize_intervention, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x), in_x_binarized


    def _get_embeddings(self, x, batch, topK=None, binarize_intervention=False, mutilate_mutations=None, threshold_input=None):
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


class PerturbationDiscoveryModelOld(GCNBase):
    """
    Class that represents Perturbation Discovery model.
    """

    def __init__(self, args: GCNArgs, edge_index: torch.Tensor):
        super().__init__(args, "perturbation", edge_index)

        self.embed_layer_diseased = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)
        self.embed_layer_treated = EmbedLayer(args.num_vars, num_features=1, num_categs=500, hidden_dim=args.embedding_layer_dim)


    def forward(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        '''
        GCN model adapted to use 1 single edge_index for all samples in the batch
        and x_j_mask to mask messages x_j (acting as mutilation procedure)
        '''

        x = self._get_embeddings(x, batch, topK, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)

        return self.out_fun(x)


    def _get_embeddings(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        if self._mutilate_graph and mutilate_mutations is None:
            raise ValueError("Mutations should not be None in intervention discovery model")

        # Random node initialization
        random_dims = torch.empty(x.shape[0], self.positional_features_dims).to(x.device)
        nn.init.normal_(random_dims)

        # Feature embedding
        x_diseased, _ = self.embed_layer_diseased(x[:, 0].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input["diseased"])
        x_treated, _ = self.embed_layer_treated(x[:, 1].view(-1, 1), topK=None, binarize_input=True, threshold_input=threshold_input["treated"])

        x_j_mask = None
        if self._mutilate_graph:
            x_j_mask = self.mutilate_graph(batch, mutilate_mutations=mutilate_mutations)

        x = self.from_node_to_out(x_diseased, x_treated, batch, random_dims, x_j_mask)

        return x
