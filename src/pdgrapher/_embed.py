import math

import torch
import torch.nn as nn


class EmbedLayer(nn.Module):

    def __init__(self, num_vars: int, num_features: int, num_categs: int, hidden_dim: int):
        """
        Embedding layer to represent categorical inputs in continuous space.

        Parameters
        ----------
        num_vars : int
                                                    Number of nodes in the graph
        num_categs : int
                                                            Max. number of categories that each variable can take.
        hidden_dim : int
                                                            Output dimensionality of the embedding layer.
        """
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.num_categs = num_categs
        self.num_features = num_features
        # Gene-wise 1/0 learnable embeddings
        # We have num_vars*num_categs*num_features possible embeddings to model.
        self.num_embeds = self.num_vars*self.num_categs*self.num_features
        self.embedding = nn.Embedding(num_embeddings=self.num_embeds, embedding_dim=self.hidden_dim)
        self.embedding.weight.data.mul_(2./math.sqrt(self.num_vars))
        self.bias = nn.Parameter(torch.zeros(self.num_vars, self.hidden_dim*self.num_features))

        # Tensor for mapping each input to its corresponding embedding range in self.embedding
        pos_trans = (torch.arange(self.num_vars*self.num_features, dtype=torch.long) * self.num_categs)
        self.register_buffer("pos_trans", pos_trans, persistent=False)


    def forward(self, x, topK=None, binarize_intervention=False, binarize_input=True, threshold_input=None):
        return self.embed_tensor(x, topK, binarize_intervention, binarize_input, threshold_input)


    def embed_tensor(self, x, topK=None, binarize_intervention=False, binarize_input=True, threshold_input=None):
        '''
        x --> [nnodes*nbatch, num_features]
        '''

        # Number of variables
        pos_trans = self.pos_trans.view(self.num_vars, -1).repeat(int(x.shape[0]/self.num_vars), 1)

        # NEED TO CODE BINARIZATION OF PREDICTED PERTURBATION FLAG HERE USING DIFFERENT TOPK
        if binarize_intervention:
            # Binarize values on U'
            uprime = x.view(-1, self.num_vars).clone()
            sorts = torch.argsort(uprime, dim=1, descending=True)
            row_indices = torch.LongTensor([i for i in range(len(topK)) for e in range(int(topK[i].item()))])
            col_indices = torch.LongTensor([e for i in range(len(topK)) for e in range(int(topK[i].item()))])
            uprime[:, :] = 0
            uprime[row_indices, sorts[row_indices, col_indices]] = 1
            uprime = uprime.view(-1)
            x_bin = uprime.view(-1, 1)
        elif binarize_input:
            difference = x - threshold_input.reshape(1, -1)
            difference = torch.where(difference >= 0, difference.to(torch.double), torch.inf).float()
            percentile = torch.argsort(difference)[:, 0]
            percentile = torch.where(percentile != len(threshold_input)-1, percentile, len(threshold_input)-2)
            x_bin = percentile.view(-1, 1)
        else:
            x_bin = x

        indices = x_bin + pos_trans

        # local embedding
        if binarize_intervention:   # Adding 'x' directly so the gradient can flow through them to model_2
            x_local = (self.embedding(indices.int()) + x.unsqueeze(-1).repeat(1, 1, self.hidden_dim)).view(indices.shape[0], -1)
        else:
            x_local = self.embedding(indices.int()).view(indices.shape[0], -1)

        bias_local = self.bias.repeat(int(x_local.shape[0]/self.num_vars), 1)
        x_local = x_local + bias_local

        return x_local, x_bin
