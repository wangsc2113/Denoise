import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, hid_dim, n_layers, graph):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.Graph = graph
        self.linear = nn.Linear(in_features = hid_dim, out_features = hid_dim)
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hid_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hid_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1.414)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1.414)
        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_item.weight, std=0.1)
        

    def forward(self):
        # all_users = self.layernorm(self.embedding_user.weight)
        # all_items = self.layernorm(self.embedding_item.weight)
        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight
        all_emb = torch.cat([all_users, all_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            # all_emb = self.linear(all_emb)
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim = 1)
        light_out = torch.mean(embs, dim = 1)
        users_emb, items_emb = torch.split(light_out, [self.num_users, self.num_items])
        return users_emb, items_emb