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

class SimGCL(nn.Module):
    def __init__(self, num_users, num_items, norm_adj, emb_size, eps, n_layers):
        super(SimGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.eps = eps
        self.hid_dim = emb_size
        self.n_layers = n_layers
        self.sparse_norm_adj = norm_adj
        # self.embedding_dict = self._init_model()
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hid_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hid_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings = torch.cat([self.embedding_user.weight, self.embedding_item.weight], 0)
        embs=[] # 作者说的，要跳过最原始的E_0

        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            embs.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])

        embs = torch.stack(embs,dim=1)
        user_all_embeddings_all_layer, item_all_embeddings_all_layer = torch.split(embs, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings,user_all_embeddings_all_layer,item_all_embeddings_all_layer
