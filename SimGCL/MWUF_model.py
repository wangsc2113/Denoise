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

class Transformer(nn.Module):
    def __init__(self, col, nh = 4, action_item_size = 64, att_emb_size = 16):
        super(Transformer, self).__init__()
        self.nh = nh
        self.att_emb_size = att_emb_size
        self.Q = nn.Parameter(torch.empty(col, att_emb_size * nh))
        self.K = nn.Parameter(torch.empty(action_item_size, att_emb_size * nh))
        self.V = nn.Parameter(torch.empty(action_item_size, att_emb_size * nh))
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(self, query_input, action_list_input):
        querys = torch.tensordot(query_input, self.Q, dims=([-1], [0]))
        keys = torch.tensordot(action_list_input, self.K, dims=([-1], [0]))
        values = torch.tensordot(action_list_input, self.V, dims=([-1], [0]))

        querys = torch.stack(torch.chunk(querys, self.nh, dim=2)) # [-2, 1]
        keys = torch.stack(torch.chunk(keys, self.nh, dim=2))
        values = torch.stack(torch.chunk(values, self.nh, dim=2))

        inner_product = torch.matmul(querys, keys.transpose(-2, -1)) / 8.0
        normalized_att_scores = torch.nn.functional.softmax(inner_product, dim=-1)
        result = torch.matmul(normalized_att_scores, values)
        result = result.permute(1, 2, 0, 3)

        mha_result = result.reshape((query_input.shape[0], self.nh * self.att_emb_size))

        return mha_result

class SimGCL(nn.Module):
    def __init__(self, num_users, num_items, norm_adj, emb_size, eps, n_layers, image_feats, text_feats):
        super(SimGCL, self).__init__()
        self.photo_one_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

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

        self.linear_onehop = nn.Linear(args.embed_size, args.embed_size, bias=False)
        self.linear_mul1 = nn.Linear(image_feats.shape[1]+text_feats.shape[1], args.embed_size*4, bias=False)
        self.linear_mul2 = nn.Linear(args.embed_size*4, args.embed_size, bias=False)
        nn.init.xavier_uniform_(self.linear_onehop.weight)
        nn.init.xavier_uniform_(self.linear_mul1.weight)
        nn.init.xavier_uniform_(self.linear_mul2.weight)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items, self.emb_size))),
        })
        return embedding_dict

    def forward(self, photo_one_hop, perturbed=False):
        photo_one_hop_embeddings = self.embedding_user(photo_one_hop)
        photo_query = torch.reshape(self.embedding_item.weight.detach(), (-1, 1, 64))
        photo_one_hop_emb = self.photo_one_hop_transformer(photo_query, photo_one_hop_embeddings) # [photo_num, 64]

        item_features_emb = torch.cat((self.image_embedding.weight, self.text_embedding.weight), dim = -1)

        item_fea_hidden = self.linear_mul1(item_features_emb)
        itea_fea_emb = self.linear_mul2(item_fea_hidden)
        item_one_hop = self.linear_onehop(photo_one_hop_emb)

        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight * itea_fea_emb + item_one_hop

        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings = torch.cat([all_users, all_items], 0)
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
