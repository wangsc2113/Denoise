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

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, hid_dim, n_layers, graph, image_feats, text_feats):
        super(LightGCN, self).__init__()
        self.photo_one_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.num_users = num_users
        self.num_items = num_items
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.Graph = graph
        self.linear = nn.Linear(in_features = hid_dim, out_features = hid_dim)
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hid_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hid_dim)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1.414)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.linear_onehop = nn.Linear(args.embed_size, args.embed_size, bias=False)
        self.linear_mul1 = nn.Linear(image_feats.shape[1]+text_feats.shape[1], args.embed_size*4, bias=False)
        self.linear_mul2 = nn.Linear(args.embed_size*4, args.embed_size, bias=False)
        nn.init.xavier_uniform_(self.linear_onehop.weight)
        nn.init.xavier_uniform_(self.linear_mul1.weight)
        nn.init.xavier_uniform_(self.linear_mul2.weight)
        

    def forward(self, photo_one_hop):
        photo_one_hop_embeddings = self.embedding_user(photo_one_hop)
        photo_query = torch.reshape(self.embedding_item.weight.detach(), (-1, 1, 64))
        photo_one_hop_emb = self.photo_one_hop_transformer(photo_query, photo_one_hop_embeddings) # [photo_num, 64]

        item_features_emb = torch.cat((self.image_embedding.weight, self.text_embedding.weight), dim = -1)

        item_fea_hidden = self.linear_mul1(item_features_emb)
        itea_fea_emb = self.linear_mul2(item_fea_hidden)
        item_one_hop = self.linear_onehop(photo_one_hop_emb)

        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight * itea_fea_emb + item_one_hop

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