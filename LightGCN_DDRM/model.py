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

import math
import time
import torch.nn.functional as F
import world

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




class DDRM(nn.Module):
    def __init__(self, num_users, num_items, hid_dim, n_layers, graph):
        super(DDRM, self).__init__()
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
        
    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt

    def computer(self):
        # all_users = self.layernorm(self.embedding_user.weight)
        # all_items = self.layernorm(self.embedding_item.weight)
        # all_users = self.embedding_user.weight
        # all_items = self.embedding_item.weight
        all_users = F.normalize(self.embedding_user.weight, p=2.0, dim=1, eps=1e-12, out=None)
        all_items = F.normalize(self.embedding_item.weight, p=2.0, dim=1, eps=1e-12, out=None)
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

    def transfer(self, user, pos, user_reverse_model, item_reverse_model, diff_model):
        users_emb, items_emb = self.computer()
        ori_user_emb = users_emb[user]
        ori_item_emb = items_emb[pos]
        ori_user_emb = F.normalize(ori_user_emb, p=2.0, dim=1, eps=1e-12, out=None)
        ori_item_emb = F.normalize(ori_item_emb, p=2.0, dim=1, eps=1e-12, out=None)

        ### add noise to user and item
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(ori_user_emb, ori_item_emb, diff_model)
        noise_user_emb = F.normalize(noise_user_emb, p=2.0, dim=1, eps=1e-12, out=None)
        noise_item_emb = F.normalize(noise_item_emb, p=2.0, dim=1, eps=1e-12, out=None)
        ### reverse
        user_model_output = user_reverse_model(noise_user_emb, ori_item_emb, ts)
        item_model_output = item_reverse_model(noise_item_emb, ori_user_emb, ts)
        user_model_output = F.normalize(user_model_output, p=2.0, dim=1, eps=1e-12, out=None)
        item_model_output = F.normalize(item_model_output, p=2.0, dim=1, eps=1e-12, out=None)

        ### get recons loss
        user_recons = diff_model.get_reconstruct_loss(ori_user_emb, user_model_output, pt)
        item_recons = diff_model.get_reconstruct_loss(ori_item_emb, item_model_output, pt)
        recons_loss = (user_recons + item_recons) / 2

        users_emb = users_emb.clone()
        items_emb = items_emb.clone()
        users_emb.index_copy_(dim = 0, index = user, source = user_model_output)
        items_emb.index_copy_(dim = 0, index = pos, source = item_model_output)

        ### update the batch user and item emb
        # return user_model_output, item_model_output, recons_loss, items_emb
        return users_emb, items_emb, recons_loss
    
    def getEmbedding(self, users, pos_items, neg_items, user_reverse_model, item_reverse_model, diff_model):
        # users_emb, pos_emb, recons_loss, all_items = self.computer(users, pos_items)
        all_users, all_items, recons_loss = self.transfer(users, pos_items, user_reverse_model, item_reverse_model, diff_model)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, recons_loss , all_users, all_items 

    def bpr_loss(self, users, pos, neg, user_reverse_model, item_reverse_model, diff_model):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0,
            reconstruct_loss, all_users, all_items) = self.getEmbedding(users.long(), pos.long(), neg.long(), user_reverse_model, item_reverse_model, diff_model)

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.nn.functional.softplus(neg_scores - pos_scores)
        return loss, reg_loss, reconstruct_loss, pos_scores, all_users, all_items

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    # def forward(self, x, timesteps):
    #     time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
    #     emb = self.emb_layer(time_emb)
    #     if self.norm:
    #         x = F.normalize(x)
    #     x = self.drop(x)

    #     h = torch.cat([x, emb], dim=-1)

    #     for i, layer in enumerate(self.in_layers):
    #         h = layer(h)
    #         h = torch.tanh(h)
    #     for i, layer in enumerate(self.out_layers):
    #         h = layer(h)
    #         if i != len(self.out_layers) - 1:
    #             h = torch.tanh(h)
    #     return h

    def forward(self, noise_emb, con_emb, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if world.config['act'] == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif world.config['act'] == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif world.config['act'] == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if world.config['act'] == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif world.config['act'] == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif world.config['act'] == 'relu':
                    all_emb = F.relu(all_emb)
        return all_emb

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding