import torch
from torch import nn
from utility.parser import parse_args
args = parse_args()

class NeuMF(nn.Module):

    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_id_embedding = nn.Embedding(n_users + 1, args.embed_size)
        self.photo_id_embedding = nn.Embedding(n_items + 1, args.embed_size)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.photo_id_embedding.weight)

        self.mlp_user = nn.ModuleList()
        self.mlp_item = nn.ModuleList()

        self.mlp_user.append(nn.Linear(args.embed_size, args.embed_size * 2))
        self.mlp_user.append(nn.ReLU())
        self.mlp_user.append(nn.Linear(args.embed_size * 2, args.embed_size))

        self.mlp_item.append(nn.Linear(args.embed_size, args.embed_size * 2))
        self.mlp_item.append(nn.ReLU())
        self.mlp_item.append(nn.Linear(args.embed_size * 2, args.embed_size))

    def forward(self, x):
        user_id_top = self.user_id_embedding.weight
        item_id_top = self.photo_id_embedding.weight
        for layer in self.mlp_user:
            user_id_top = layer(user_id_top) * x

        for layer in self.mlp_item:
            item_id_top = layer(item_id_top) * x

        return user_id_top, item_id_top
