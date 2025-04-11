import torch
from torch import nn
from utility.parser import parse_args
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

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, image_feats, text_feats):
        super().__init__()
        self.photo_one_hop_transformer = Transformer(col = 64, nh = 4, action_item_size = 64, att_emb_size = 16)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.user_id_embedding = nn.Embedding(n_users, args.embed_size)
        self.photo_id_embedding = nn.Embedding(n_items, args.embed_size)
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

        self.linear_onehop = nn.Linear(args.embed_size, args.embed_size, bias=False)
        self.linear_mul1 = nn.Linear(image_feats.shape[1]+text_feats.shape[1], args.embed_size*4, bias=False)
        self.linear_mul2 = nn.Linear(args.embed_size*4, args.embed_size, bias=False)
        nn.init.xavier_uniform_(self.linear_onehop.weight)
        nn.init.xavier_uniform_(self.linear_mul1.weight)
        nn.init.xavier_uniform_(self.linear_mul2.weight)

    def forward(self, x, photo_one_hop):
        photo_one_hop_embeddings = self.user_id_embedding(photo_one_hop)
        # print ('photo_one_hop_embeddings.size: ', photo_one_hop_embeddings.size())
        photo_query = torch.reshape(self.photo_id_embedding.weight.detach(), (-1, 1, 64))
        # print ('photo_query.size: ', photo_query.size())
        photo_one_hop_emb = self.photo_one_hop_transformer(photo_query, photo_one_hop_embeddings) # [photo_num, 64]

        item_features_emb = torch.cat((self.image_embedding.weight, self.text_embedding.weight), dim = -1)

        item_fea_hidden = self.linear_mul1(item_features_emb)
        itea_fea_emb = self.linear_mul2(item_fea_hidden)
        item_one_hop = self.linear_onehop(photo_one_hop_emb)

        user_id_top = self.user_id_embedding.weight
        item_id_top = self.photo_id_embedding.weight * itea_fea_emb + item_one_hop
        # user_id_top = self.user_id_embedding.weight
        # item_id_top = self.photo_id_embedding.weight

        for layer in self.mlp_user:
            user_id_top = layer(user_id_top) * x

        for layer in self.mlp_item:
            item_id_top = layer(item_id_top) * x

        return user_id_top, item_id_top
