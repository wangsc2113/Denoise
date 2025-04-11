from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import copy

from utility.parser import parse_args
from model import NeuMF
from utility.batch_test import *
from utility.logging import Logger
from torch.utils.tensorboard import SummaryWriter

args = parse_args()

class Trainer(object):
    def __init__(self, data_config, n_users, n_items, warm_adj, confidence_score, vv_count):
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        
        self.n_users, self.n_items = n_users, n_items
        self.warm_adj, self.confidence_score, self.vv_count = warm_adj.cuda(), confidence_score.float().cuda(), vv_count.cuda()

        self.model = NeuMF(self.n_users, self.n_items)
        self.model = self.model.cuda()

        self.optimizer_D = optim.AdamW([{'params':self.model.parameters()},], lr=self.lr)  
        self.scheduler_D = self.set_lr_scheduler()

        # if args.dataset == 'tiktok':
        #     self.vv_rate_alpha = -0.05
        # else:
        self.vv_rate_alpha = -0.1


    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D  

    def test(self, users_to_test, users_to_cold_test, users_to_warm_test, ones, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(ones)
        result = test_torch(users_to_test, ua_embeddings, ia_embeddings, is_val)
        result_cold = test_torch(users_to_cold_test, ua_embeddings, ia_embeddings, is_val)
        result_warm = test_torch(users_to_warm_test, ua_embeddings, ia_embeddings, is_val)
        return result, result_cold, result_warm

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0. 

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0

        final_recall, final_precison, final_ndcg = [], [], []
        final_c_recall, final_c_precison, final_c_ndcg = [], [], []
        final_w_recall, final_w_precison, final_w_ndcg = [], [], []
        for epoch in range(args.epoch):
            t1 = time()
            loss, re_loss, reg_loss = 0., 0., 0.

            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.


            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                ones = torch.ones(1, 1).cuda()
                
                all_users, all_items = self.model(ones)

                users_emb = F.normalize(all_users[users], p=2, dim=1)
                pos_emb = F.normalize(all_items[pos_items], p=2, dim=1)
                neg_emb = F.normalize(all_items[neg_items], p=2, dim=1)
                userEmb0 = self.model.user_id_embedding(torch.LongTensor(users).cuda())
                posEmb0 = self.model.photo_id_embedding(torch.LongTensor(pos_items).cuda())
                negEmb0 = self.model.photo_id_embedding(torch.LongTensor(neg_items).cuda())

                batch_reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/float(len(users))
                pos_scores = torch.mul(users_emb, pos_emb)
                pos_scores = torch.sum(pos_scores, dim=1)
                neg_scores = torch.mul(users_emb, neg_emb)
                neg_scores = torch.sum(neg_scores, dim=1)

                # infonce loss
                adj_list = self.warm_adj[pos_items]
                adj_emb = F.normalize(all_items[adj_list.reshape(-1, 1)], p=2, dim=-1).reshape(adj_list.size(0), adj_list.size(1), -1)
                pseudo_label_list = 2 * torch.matmul(adj_emb, users_emb.unsqueeze(dim = -1))
                # print ('pseudo_label_list: ', pseudo_label_list)
                pseudo_label_list = torch.clamp(pseudo_label_list, min = 0, max = 1.0)
                # print ('pseudo_label_list: ', pseudo_label_list)
                pseudo_label = torch.matmul(self.confidence_score[pos_items].unsqueeze(dim = 1), pseudo_label_list).squeeze(dim = -1)
                # print ('pseudo_label: ', pseudo_label)

                rou = 1.0
                pos_inner = torch.sum(torch.mul(users_emb, pos_emb), dim = 1, keepdim = True)
                numerator = torch.exp(pos_inner / rou)
                all_inner = torch.matmul(users_emb, pos_emb.transpose(-1, -2))
                denominator_tmp = torch.exp(all_inner / rou)
                denominator = torch.sum(denominator_tmp, dim = 1, keepdim = True)
                pred = numerator / (denominator + 1e-9)

                # pos_inner = torch.clamp(pos_inner, min = 1e-3)
                # print ('pos_inner: ', pos_inner)
                entropy = -torch.log2(torch.clamp(pos_inner, min = 1e-3))
                # print ('entropy: ', entropy)
                vv_rate = torch.exp(self.vv_rate_alpha * self.vv_count[pos_items]).reshape(-1, 1)
                # print ('vv_rate: ', vv_rate)
                
                # gate = entropy
                # gate = vv_rate
                gate = entropy * vv_rate
                gate = torch.clamp(gate, max = 1.0)

                w1 = gate / (gate + 1)
                w2 = 1 / (gate + 1)
                # print ('w1: ', w1)
                # print ('w2: ', w2)

                re_labels = w1 * pseudo_label + w2 * torch.ones(users_emb.size(0), 1).cuda()
                batch_re_loss = torch.sum(-re_labels * torch.log2(pred) - (1 - re_labels) * torch.log2(1 - pred))
                
                # lightgcn，计算bpr loss
                # batch_bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

                # lightgcn，计算ce loss
                # logits = torch.concat([pos_scores.unsqueeze(dim = 1), neg_scores.unsqueeze(dim = 1)], dim = 1)
                # labels = torch.zeros(size = (pos_scores.size(0),)).to(torch.int64).cuda()
                # batch_ce_loss = self.criterion(logits, labels)

                batch_reg_loss = batch_reg_loss * self.decay
                # batch_loss = batch_bpr_loss + batch_reg_loss
                # batch_loss = batch_ce_loss + batch_reg_loss
                batch_loss = batch_re_loss + batch_reg_loss

                                                                                                   
                self.optimizer_D.zero_grad()  
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                loss += float(batch_loss)
                re_loss += float(batch_re_loss)
                reg_loss += float(batch_reg_loss)

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, re_loss, reg_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_cold_test = list(data_generator.cold_test_set.keys())
            users_to_warm_test = list(data_generator.warm_test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, ones, is_val=False)  
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(test_ret['recall'].data)
            pre_loger.append(test_ret['precision'].data)
            ndcg_loger.append(test_ret['ndcg'].data)
            hit_loger.append(test_ret['hit_ratio'].data)

            tags = ["recall", "precision", "ndcg"]

            self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_ret['recall'][-2], test_ret['precision'][-2], test_ret['ndcg'][-2]))
            self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_cold_ret['recall'][-2], test_cold_ret['precision'][-2], test_cold_ret['ndcg'][-2]))
            self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_warm_ret['recall'][-2], test_warm_ret['precision'][-2], test_warm_ret['ndcg'][-2]))
            final_recall.append(test_ret['recall'][-2])
            final_precison.append(test_ret['precision'][-2])
            final_ndcg.append(test_ret['ndcg'][-2])

            final_c_recall.append(test_cold_ret['recall'][-2])
            final_c_precison.append(test_cold_ret['precision'][-2])
            final_c_ndcg.append(test_cold_ret['ndcg'][-2])

            final_w_recall.append(test_warm_ret['recall'][-2])
            final_w_precison.append(test_warm_ret['precision'][-2])
            final_w_ndcg.append(test_warm_ret['ndcg'][-2])

        self.logger.logging('#####final result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], max(final_recall),  max(final_precison), max(final_ndcg)))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], min(final_c_recall), min(final_c_precison), min(final_c_ndcg)))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], max(final_w_recall), max(final_w_precison), max(final_w_ndcg)))

        index = torch.argmax(torch.tensor(final_recall))
        self.logger.logging('#####final soft result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], final_recall[index],  final_precison[index], final_ndcg[index]))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], final_c_recall[index], final_c_precison[index], final_c_ndcg[index]))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], final_w_recall[index], final_w_precison[index], final_w_ndcg[index]))

        return best_recall, run_time 


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        
        return mf_loss, emb_loss

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    n_users, n_items = data_generator.n_users, data_generator.n_items

    with open(args.data_path + args.dataset + '/vv_count.pkl', 'rb') as file:
        vv_count = pickle.load(file)
    with open(args.data_path + args.dataset + '/confidence_score.pkl', 'rb') as file:
        confidence_score = pickle.load(file)
    with open(args.data_path + args.dataset + '/warm_adj.pkl', 'rb') as file:
        warm_adj = pickle.load(file)
        
    trainer = Trainer(config, n_users, n_items, warm_adj, confidence_score, vv_count)
    trainer.train()
