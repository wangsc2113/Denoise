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

from model import LightGCN
from utility.parser import parse_args
from utility.batch_test import *
from utility.logging import Logger
from torch.utils.tensorboard import SummaryWriter

args = parse_args()

class Trainer(object):
    def __init__(self, data_config, n_users, n_items, graph):
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
 
        # self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset)) # [[], [], ...]
        # self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset)) # [[], [], ...]

        # if args.dataset == 'tiktok':
        #     self.audio_feats = np.load(args.data_path + '{}/audio_feat.npy'.format(args.dataset))
        # else:
        #     self.audio_feats = None

        self.n_users, self.n_items = n_users, n_items
        self.graph = self._convert_sp_mat_to_sp_tensor(graph).cuda()
        self.model = LightGCN(self.n_users, self.n_items, self.emb_dim, n_layers = 3, graph = self.graph)
        self.model = self.model.cuda()

        self.optimizer_D = optim.Adam([{'params':self.model.parameters()},], lr=self.lr)  
        # self.scheduler_D = self.set_lr_scheduler()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D  

    def test(self, users_to_test, users_to_test_cold, users_to_test_warm, is_val):
        self.model.eval()
        with torch.no_grad():
            users_emb, items_emb = self.model()
        result = test_torch(users_to_test, users_emb, items_emb, is_val)
        result_cold = test_torch(users_to_test_cold, users_emb, items_emb, is_val)
        result_warm = test_torch(users_to_test_warm, users_emb, items_emb, is_val)
        return result, result_cold, result_warm


    def train(self):
        # self.criterion = nn.MSELoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()

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
            loss, bpr_loss, reg_loss, ce_loss = 0., 0., 0., 0.

            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.

            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                all_users, all_items = self.model()
                # print (len(users), len(pos_items), len(neg_items))
                # print (users.size(), pos_items.size(), neg_items.size())
                users_emb = F.normalize(all_users[users], p=2, dim=1)
                pos_emb = F.normalize(all_items[pos_items], p=2, dim=1)
                neg_emb = F.normalize(all_items[neg_items], p=2, dim=1)
                userEmb0 = self.model.embedding_user(torch.LongTensor(users).cuda())
                posEmb0 = self.model.embedding_item(torch.LongTensor(pos_items).cuda())
                negEmb0 = self.model.embedding_item(torch.LongTensor(neg_items).cuda())

                batch_reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/float(len(users))
                pos_scores = torch.mul(users_emb, pos_emb)
                pos_scores = torch.sum(pos_scores, dim=1)
                neg_scores = torch.mul(users_emb, neg_emb)
                neg_scores = torch.sum(neg_scores, dim=1)
                # print ('pos_scores.size, neg_scores.size: ', pos_scores.size(), neg_scores.size())

                # lightgcn，计算bpr loss
                batch_bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

                # lightgcn，计算ce loss
                # logits = torch.concat([pos_scores.unsqueeze(dim = 1), neg_scores.unsqueeze(dim = 1)], dim = 1)
                # labels = torch.zeros(size = (pos_scores.size(0),)).to(torch.int64).cuda()
                # batch_ce_loss = self.criterion(logits, labels)

                batch_reg_loss = batch_reg_loss * self.decay
                batch_loss = batch_bpr_loss + batch_reg_loss
                # batch_loss = batch_ce_loss + batch_reg_loss

                self.optimizer_D.zero_grad()
                batch_loss.backward()
                self.optimizer_D.step()

                loss += float(batch_loss)
                bpr_loss += float(batch_bpr_loss)
                # ce_loss += float(batch_ce_loss)
                reg_loss += float(batch_reg_loss)

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, bpr_loss, reg_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            users_to_cold_test = list(data_generator.cold_test_set.keys())
            users_to_warm_test = list(data_generator.warm_test_set.keys())

            test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, is_val = False)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(test_ret['recall'].data)
            pre_loger.append(test_ret['precision'].data)
            ndcg_loger.append(test_ret['ndcg'].data)
            hit_loger.append(test_ret['hit_ratio'].data)

            tags = ["recall", "precision", "ndcg"]

            # if args.verbose > 0:
            #     perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f], ' \
            #                'precision=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]' % \
            #                (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, cl_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4], ret['recall'][5], ret['recall'][6], ret['recall'][7], ret['recall'][8], ret['recall'][9], ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['precision'][4], ret['precision'][5], ret['precision'][6], ret['precision'][7], ret['precision'][8], ret['precision'][9], ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4], ret['ndcg'][5], ret['ndcg'][6], ret['ndcg'][7], ret['ndcg'][8], ret['ndcg'][9])
            #     self.logger.logging(perf_str)

            self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_ret['recall'][-2], test_ret['precision'][-2], test_ret['ndcg'][-2]))
            self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_cold_ret['recall'][-2], test_cold_ret['precision'][-2], test_cold_ret['ndcg'][-2]))
            self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_warm_ret['recall'][-2], test_warm_ret['precision'][-2], test_warm_ret['ndcg'][-2]))
            final_recall.append(test_ret['recall'][-1])
            final_precison.append(test_ret['precision'][-1])
            final_ndcg.append(test_ret['ndcg'][-1])

            final_c_recall.append(test_cold_ret['recall'][-1])
            final_c_precison.append(test_cold_ret['precision'][-1])
            final_c_ndcg.append(test_cold_ret['ndcg'][-1])

            final_w_recall.append(test_warm_ret['recall'][-1])
            final_w_precison.append(test_warm_ret['precision'][-1])
            final_w_ndcg.append(test_warm_ret['ndcg'][-1])
            # if ret['recall'][-1] > best_recall:
            #     best_recall = ret['recall'][-1]
            #     test_ret = self.test(users_to_test, is_val=False)
            #     self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], test_ret['recall'][-2], test_ret['precision'][-2], test_ret['ndcg'][-2]))
                # self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_cold_ret['recall'][-1], test_cold_ret['precision'][-1], test_cold_ret['ndcg'][-1]))
                # self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_warm_ret['recall'][-1], test_warm_ret['precision'][-1], test_warm_ret['ndcg'][-1]))
            #     stopping_step = 0
            # elif stopping_step < args.early_stopping_patience:
            #     stopping_step += 1
            #     self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            # else:
            #     self.logger.logging('#####Early stop! #####')
            #     break
        self.logger.logging('#####final result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], max(final_recall),  max(final_precison), max(final_ndcg)))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], min(final_c_recall), min(final_c_precison), min(final_c_ndcg)))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], max(final_w_recall), max(final_w_precison), max(final_w_ndcg)))

        index = torch.argmax(torch.tensor(final_c_recall))
        self.logger.logging('#####final soft result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_recall[index],  final_precison[index], final_ndcg[index]))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_c_recall[index], final_c_precison[index], final_c_ndcg[index]))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_w_recall[index], final_w_precison[index], final_w_ndcg[index]))

        index = torch.argmax(torch.tensor(final_recall))
        self.logger.logging('#####final hard result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_recall[index],  final_precison[index], final_ndcg[index]))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_c_recall[index], final_c_precison[index], final_c_ndcg[index]))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_w_recall[index], final_w_precison[index], final_w_ndcg[index]))


        return best_recall, run_time 

    def transfer(self, pid_emb, sid_emb, vv_rate):
        loss_hot = self.criterion(sid_emb, pid_emb.detach()).mean(dim=1, keepdim=True)
        loss_cold = self.criterion(pid_emb, sid_emb.detach()).mean(dim=1, keepdim=True)
        #loss = vv_rate * loss_hot + (1 - vv_rate) * loss_cold
        loss = loss_cold
        return loss.mean()

    def bpr_loss(self, users, pos_items, neg_items, u_v_emb, u_t_emb, u_a_emb):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        # cl_loss = 0.0
        
        if args.dataset == 'tiktok':
            u_t_emb_transposed = torch.transpose(u_t_emb, 0, 1)
            u_a_emb_transposed = torch.transpose(u_a_emb, 0, 1)
            
            orthogonal_inner_product_vt = torch.matmul(u_v_emb, u_t_emb_transposed)
            orthogonal_inner_product_va = torch.matmul(u_v_emb, u_a_emb_transposed)
            orthogonal_inner_product_ta = torch.matmul(u_t_emb, u_a_emb_transposed)
            
            cl_loss = torch.sum(torch.square(orthogonal_inner_product_vt)) + torch.sum(torch.square(orthogonal_inner_product_va)) + torch.sum(torch.square(orthogonal_inner_product_ta))
        else:
            u_t_emb_transposed = torch.transpose(u_t_emb, 0, 1)
            orthogonal_inner_product = torch.matmul(u_v_emb, u_t_emb_transposed) 
            cl_loss =  torch.sum(torch.square(orthogonal_inner_product))
        return mf_loss, emb_loss, reg_loss, args.alpha * cl_loss

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
    graph, _, _ = data_generator.get_adj_mat()

    trainer = Trainer(config, n_users, n_items, graph)
    trainer.train()
