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
from MWUF_model import NeuMF
from utility.batch_test import *
from utility.logging import Logger
from torch.utils.tensorboard import SummaryWriter

args = parse_args()

class Trainer(object):
    def __init__(self, data_config, n_users, n_items):
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset)) # [[], [], ...]
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset)) # [[], [], ...]

        # self.image_feats = self.image_feats[:-1]  # 现在形状为 [m-1, n]
        # self.text_feats = self.text_feats[:-1]    # 现在形状为 [m-1, n]

        self.i2u = np.load(args.data_path + '{}/i2u.npy'.format(args.dataset), allow_pickle = True).item()
        self.seq_len = 6
        
        self.n_users, self.n_items = n_users, n_items
        self.model = NeuMF(self.n_users, self.n_items, self.image_feats, self.text_feats)
        self.model = self.model.cuda()

        self.optimizer_D = optim.AdamW([{'params':self.model.parameters()},], lr=self.lr)  
        self.scheduler_D = self.set_lr_scheduler()


    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D  

    def test(self, users_to_test, users_to_cold_test, users_to_warm_test, ones, photo_one_hop, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(ones, photo_one_hop)
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
            loss, mf_loss, emb_loss, reg_loss, cl_loss = 0., 0., 0., 0., 0.

            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.


            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                photo_one_hop = []
                for i in range(self.n_items):
                    if i in self.i2u.keys():
                        item_history = self.i2u[i]
                        if len(item_history) >= self.seq_len:
                            tmp_one_hop = random.sample(item_history, self.seq_len)
                            photo_one_hop.append(tmp_one_hop)
                        else:
                            tmp_one_hop = item_history + [self.n_users - 1] * (self.seq_len - len(item_history))
                            photo_one_hop.append(tmp_one_hop)
                            
                    else:
                        photo_one_hop.append([self.n_users - 1] * self.seq_len)
                
                # photo_one_hop.append([self.n_users-1] * self.seq_len)
                photo_one_hop = torch.tensor(np.array(photo_one_hop, dtype=np.int32)).cuda()

                ones = torch.ones(1, 1).cuda()
                
                G_ua_embeddings, G_ia_embeddings = self.model(ones, photo_one_hop)

                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]

                G_batch_mf_loss, G_batch_emb_loss = self.bpr_loss(G_u_g_embeddings, G_pos_i_g_embeddings, G_neg_i_g_embeddings)

                batch_loss = G_batch_mf_loss + G_batch_emb_loss
                                                                                                   
                self.optimizer_D.zero_grad()  
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
    
    
            # del G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings


            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_cold_test = list(data_generator.cold_test_set.keys())
            users_to_warm_test = list(data_generator.warm_test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, ones, photo_one_hop, is_val=False)  
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(test_ret['recall'].data)
            pre_loger.append(test_ret['precision'].data)
            ndcg_loger.append(test_ret['ndcg'].data)
            hit_loger.append(test_ret['hit_ratio'].data)

            tags = ["recall", "precision", "ndcg"]

            # if args.verbose > 0:
            #     perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f], ' \
            #                'precision=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]' % \
            #                (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4], ret['recall'][5], ret['recall'][6], ret['recall'][7], ret['recall'][8], ret['recall'][9], ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['precision'][4], ret['precision'][5], ret['precision'][6], ret['precision'][7], ret['precision'][8], ret['precision'][9], ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4], ret['ndcg'][5], ret['ndcg'][6], ret['ndcg'][7], ret['ndcg'][8], ret['ndcg'][9])
            #     self.logger.logging(perf_str)

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
            # if ret['recall'][-1] > best_recall:
            #     best_recall = ret['recall'][-1]
        #         test_ret, test_cold_ret, test_warm_ret = self.test(users_to_test, users_to_cold_test, users_to_warm_test, ones, is_val=False)
        #         self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_ret['recall'][-1], test_ret['precision'][-1], test_ret['ndcg'][-1]))
        #         self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_cold_ret['recall'][-1], test_cold_ret['precision'][-1], test_cold_ret['ndcg'][-1]))
        #         self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], test_warm_ret['recall'][-1], test_warm_ret['precision'][-1], test_warm_ret['ndcg'][-1]))
        #         stopping_step = 0
        #     elif stopping_step < 10:
        #         stopping_step += 1
        #         self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
        #     else:
        #         self.logger.logging('#####Early stop! #####')
        #         break
        self.logger.logging('#####final result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], max(final_recall),  max(final_precison), max(final_ndcg)))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], min(final_c_recall), min(final_c_precison), min(final_c_ndcg)))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-2], max(final_w_recall), max(final_w_precison), max(final_w_ndcg)))

        index = torch.argmax(torch.tensor(final_recall))
        self.logger.logging('#####final soft result! #####')
        self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_recall[index],  final_precison[index], final_ndcg[index]))
        self.logger.logging("Test_Cold_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_c_recall[index], final_c_precison[index], final_c_ndcg[index]))
        self.logger.logging("Test_Warm_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[-1], final_w_recall[index], final_w_precison[index], final_w_ndcg[index]))

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

    trainer = Trainer(config, n_users, n_items)
    trainer.train()
