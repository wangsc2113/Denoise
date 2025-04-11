import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    #useless
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')    
    parser.add_argument('--core', type=int, default=5, help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--lambda_coeff', type=float, default=0.9, help='Lambda value of skip connection')

    parser.add_argument('--early_stopping_patience', type=int, default=7, help='') 
    parser.add_argument('--layers', type=int, default=1, help='Number of feature graph conv layers')  
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--metapath_threshold', default=2, type=int, help='metapath_threshold') 
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--ssl_c_rate', type=float, default=1.3, help='ssl_c_rate')
    parser.add_argument('--ssl_s_rate', type=float, default=0.8, help='ssl_s_rate')
    parser.add_argument('--g_rate', type=float, default=0.000029, help='ssl_s_rate')
    parser.add_argument('--sample_num', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_neg', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_ii', default=8, type=int, help='sample_num') 
    parser.add_argument('--sample_num_co', default=2, type=int, help='sample_num') 
    parser.add_argument('--mask_rate', default=0.75, type=float, help='sample_num') 
    parser.add_argument('--gss_rate', default=0.85, type=float, help='gene_self_subgraph_rate') 
    parser.add_argument('--anchor_rate', default=0.75, type=float, help='anchor_rate') 
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='feat_reg_decay') 
    parser.add_argument('--ad1_rate', default=0.2, type=float, help='ad1_rate') 
    parser.add_argument('--ad2_rate', default=0.2, type=float, help='ad1_rate')     
    parser.add_argument('--ad_sampNum', type=int, default=1, help='ad topk')  
    parser.add_argument('--ad_topk_multi_num', type=int, default=100, help='ad topk')  
    parser.add_argument('--fake_gene_rate', default=0.0001, type=float, help='fake_gene_rate') 
    parser.add_argument('--ID_layers', type=int, default=1, help='Number of item graph conv layers')  
    parser.add_argument('--reward_rate', default=1, type=float, help='fake_gene_rate') 
    parser.add_argument('--G_embed_size', type=int, default=64, help='Embedding size.')   
    parser.add_argument('--model_num', default=2, type=float, help='fake_gene_rate') 
    parser.add_argument('--negrate', default=0.01, type=float, help='item_neg_sample_rate')
    parser.add_argument('--cis', default=25, type=int, help='') 
    parser.add_argument('--confidence', default=0.5, type=float, help='') 
    parser.add_argument('--ii_it', default=15, type=int, help='') 
    parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
    parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
    parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  #


    #train
    parser.add_argument('--data_path', nargs='?', default='/home/chengaode/wangshicheng/label_denoising/Dataset/', help='Input data path.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='', help='Choose a dataset from {sports, baby, clothing, tiktok, allrecipes}')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  #default: 1000
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=64,help='Embedding size.')                     
    parser.add_argument('--D_lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--topk', type=int, default=10, help='K value of k-NN sparsification')  
    parser.add_argument('--cf_model', nargs='?', default='slmrec', help='Downstream Collaborative Filtering model {mf}')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--cl_rate', type=float, default=0.03, help='Control the effect of the contrastive auxiliary task')        
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='for emb_loss.')  #default: '[1e-5,1e-5,1e-2]'
    # parser.add_argument('--lr', type=float, default=0.00055, help='Learning rate.')
    parser.add_argument('--emm', default=1e-3, type=float, help='for feature embedding bpr')  #
    parser.add_argument('--L2_alpha', default=1e-3, type=float, help='')  #
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='for opt_D')  #


    #GNN
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.55, help='model_cat_rate')
    parser.add_argument('--gnn_cat_rate', type=float, default=0.55, help='gnn_cat_rate')
    parser.add_argument('--id_cat_rate', type=float, default=0.36, help='before GNNs')
    parser.add_argument('--id_cat_rate1', type=float, default=0.36, help='id_cat_rate')
    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention. For multi-model relation.')  #
    parser.add_argument('--dgl_nei_num', default=8, type=int, help='dgl_nei_num')  #


    #GAN
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')  #default: '[64, 64]'
    parser.add_argument('--G_rate', default=0.0001, type=float, help='for D model1')  #
    parser.add_argument('--G_drop1', default=0.31, type=float, help='for D model2')  #
    parser.add_argument('--G_drop2', default=0.5, type=float, help='')  #
    parser.add_argument('--gp_rate', default=1, type=float, help='gradient penal')  #

    parser.add_argument('--real_data_tau', default=0.005, type=float, help='for real_data soft')  #
    parser.add_argument('--ui_pre_scale', default=100, type=int, help='ui_pre_scale')  


    #cl
    parser.add_argument('--T', default=1, type=int, help='it for ui update')  
    parser.add_argument('--tau', default=0.5, type=float, help='')  #
    parser.add_argument('--geneGraph_rate', default=0.1, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_pos', default=2, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_neg', default=-1, type=float, help='')  #     
    parser.add_argument('--m_topk_rate', default=0.0001, type=float, help='for reconstruct')  
    parser.add_argument('--log_log_scale', default=0.00001, type=int, help='log_log_scale')  
    parser.add_argument('--point', default='', type=str, help='point')  

    parser.add_argument('--vv_rate_alpha', default=-0.05, type=float)  #

    # parser.add_argument('--batch_size', type=int,default=2048,
    #                     help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=float,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=512,
                        help="the batch size of users for testing")
    # parser.add_argument('--data_path', type=str, default='/storage/jjzhao/jujia_ws/diff/data/yelp_noisy',
                        # help='the path to dataset')
    # parser.add_argument('--dataset', type=str,default='yelp_noisy',
    #                     help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--data_type', type=str, default='time',
                        help='time or random')
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20, 50, 100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    # parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log', type=str)
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")

    # diff reverse params (DNN)
    parser.add_argument('--dims', type=str, default='[200,600]', help='the dims for the DNN')
    parser.add_argument('--act', type=str, default='tanh', help='the activate function for the DNN')
    parser.add_argument('--w_dims', type=str, default='[200,600]', help='the dims for the W DNN')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--diff_lr', type=float,default=0.001, help="the learning rate")
    parser.add_argument('--joint1', type=float,default=0.001, help="the joint1 rate")
    parser.add_argument('--joint2', type=float,default=0.001, help="the joint2 rate")
    # diff params
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=2, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=5e-3, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.005, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=2, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')    

    parser.add_argument('--alpha', type=float, default=0.1, help='balance rec loss and reconstruct loss')

    # denoising 
    # parser.add_argument('--drop_rate', 
    #     type = float,
    #     help = 'drop rate',
    #     default = 0.2)
    parser.add_argument('--num_gradual', 
        type = int, 
        default = 30000,
        help='how many epochs to linearly increase drop_rate')
    parser.add_argument('--exponent', 
        type = float, 
        default = 1, 
        help='exponent of the drop rate {0.5, 1, 2}')
    parser.add_argument('--beta', 
        type = float,
        help = 'drop rate',
        default = 0.2)

    return parser.parse_args()


