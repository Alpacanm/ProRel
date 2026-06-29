import os
import shutil
import time
import numpy as np
import random
import argparse
import torch
from model.trainer import Trainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker, create_cross_Graph, norm_UV_VU_adj


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='phone_electronic, sport_phone, sport_cloth, electronic_cloth', help='')

parser.add_argument('--model_name', type=str, default='ProRel', help='ProRel,DIDA_CDR,DisenCDR,LightGCN')
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--GNN', type=int, default=3, help='GNN layer.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=1e-4, help='the weight decay of optimizer.')

parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--lambda1', type=float, default=1.0, help='')
parser.add_argument('--lambda2', type=float, default=1.0, help='')
parser.add_argument(
    '--use_prototype_denoise',
    type=int,
    default=1,
    choices=[0, 1],
    help='whether to use the complete prototype-guided denoising module',
)
parser.add_argument('--lambda_refine', type=float, default=0.01, help='weight for prototype-guided edge refinement loss')
parser.add_argument('--lambda_edge_denoise', type=float, default=0.1, help='weight for contrastive edge denoising loss')
parser.add_argument('--edge_neg_ratio', type=float, default=1.0, help='number of sampled unobserved negative edges per positive edge for edge denoising')
parser.add_argument('--max_edge_denoise_samples', type=int, default=8192, help='maximum positive edges per domain used in one edge denoising loss call')
parser.add_argument('--share_gnn', type=int, default=3, help='number of LightGCN layers for shared graph propagation before prototype propagation; defaults to GNN')
parser.add_argument('--k_user', type=int, default=32, help='number of user prototypes')
parser.add_argument('--k_item', type=int, default=32, help='number of item prototypes for each domain')
parser.add_argument('--kmeans_max_iter', type=int, default=100, help='maximum iterations for prototype assignment')
parser.add_argument(
    '--edge_reliability_mode',
    type=str,
    default='learned',
    choices=['learned', 'uniform', 'shuffle'],
    help='edge reliability mode for prototype-guided shared graph refinement',
)
parser.add_argument(
    '--pgrl_noise_ratio',
    type=float,
    default=0.0,
    help='ratio of synthetic unobserved edges injected only into the PGRL/shared graph',
)
parser.add_argument(
    '--pgrl_noise_seed',
    type=int,
    default=None,
    help='random seed for synthetic PGRL noise edges; defaults to --seed',
)
parser.add_argument('--use_reliability_regularization', type=int, default=1, choices=[0, 1], help='whether to use reliability-aware regularization')
parser.add_argument('--lambda_regularization', type=float, default=0.5, help='weight for reliability-aware regularization')
parser.add_argument('--regularization_alpha', type=float, default=1.0, help='weight for representation preservation terms')
parser.add_argument('--regularization_beta', type=float, default=0.1, help='weight for HSIC regularization terms')
parser.add_argument('--regularization_sigma', type=float, default=1.0, help='RBF kernel sigma for HSIC')
parser.add_argument('--regularization_temp', type=float, default=0.05, help='temperature for InfoNCE')
parser.add_argument('--min_hsic_batch', type=int, default=16, help='minimum batch size for HSIC terms')
parser.add_argument('--fusion_mode', type=str, default='ours', choices=['ours', 'mean', 'sum'], help='user fusion strategy')
parser.add_argument('--reg_weight', type=float, default=1e-4, help="the weight decay of BPR Loss")
parser.add_argument('--temp', type=float, default=0.05, help='')

parser.add_argument('--leaky', type=float, default=0.1, help='leakyReLU for GCN')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--beta', type=float, default=0.9, help='paramter in DisenCDR')

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def edge_noise_summary(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    clean = scores[labels == 1]
    noisy = scores[labels == 0]
    return {
        'clean_count': int(clean.size),
        'noisy_count': int(noisy.size),
        'clean_mean': float(clean.mean()) if clean.size else np.nan,
        'noisy_mean': float(noisy.mean()) if noisy.size else np.nan,
        'clean_median': float(np.median(clean)) if clean.size else np.nan,
        'noisy_median': float(np.median(noisy)) if noisy.size else np.nan,
        'mean_gap': float(clean.mean() - noisy.mean()) if clean.size and noisy.size else np.nan,
        'score_std': float(scores.std()) if scores.size else np.nan,
        'auc_clean_vs_noisy': float(binary_auc(labels, scores)),
    }

args = parser.parse_args()
if args.pgrl_noise_seed is None:
    args.pgrl_noise_seed = args.seed
init_time = time.time()
opt = vars(args)
seed_everything(opt["seed"])

log = os.path.join(
    'logs/{}'.format(args.model_name),
    '{}_{}_{}_{}_{}_reg_{}_ku_{}_ki_{}_edge_{}_seed_{}'.format(
        args.dataset,
        args.GNN,
        args.lambda1,
        args.lambda2,
        args.temp,
        args.lambda_regularization,
        args.k_user,
        args.k_item,
        args.edge_reliability_mode,
        args.seed,
    ),
)
if args.pgrl_noise_ratio > 0:
    log = '{}_pgrl_noise_{}'.format(log, args.pgrl_noise_ratio)
if os.path.isdir(log):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
    time.sleep(5)
    shutil.rmtree(log)

os.makedirs(log)
print("made the log directory", log)
print(opt)
with open(log + '/tmp.txt', 'a') as f:
    f.write(str(opt))

datasetname  = opt["dataset"]
source_G = GraphMaker(opt, datasetname)
source_UV = source_G.UV
source_VU = source_G.VU
source_adj = source_G.adj

datasetname = datasetname.split("_")
datasetname = datasetname[1] + "_" + datasetname[0]
target_G = GraphMaker(opt, datasetname)
target_UV = target_G.UV
target_VU = target_G.VU
target_adj = target_G.adj

cross_payload = create_cross_Graph(source_UV, source_VU, target_UV, target_VU, opt['dataset'], opt)
if cross_payload.get('noise_stats') is not None:
    print("PGRL noise stats:", cross_payload['noise_stats'])
if opt['model_name'] != 'DGCF':
    source_UV, source_VU, target_UV, target_VU = norm_UV_VU_adj(source_UV, source_VU, target_UV, target_VU, opt['dataset'])
    source_UV = source_UV.cuda()
    source_VU = source_VU.cuda()
    target_UV = target_UV.cuda()
    target_VU = target_VU.cuda()

print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
source_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
target_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)

print("user_num", opt["source_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("source train data : {}, target train data {}, source test data : {}, source test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data)))

opt['inter_num_s'] = len(train_batch.source_train_data)
opt['inter_num_t'] = len(train_batch.target_train_data)

source_adj = source_adj.cuda()
target_adj = target_adj.cuda()
for key, value in cross_payload.items():
    if torch.is_tensor(value):
        cross_payload[key] = value.cuda()

trainer = Trainer(opt, source_adj, target_adj, cross_payload, source_UV, source_VU, target_UV, target_VU)

best_s_score = [0, 0]
best_t_score = [0, 0]
best_epoch_s = 0
best_epoch_t = 0
for epoch in range(1, opt['num_epoch'] + 1):
    opt['_current_epoch'] = epoch
    trainer.model.opt['_current_epoch'] = epoch
    start_time = time.time()
    batch_all_loss = []
    for i, batch in enumerate(train_batch):
        loss = trainer.reconstruct_graph(batch)
        batch_all_loss.append(loss)

    epoch_all_loss = np.mean(batch_all_loss)
    duration = time.time() - start_time
    print('epoch:{}, time:{:.2f}, loss:{:.4f}'.format(epoch, duration, epoch_all_loss))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('epoch:{}, time:{:.2f}, loss:{:.4f}\n'.format(epoch, duration, epoch_all_loss))

    if epoch <= opt['num_epoch'] - 10:
        continue

    trainer.model.eval()
    trainer.evaluate_embedding()

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(source_dev_batch):
        predictions = trainer.source_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    s_ndcg = NDCG / valid_entity
    s_hit = HT / valid_entity
    if s_ndcg > best_s_score[1]:
        best_s_score[1] = s_ndcg
        best_s_score[0] = s_hit
        best_epoch_s = epoch
        torch.save(trainer.model.state_dict(), os.path.join(log, 'best_ndcg1.pkl'))

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(target_dev_batch):
        predictions = trainer.target_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    t_ndcg = NDCG / valid_entity
    t_hit = HT / valid_entity
    if t_ndcg > best_t_score[1]:
        best_t_score[1] = t_ndcg
        best_t_score[0] = t_hit
        best_epoch_t = epoch
        torch.save(trainer.model.state_dict(), os.path.join(log, 'best_ndcg2.pkl'))

    print('test: hr1:{:.4f},ndcg1:{:.4f}, hr2:{:.4f},ndcg2:{:.4f}'.format(s_hit, s_ndcg, t_hit, t_ndcg))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('test: hr1:{:.4f},ndcg1:{:.4f}, hr2:{:.4f},ndcg2:{:.4f}\n'.format(s_hit, s_ndcg, t_hit, t_ndcg))

print('best epoch1 {}: hr1:{:.4f},ndcg1:{:.4f}, best epoch2 {}: hr2:{:.4f},ndcg2:{:.4f}'.\
      format(best_epoch_s, best_s_score[0], best_s_score[1], best_epoch_t, best_t_score[0], best_t_score[1]))
with open(log + '/tmp.txt', 'a') as f:
        f.write('best epoch1 {}: hr1:{:.4f},ndcg1:{:.4f}, best epoch2 {}: hr2:{:.4f},ndcg2:{:.4f}'.\
            format(best_epoch_s, best_s_score[0], best_s_score[1], best_epoch_t, best_t_score[0], best_t_score[1]))

if getattr(trainer.model, 'prototype_refinement', None) is not None:
    edge_probability = trainer.model.prototype_refinement.last_edge_probability
    if edge_probability:
        edge_prob_path = os.path.join(log, 'edge_reliability_probs.npz')
        source_prob = edge_probability['source'].detach().cpu().numpy()
        target_prob = edge_probability['target'].detach().cpu().numpy()
        source_label = cross_payload['source_edge_label'].detach().cpu().numpy()
        target_label = cross_payload['target_edge_label'].detach().cpu().numpy()
        source_edge_index = cross_payload['source_edge_index'].detach().cpu().numpy()
        target_edge_index = cross_payload['target_edge_index'].detach().cpu().numpy()
        source_summary = edge_noise_summary(source_label, source_prob)
        target_summary = edge_noise_summary(target_label, target_prob)
        np.savez(
            edge_prob_path,
            source=source_prob,
            target=target_prob,
            source_label=source_label,
            target_label=target_label,
            source_edge_index=source_edge_index,
            target_edge_index=target_edge_index,
            pgrl_noise_ratio=np.array([args.pgrl_noise_ratio], dtype=np.float32),
        )
        edge_summary_path = os.path.join(log, 'edge_reliability_noise_summary.txt')
        with open(edge_summary_path, 'w') as f:
            f.write('source {}\n'.format(source_summary))
            f.write('target {}\n'.format(target_summary))
        print('saved edge reliability probabilities to {}'.format(edge_prob_path))
        print('edge reliability noise summary source:', source_summary)
        print('edge reliability noise summary target:', target_summary)
        with open(log + '/tmp.txt', 'a') as f:
            f.write('\nsaved edge reliability probabilities to {}'.format(edge_prob_path))
            f.write('\nsource edge reliability noise summary: {}'.format(source_summary))
            f.write('\ntarget edge reliability noise summary: {}'.format(target_summary))
print(opt)
