import numpy as np
import scipy.sparse as sp
import torch
import codecs
import os
import pickle

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _sample_unobserved_edges(adj, num_noise, seed):
    if num_noise <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    coo = adj.tocoo()
    n_users, n_items = adj.shape
    observed = set(zip(coo.row.tolist(), coo.col.tolist()))
    sampled = set()
    rng = np.random.RandomState(seed)
    max_edges = n_users * n_items - len(observed)
    num_noise = min(num_noise, max_edges)

    while len(sampled) < num_noise:
        remaining = num_noise - len(sampled)
        draw_size = max(remaining * 2, 1024)
        users = rng.randint(0, n_users, size=draw_size)
        items = rng.randint(0, n_items, size=draw_size)
        for user, item in zip(users, items):
            edge = (int(user), int(item))
            if edge in observed or edge in sampled:
                continue
            sampled.add(edge)
            if len(sampled) == num_noise:
                break

    sampled = np.array(list(sampled), dtype=np.int64)
    return sampled[:, 0], sampled[:, 1]


def _augment_with_noise_edges(adj, noise_ratio, seed):
    coo = adj.tocoo()
    clean_count = coo.nnz
    noise_count = int(round(clean_count * noise_ratio))
    noise_rows, noise_cols = _sample_unobserved_edges(adj, noise_count, seed)

    rows = np.concatenate([coo.row.astype(np.int64), noise_rows])
    cols = np.concatenate([coo.col.astype(np.int64), noise_cols])
    labels = np.concatenate(
        [
            np.ones(clean_count, dtype=np.int64),
            np.zeros(len(noise_rows), dtype=np.int64),
        ]
    )
    augmented = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=adj.shape,
        dtype=np.float32,
    )
    return augmented, labels, {
        'clean_count': clean_count,
        'noise_count': len(noise_rows),
        'noise_ratio': float(len(noise_rows)) / float(clean_count) if clean_count else 0.0,
    }


class GraphMaker(object):
    def __init__(self, opt, datasetname):
        self.opt = opt
        self.user = set()
        self.item = set()
        user_map = {}
        item_map = {}
        data=[]
        filename = "../dataset/" + datasetname + "/train.txt"
        self.datasetname = datasetname
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if line[0] not in user_map.keys():
                    user_map[line[0]] = len(user_map)
                if line[1] not in item_map.keys():
                    item_map[line[1]] = len(item_map)
                line[0] = user_map[line[0]]
                line[1] = item_map[line[1]]
                data.append((int(line[0]),int(line[1]),float(line[2])))
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)

        print("number_user", len(self.user))
        print("number_item", len(self.item))

        self.raw_data = data
        if not os.path.exists(os.path.join(os.getcwd(), '../dataset', datasetname, f'all_adj.pkl')):
            self.UV, self.VU, self.adj = self.preprocess(data, opt)
        else:
            print("real graph loaded!")
            self.adj = pickle.load(open(os.path.join('../dataset', datasetname, f'all_adj.pkl'), 'rb'))
            self.UV = pickle.load(open(os.path.join('../dataset', datasetname, f'UV_adj.pkl'), 'rb'))
            self.VU = pickle.load(open(os.path.join('../dataset', datasetname, f'VU_adj.pkl'), 'rb'))

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),dtype=np.float32)
        all_adj = normalize(all_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)
        pickle.dump(all_adj, open(os.path.join('../dataset', self.datasetname, f'all_adj.pkl'), 'wb'))
        pickle.dump(UV_adj, open(os.path.join('../dataset', self.datasetname, f'UV_adj.pkl'), 'wb'))
        pickle.dump(VU_adj, open(os.path.join('../dataset', self.datasetname, f'VU_adj.pkl'), 'wb'))
        print("real graph saved!")
        return UV_adj, VU_adj, all_adj

def create_cross_Graph(source_UV, source_VU, target_UV, target_VU, datasetname, opt):
    payload = {}
    noise_ratio = float(opt.get('pgrl_noise_ratio', 0.0))
    noise_seed = int(opt.get('pgrl_noise_seed', opt.get('seed', 2020)))
    if noise_ratio > 0:
        source_UV_for_cross, source_edge_label, source_noise_stats = _augment_with_noise_edges(
            source_UV, noise_ratio, noise_seed + 101
        )
        target_UV_for_cross, target_edge_label, target_noise_stats = _augment_with_noise_edges(
            target_UV, noise_ratio, noise_seed + 211
        )
        source_VU_for_cross = source_UV_for_cross.transpose().tocoo()
        target_VU_for_cross = target_UV_for_cross.transpose().tocoo()
    else:
        source_UV_for_cross = source_UV.tocoo()
        target_UV_for_cross = target_UV.tocoo()
        source_VU_for_cross = source_VU.tocoo()
        target_VU_for_cross = target_VU.tocoo()
        source_edge_label = np.ones(source_UV_for_cross.nnz, dtype=np.int64)
        target_edge_label = np.ones(target_UV_for_cross.nnz, dtype=np.int64)
        source_noise_stats = {
            'clean_count': source_UV_for_cross.nnz,
            'noise_count': 0,
            'noise_ratio': 0.0,
        }
        target_noise_stats = {
            'clean_count': target_UV_for_cross.nnz,
            'noise_count': 0,
            'noise_ratio': 0.0,
        }

    source_edges = source_UV_for_cross.tocoo()
    target_edges = target_UV_for_cross.tocoo()
    payload['source_edge_index'] = torch.from_numpy(
        np.vstack((source_edges.row, source_edges.col)).astype(np.int64)
    )
    payload['target_edge_index'] = torch.from_numpy(
        np.vstack((target_edges.row, target_edges.col)).astype(np.int64)
    )
    payload['source_edge_label'] = torch.from_numpy(source_edge_label)
    payload['target_edge_label'] = torch.from_numpy(target_edge_label)
    payload['noise_stats'] = {
        'source': source_noise_stats,
        'target': target_noise_stats,
    }
    payload['source_to_user'] = sparse_mx_to_torch_sparse_tensor(normalize(source_UV_for_cross))
    payload['user_to_source'] = sparse_mx_to_torch_sparse_tensor(normalize(source_VU_for_cross))
    payload['target_to_user'] = sparse_mx_to_torch_sparse_tensor(normalize(target_UV_for_cross))
    payload['user_to_target'] = sparse_mx_to_torch_sparse_tensor(normalize(target_VU_for_cross))

    n_users = source_UV.shape[0]
    n_items1 = source_UV.shape[1]
    n_items2 = target_UV.shape[1]
    cross_adj = sp.dok_matrix((n_users + n_items1 + n_items2, n_users + n_items1 + n_items2), dtype=np.float32)
    cross_adj = cross_adj.tolil()
    cross_adj[:n_users, n_users:n_items1 + n_users] = source_UV_for_cross.tolil()
    cross_adj[:n_users, n_users + n_items1:] = target_UV_for_cross.tolil()
    cross_adj[n_users:n_items1 + n_users, :n_users] = source_VU_for_cross.tolil()
    cross_adj[n_users + n_items1:, :n_users] = target_VU_for_cross.tolil()
    payload['cross_adj'] = sparse_mx_to_torch_sparse_tensor(normalize(cross_adj.todok()))
    return payload

def norm_UV_VU_adj(source_UV, source_VU, target_UV, target_VU, datasetname):
    if not os.path.exists(os.path.join(os.getcwd(), '../dataset', datasetname, 'UV_adj_norm.pkl')):
        source_UV = normalize(source_UV)
        source_VU = normalize(source_VU)
        target_UV = normalize(target_UV)
        target_VU = normalize(target_VU)

        source_UV = sparse_mx_to_torch_sparse_tensor(source_UV)
        source_VU = sparse_mx_to_torch_sparse_tensor(source_VU)
        target_UV = sparse_mx_to_torch_sparse_tensor(target_UV)
        target_VU = sparse_mx_to_torch_sparse_tensor(target_VU)

        pickle.dump(source_UV, open(os.path.join('../dataset', datasetname, f'UV_adj_norm.pkl'), 'wb'))
        pickle.dump(source_VU, open(os.path.join('../dataset', datasetname, f'VU_adj_norm.pkl'), 'wb'))

        datasetname = datasetname.split("_")
        datasetname = datasetname[1] + "_" + datasetname[0]

        pickle.dump(target_UV, open(os.path.join('../dataset', datasetname, f'UV_adj_norm.pkl'), 'wb'))
        pickle.dump(target_VU, open(os.path.join('../dataset', datasetname, f'VU_adj_norm.pkl'), 'wb'))
        print("normalized UV and VU saved!")
    else:
        source_UV = pickle.load(open(os.path.join('../dataset', datasetname, 'UV_adj_norm.pkl'), 'rb'))
        source_VU = pickle.load(open(os.path.join('../dataset', datasetname, 'VU_adj_norm.pkl'), 'rb'))

        datasetname = datasetname.split("_")
        datasetname = datasetname[1] + "_" + datasetname[0]

        target_UV = pickle.load(open(os.path.join('../dataset', datasetname, 'UV_adj_norm.pkl'), 'rb'))
        target_VU = pickle.load(open(os.path.join('../dataset', datasetname, 'VU_adj_norm.pkl'), 'rb'))
        print("normalized UV and VU loaded!")
    return source_UV, source_VU, target_UV, target_VU
