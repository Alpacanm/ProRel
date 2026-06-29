import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def normalize_rows(embeddings):
    return F.normalize(embeddings, dim=1).detach().cpu().numpy().astype(np.float32)


def run_kmeans(x, k, max_iter=100, seed=2020):
    n = x.shape[0]
    if k >= n:
        assign = np.arange(n, dtype=np.int64) % k
        centers = np.zeros((k, x.shape[1]), dtype=np.float32)
        for i in range(n):
            centers[assign[i]] = x[i]
        return assign, centers

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()
    assign = np.full(n, -1, dtype=np.int64)

    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assign = dist.argmin(axis=1)
        if np.array_equal(assign, new_assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = x[mask].mean(axis=0)
            else:
                centers[c] = x[rng.integers(0, n)]
    return assign, centers.astype(np.float32)


def assignment_stats(x, assign, centers):
    sizes = np.bincount(assign, minlength=centers.shape[0]).tolist()
    normalized_centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8, None)
    normalized_x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    intra = []
    for idx, cid in enumerate(assign):
        intra.append(float((normalized_x[idx] * normalized_centers[cid]).sum()))
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter = normalized_centers @ normalized_centers.T
    if inter.shape[0] > 1:
        inter_mean = float((inter.sum() - np.trace(inter)) / (inter.shape[0] * (inter.shape[0] - 1)))
    else:
        inter_mean = 1.0
    return {
        'prototype_sizes': sizes,
        'intra_prototype_cosine_mean': intra_mean,
        'inter_center_cosine_mean': inter_mean,
    }


def build_incidence(num_nodes, num_prototypes, assign):
    rows = np.arange(num_nodes, dtype=np.int64)
    cols = assign.astype(np.int64)
    data = np.ones(num_nodes, dtype=np.float32)
    node_to_prototype = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_prototypes), dtype=np.float32)
    prototype_to_node = sp.coo_matrix((data, (cols, rows)), shape=(num_prototypes, num_nodes), dtype=np.float32)
    return node_to_prototype, prototype_to_node


def normalize_sparse_rows(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    inv = np.power(rowsum, -1, where=rowsum != 0)
    inv[rowsum == 0] = 0.0
    return sp.diags(inv).dot(mx).tocoo()


def prototype_config(dataset, k_user, k_item):
    return {
        'dataset': dataset,
        'k_user': k_user,
        'k_item': k_item,
    }


def cache_matches(cache, dataset, k_user, k_item):
    if cache is None:
        return False
    return cache.get('config', {}) == prototype_config(dataset, k_user, k_item)


def save_prototype_bundle(dataset_dir, dataset, user_embeddings, source_embeddings, target_embeddings, k_user, k_item, seed=2020):
    user_x = normalize_rows(user_embeddings)
    source_x = normalize_rows(source_embeddings)
    target_x = normalize_rows(target_embeddings)

    user_assign, user_centers = run_kmeans(user_x, k_user, seed=seed)
    source_assign, source_centers = run_kmeans(source_x, k_item, seed=seed)
    target_assign, target_centers = run_kmeans(target_x, k_item, seed=seed)

    user_to_prototype, prototype_to_user = build_incidence(len(user_assign), k_user, user_assign)
    source_to_prototype, prototype_to_source = build_incidence(len(source_assign), k_item, source_assign)
    target_to_prototype, prototype_to_target = build_incidence(len(target_assign), k_item, target_assign)

    bundle = {
        'config': prototype_config(dataset, k_user, k_item),
        'user_assign': user_assign,
        'source_assign': source_assign,
        'target_assign': target_assign,
        'user_centers': user_centers,
        'source_centers': source_centers,
        'target_centers': target_centers,
        'user_stats': assignment_stats(user_x, user_assign, user_centers),
        'source_stats': assignment_stats(source_x, source_assign, source_centers),
        'target_stats': assignment_stats(target_x, target_assign, target_centers),
        'user_to_prototype': user_to_prototype,
        'prototype_to_user': prototype_to_user,
        'source_to_prototype': source_to_prototype,
        'prototype_to_source': prototype_to_source,
        'target_to_prototype': target_to_prototype,
        'prototype_to_target': prototype_to_target,
    }

    with open(os.path.join(dataset_dir, 'prototype_bundle.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    return bundle


def load_prototype_bundle(dataset_dir):
    path = os.path.join(dataset_dir, 'prototype_bundle.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)
