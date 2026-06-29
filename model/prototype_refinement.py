import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.prototype_utils import build_incidence, run_kmeans, normalize_rows


class PrototypeGuidedRefinement(nn.Module):
    """Prototype-guided edge refinement and propagation for the shared graph."""

    def __init__(self, opt, emb_size, n_layers):
        super(PrototypeGuidedRefinement, self).__init__()
        self.opt = opt
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.dropout = opt['dropout']
        self.k_user = opt.get('k_user', 32)
        self.k_item = opt.get('k_item', 32)
        self.share_n_layers = opt.get('share_gnn') if opt.get('share_gnn') is not None else n_layers
        self.edge_reliability_mode = opt.get('edge_reliability_mode', 'learned')

        self.threshold = 0.5
        self.gumbel_tau = 1.0
        self.edge_eps = 1e-7
        self.min_keep_ratio = 0.6
        self.edge_neg_ratio = float(opt.get('edge_neg_ratio', 1.0))
        self.max_edge_denoise_samples = int(opt.get('max_edge_denoise_samples', 8192))

        edge_dim = emb_size * 6
        self.source_edge_gate = nn.Sequential(
            nn.Linear(edge_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
        )
        self.target_edge_gate = nn.Sequential(
            nn.Linear(edge_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

        self.user_msg_gate = nn.Linear(emb_size, 6)
        self.source_msg_gate = nn.Linear(emb_size, 3)
        self.target_msg_gate = nn.Linear(emb_size, 3)
        self.prototype_momentum = nn.Parameter(torch.tensor(0.5))
        self.cross_prototype_gate = nn.Linear(emb_size, 2)

        self.prototype_bundle = None
        self.prototype_bundle_epoch = None
        self.last_refinement_loss = None
        self.last_edge_denoise_loss = None
        self.last_stats = {}
        self.last_edge_probability = {}
        self._shuffle_perm_cache = {}

    def sparse_dropout(self, mat):
        if self.dropout == 0.0:
            return mat
        values = F.dropout(mat._values(), p=self.dropout, training=self.training)
        return torch.sparse_coo_tensor(
            mat._indices(), values, mat.size(), device=values.device
        ).coalesce()

    def _kmeans_bundle_one(self, embeddings, k):
        x = normalize_rows(embeddings)
        assign, centers = run_kmeans(
            x,
            k,
            max_iter=self.opt.get('kmeans_max_iter', 100),
            seed=self.opt.get('seed', 2020),
        )
        node_to_prototype, prototype_to_node = build_incidence(embeddings.size(0), k, assign)

        def to_sparse(mat):
            mat = mat.tocoo()
            indices = torch.tensor(
                np.vstack([mat.row, mat.col]), dtype=torch.long, device=embeddings.device
            )
            values = torch.tensor(mat.data, dtype=embeddings.dtype, device=embeddings.device)
            return torch.sparse_coo_tensor(indices, values, mat.shape, device=embeddings.device).coalesce()

        centers = torch.from_numpy(centers).to(embeddings.device, dtype=embeddings.dtype)
        return {
            'assignment': torch.from_numpy(assign).long().to(embeddings.device),
            'centers': F.normalize(centers, dim=1),
            'node_to_prototype': to_sparse(node_to_prototype),
            'prototype_to_node': to_sparse(prototype_to_node),
        }

    def build_prototype_bundle(self, user_emb, source_item_emb, target_item_emb):
        return {
            'user': self._kmeans_bundle_one(user_emb, self.k_user),
            'source': self._kmeans_bundle_one(source_item_emb, self.k_item),
            'target': self._kmeans_bundle_one(target_item_emb, self.k_item),
        }

    def prepare_prototype_bundle(self, user_emb, source_item_emb, target_item_emb):
        current_epoch = self.opt.get('_current_epoch', 0)
        if self.training and self.prototype_bundle_epoch != current_epoch:
            self.prototype_bundle = self.build_prototype_bundle(user_emb, source_item_emb, target_item_emb)
            self.prototype_bundle_epoch = current_epoch
        elif self.prototype_bundle is None:
            self.prototype_bundle = self.build_prototype_bundle(user_emb, source_item_emb, target_item_emb)
            self.prototype_bundle_epoch = current_epoch
        return self.prototype_bundle

    def _edge_features(self, user_emb, item_emb, edge_index, user_prototype, item_prototype):
        users, items = edge_index[0], edge_index[1]
        edge_user = user_emb[users]
        edge_item = item_emb[items]
        edge_user_center = user_prototype['centers'][user_prototype['assignment'][users]]
        edge_item_center = item_prototype['centers'][item_prototype['assignment'][items]]
        return torch.cat(
            [
                edge_user,
                edge_item,
                edge_user * edge_item,
                edge_user_center,
                edge_item_center,
                edge_user_center * edge_item_center,
            ],
            dim=1,
        )

    def _sample_edge_weights(self, logits):
        probability = torch.sigmoid(logits)
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            soft = torch.sigmoid((logits + gumbels) / self.gumbel_tau)
            hard = (soft > self.threshold).to(soft.dtype)
            mask = hard.detach() - soft.detach() + soft
        else:
            mask = (probability > self.threshold).to(probability.dtype)
        return self.edge_eps + (1.0 - self.edge_eps) * mask, probability

    def _subsample_edges(self, edge_index, max_samples):
        num_edges = edge_index.size(1)
        if max_samples <= 0 or num_edges <= max_samples:
            return edge_index
        perm = torch.randperm(num_edges, device=edge_index.device)[:max_samples]
        return edge_index[:, perm]

    def _clean_edges(self, edge_index, edge_label):
        if edge_label is None:
            return edge_index
        clean_mask = edge_label.to(edge_index.device).bool()
        if clean_mask.sum() == 0:
            return edge_index[:, :0]
        return edge_index[:, clean_mask]

    def _edge_denoise_loss(
        self,
        gate,
        user_emb,
        item_emb,
        edge_index,
        edge_label,
        user_prototype,
        item_prototype,
        n_items,
    ):
        if self.edge_reliability_mode == 'uniform' or self.edge_neg_ratio <= 0:
            zero = user_emb.new_tensor(0.0)
            return zero, zero, zero

        pos_edges = self._clean_edges(edge_index, edge_label)
        pos_edges = self._subsample_edges(pos_edges, self.max_edge_denoise_samples)
        num_pos = pos_edges.size(1)
        if num_pos == 0:
            zero = user_emb.new_tensor(0.0)
            return zero, zero, zero

        num_neg = max(1, int(round(num_pos * self.edge_neg_ratio)))
        base_index = torch.randint(num_pos, (num_neg,), device=edge_index.device)
        neg_users = pos_edges[0, base_index]
        neg_items = torch.randint(n_items, (num_neg,), device=edge_index.device)
        neg_edges = torch.stack([neg_users, neg_items], dim=0)

        pos_logits = gate(
            self._edge_features(user_emb, item_emb, pos_edges, user_prototype, item_prototype)
        ).squeeze(-1)
        neg_logits = gate(
            self._edge_features(user_emb, item_emb, neg_edges, user_prototype, item_prototype)
        ).squeeze(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, torch.sigmoid(pos_logits).detach().mean(), torch.sigmoid(neg_logits).detach().mean()

    def _fixed_permutation(self, num_edges, device, offset):
        key = (num_edges, device.type, device.index, offset)
        if key not in self._shuffle_perm_cache:
            generator = torch.Generator()
            generator.manual_seed(int(self.opt.get('seed', 2020)) + offset)
            perm = torch.randperm(num_edges, generator=generator)
            self._shuffle_perm_cache[key] = perm
        return self._shuffle_perm_cache[key].to(device)

    def _apply_edge_reliability_mode(self, weight, probability, domain_offset):
        if self.edge_reliability_mode == 'learned':
            return weight, probability
        if self.edge_reliability_mode == 'uniform':
            uniform = torch.ones_like(weight)
            return uniform, uniform
        if self.edge_reliability_mode == 'shuffle':
            perm = self._fixed_permutation(weight.size(0), weight.device, domain_offset)
            return weight[perm], probability[perm]
        raise ValueError('Unknown edge_reliability_mode: {}'.format(self.edge_reliability_mode))

    def _row_normalized_sparse(self, rows, cols, values, n_rows, n_cols):
        degree = torch.zeros(n_rows, dtype=values.dtype, device=values.device)
        degree.index_add_(0, rows, values)
        normalized_values = values / degree[rows].clamp_min(self.edge_eps)
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(
            indices, normalized_values, (n_rows, n_cols), device=values.device
        ).coalesce()

    def refine_edges(self, user_emb, source_item_emb, target_item_emb, bundle, cross_payload):
        source_edges = cross_payload['source_edge_index']
        target_edges = cross_payload['target_edge_index']

        source_features = self._edge_features(
            user_emb, source_item_emb, source_edges, bundle['user'], bundle['source']
        )
        target_features = self._edge_features(
            user_emb, target_item_emb, target_edges, bundle['user'], bundle['target']
        )
        source_logits = self.source_edge_gate(source_features).squeeze(-1)
        target_logits = self.target_edge_gate(target_features).squeeze(-1)
        source_weight, source_probability = self._sample_edge_weights(source_logits)
        target_weight, target_probability = self._sample_edge_weights(target_logits)
        source_weight, source_probability = self._apply_edge_reliability_mode(
            source_weight, source_probability, domain_offset=11
        )
        target_weight, target_probability = self._apply_edge_reliability_mode(
            target_weight, target_probability, domain_offset=29
        )
        if self.edge_reliability_mode == 'uniform':
            loss = source_probability.new_tensor(0.0)
        else:
            loss = (
                F.relu(self.min_keep_ratio - source_probability.mean()).pow(2)
                + F.relu(self.min_keep_ratio - target_probability.mean()).pow(2)
            )

        n_users = user_emb.size(0)
        n_source = source_item_emb.size(0)
        n_target = target_item_emb.size(0)
        source_denoise_loss, source_pos_prob, source_neg_prob = self._edge_denoise_loss(
            self.source_edge_gate,
            user_emb,
            source_item_emb,
            source_edges,
            cross_payload.get('source_edge_label'),
            bundle['user'],
            bundle['source'],
            n_source,
        )
        target_denoise_loss, target_pos_prob, target_neg_prob = self._edge_denoise_loss(
            self.target_edge_gate,
            user_emb,
            target_item_emb,
            target_edges,
            cross_payload.get('target_edge_label'),
            bundle['user'],
            bundle['target'],
            n_target,
        )
        denoise_loss = source_denoise_loss + target_denoise_loss
        su, si = source_edges[0], source_edges[1]
        tu, ti = target_edges[0], target_edges[1]

        payload = {
            'source_to_user': self._row_normalized_sparse(su, si, source_weight, n_users, n_source),
            'user_to_source': self._row_normalized_sparse(si, su, source_weight, n_source, n_users),
            'target_to_user': self._row_normalized_sparse(tu, ti, target_weight, n_users, n_target),
            'user_to_target': self._row_normalized_sparse(ti, tu, target_weight, n_target, n_users),
        }

        cross_rows = torch.cat([su, si + n_users, tu, ti + n_users + n_source])
        cross_cols = torch.cat([si + n_users, su, ti + n_users + n_source, tu])
        cross_weight = torch.cat([source_weight, source_weight, target_weight, target_weight])
        payload['cross_adj'] = self._row_normalized_sparse(
            cross_rows,
            cross_cols,
            cross_weight,
            n_users + n_source + n_target,
            n_users + n_source + n_target,
        )
        stats = {
            'source_keep': source_probability.detach().mean(),
            'target_keep': target_probability.detach().mean(),
            'source_edge_pos_prob': source_pos_prob,
            'source_edge_neg_prob': source_neg_prob,
            'target_edge_pos_prob': target_pos_prob,
            'target_edge_neg_prob': target_neg_prob,
            'edge_reliability_mode': self.edge_reliability_mode,
        }
        self.last_edge_probability = {
            'source': source_probability.detach(),
            'target': target_probability.detach(),
        }
        return payload, loss, denoise_loss, stats

    def lightgcn_forward(self, all_embeddings, norm_adj_matrix, n_users, n_items):
        embeddings = [all_embeddings]
        for _ in range(self.share_n_layers):
            all_embeddings = torch.sparse.mm(self.sparse_dropout(norm_adj_matrix), all_embeddings)
            embeddings.append(all_embeddings)
        all_embeddings = torch.stack(embeddings, dim=1).mean(dim=1)
        return torch.split(all_embeddings, [n_users, n_items])

    def global_cross_forward(self, user_emb, source_item_emb, target_item_emb, payload):
        all_embeddings = torch.cat([user_emb, source_item_emb, target_item_emb], dim=0)
        user_output, item_output = self.lightgcn_forward(
            all_embeddings,
            payload['cross_adj'],
            user_emb.size(0),
            source_item_emb.size(0) + target_item_emb.size(0),
        )
        source_output, target_output = torch.split(
            item_output, [source_item_emb.size(0), target_item_emb.size(0)]
        )
        return user_output, source_output, target_output

    def prototype_propagation(self, user_emb, source_item_emb, target_item_emb, bundle, payload):
        source_to_user = self.sparse_dropout(payload['source_to_user'])
        target_to_user = self.sparse_dropout(payload['target_to_user'])
        user_to_source = self.sparse_dropout(payload['user_to_source'])
        user_to_target = self.sparse_dropout(payload['user_to_target'])

        user_state = user_emb
        source_state = source_item_emb
        target_state = target_item_emb
        user_history = [user_state]
        source_history = [source_state]
        target_history = [target_state]

        user_prototype_state = bundle['user']['centers']
        source_prototype_state = bundle['source']['centers']
        target_prototype_state = bundle['target']['centers']
        u_uc = bundle['user']['node_to_prototype']
        uc_u = bundle['user']['prototype_to_node']
        s_sc = bundle['source']['node_to_prototype']
        sc_s = bundle['source']['prototype_to_node']
        t_tc = bundle['target']['node_to_prototype']
        tc_t = bundle['target']['prototype_to_node']

        for _ in range(self.n_layers):
            user_from_source = torch.sparse.mm(source_to_user, source_state)
            user_from_target = torch.sparse.mm(target_to_user, target_state)
            source_from_user = torch.sparse.mm(user_to_source, user_state)
            target_from_user = torch.sparse.mm(user_to_target, user_state)

            user_from_prototype = torch.sparse.mm(u_uc, user_prototype_state)
            source_from_prototype = torch.sparse.mm(s_sc, source_prototype_state)
            target_from_prototype = torch.sparse.mm(t_tc, target_prototype_state)
            cross_gate = torch.sigmoid(self.cross_prototype_gate(user_state))
            user_from_source_prototype = cross_gate[:, :1] * source_from_prototype.mean(
                dim=0, keepdim=True
            ).expand_as(user_state)
            user_from_target_prototype = cross_gate[:, 1:] * target_from_prototype.mean(
                dim=0, keepdim=True
            ).expand_as(user_state)

            user_messages = torch.stack(
                [
                    user_state,
                    user_from_source,
                    user_from_target,
                    user_from_prototype,
                    user_from_source_prototype,
                    user_from_target_prototype,
                ],
                dim=1,
            )
            user_weights = F.softmax(self.user_msg_gate(user_state), dim=1).unsqueeze(-1)
            user_state = F.normalize((user_messages * user_weights).sum(dim=1), dim=1)

            source_messages = torch.stack(
                [source_state, source_from_user, source_from_prototype], dim=1
            )
            source_weights = F.softmax(self.source_msg_gate(source_state), dim=1).unsqueeze(-1)
            source_state = F.normalize((source_messages * source_weights).sum(dim=1), dim=1)

            target_messages = torch.stack(
                [target_state, target_from_user, target_from_prototype], dim=1
            )
            target_weights = F.softmax(self.target_msg_gate(target_state), dim=1).unsqueeze(-1)
            target_state = F.normalize((target_messages * target_weights).sum(dim=1), dim=1)

            momentum = torch.sigmoid(self.prototype_momentum)
            user_prototype_state = F.normalize(
                momentum * user_prototype_state + (1.0 - momentum) * torch.sparse.mm(uc_u, user_state),
                dim=1,
            )
            source_prototype_state = F.normalize(
                momentum * source_prototype_state
                + (1.0 - momentum) * torch.sparse.mm(sc_s, source_state),
                dim=1,
            )
            target_prototype_state = F.normalize(
                momentum * target_prototype_state
                + (1.0 - momentum) * torch.sparse.mm(tc_t, target_state),
                dim=1,
            )

            user_history.append(user_state)
            source_history.append(source_state)
            target_history.append(target_state)

        return (
            torch.stack(user_history, dim=1).mean(dim=1),
            torch.stack(source_history, dim=1).mean(dim=1),
            torch.stack(target_history, dim=1).mean(dim=1),
        )

    def forward(self, user_emb, source_item_emb, target_item_emb, cross_payload):
        bundle = self.prepare_prototype_bundle(user_emb, source_item_emb, target_item_emb)
        payload, loss, denoise_loss, stats = self.refine_edges(
            user_emb, source_item_emb, target_item_emb, bundle, cross_payload
        )
        base_user, base_source, base_target = self.global_cross_forward(
            user_emb, source_item_emb, target_item_emb, payload
        )
        outputs = self.prototype_propagation(
            base_user, base_source, base_target, bundle, payload
        )
        self.last_refinement_loss = loss
        self.last_edge_denoise_loss = denoise_loss
        self.last_stats = stats
        return outputs
