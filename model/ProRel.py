import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from model.prototype_refinement import PrototypeGuidedRefinement
from model.regularization_loss import ReliabilityRegularizationLoss


class ProRel(nn.Module):
    def __init__(self, opt, s_adj, t_adj, cross_payload):
        super(ProRel, self).__init__()
        self.opt = opt
        self.cross_payload = cross_payload
        self.s_adj = s_adj
        self.t_adj = t_adj
        self.n_layers = opt['GNN']
        self.dropout = opt['dropout']
        self.emb_size = opt['feature_dim']
        self.temp = opt['temp']
        self.fusion_mode = opt.get('fusion_mode', 'ours')
        self.use_prototype_denoise = bool(opt.get('use_prototype_denoise', 1))
        self.use_reliability_regularization = bool(opt.get('use_reliability_regularization', 1))
        self.lambda_regularization = opt.get('lambda_regularization', 0.0)
        self.reliability_regularization = ReliabilityRegularizationLoss(opt)
        self.lambda_refine = opt.get('lambda_refine', 0.0)
        self.lambda_edge_denoise = opt.get('lambda_edge_denoise', 0.0)

        self.user_num = opt["source_user_num"]
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], self.emb_size)
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], self.emb_size)
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], self.emb_size)
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], self.emb_size)
        self.share_user_embedding = nn.Embedding(opt["source_user_num"], self.emb_size)

        self.pop_encoder = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(opt['dropout']),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.int_encoder = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(opt['dropout']),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.domain_cls = nn.Sequential(nn.Linear(self.emb_size, 2), nn.Sigmoid())

        self.user_index = torch.arange(0, opt["source_user_num"], 1).cuda()
        self.source_item_index = torch.arange(0, opt["source_item_num"], 1).cuda()
        self.target_item_index = torch.arange(0, opt["target_item_num"], 1).cuda()

        self.att1_s = nn.Sequential(nn.Linear(self.emb_size * 6, self.emb_size))
        self.att2_s = nn.Linear(self.emb_size, 3)
        self.att1_t = nn.Sequential(nn.Linear(self.emb_size * 6, self.emb_size))
        self.att2_t = nn.Linear(self.emb_size, 3)

        self.agg_s = nn.Linear(self.emb_size * 2, self.emb_size)
        self.agg_t = nn.Linear(self.emb_size * 2, self.emb_size)
        self.prototype_refinement = (
            PrototypeGuidedRefinement(opt, self.emb_size, self.n_layers)
            if self.use_prototype_denoise
            else None
        )

        self.loss_cos = nn.CosineEmbeddingLoss()
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_KLD = nn.KLDivLoss(reduction='batchmean')

        self.restore_user_s = None
        self.restore_item_s = None
        self.restore_user_t = None
        self.restore_item_t = None
        self.restore_user_sha = None
        self.restore_item_sha_s = None
        self.restore_item_sha_t = None
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.source_user_embedding.weight.data)
        xavier_uniform_(self.target_user_embedding.weight.data)
        xavier_uniform_(self.source_item_embedding.weight.data)
        xavier_uniform_(self.target_item_embedding.weight.data)
        xavier_uniform_(self.share_user_embedding.weight.data)

    def predict_dot(self, user_embedding, item_embedding):
        return (user_embedding * item_embedding).sum(dim=-1)

    def emb_fusion(self, spe_emb, int_emb, pop_emb, domain='source'):
        if self.fusion_mode == 'mean':
            return (spe_emb + int_emb + pop_emb) / 3.0
        if self.fusion_mode == 'sum':
            return spe_emb + int_emb + pop_emb

        att1, att2 = (self.att1_s, self.att2_s) if domain == 'source' else (self.att1_t, self.att2_t)
        spe_int = spe_emb * int_emb
        spe_pop = spe_emb * pop_emb
        int_pop = int_emb * pop_emb
        x = torch.concat([spe_emb, int_emb, pop_emb, spe_int, spe_pop, int_pop], dim=1)
        x = F.relu(att1(x).cuda(), inplace=True)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = att2(x).cuda()
        att_w1, att_w2, att_w3 = F.softmax(x, dim=1).chunk(3, dim=1)
        return torch.mul(spe_emb, att_w1) + torch.mul(int_emb, att_w2) + torch.mul(pop_emb, att_w3) + spe_emb

    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat._indices()
        values = nn.functional.dropout(mat._values(), p=dropout, training=self.training)
        return torch.sparse_coo_tensor(indices, values, mat.size(), device=values.device).coalesce()

    def global_cross_forward(self, share_user, source_item, target_item):
        all_embeddings_share = torch.cat([share_user, source_item, target_item], dim=0)
        user_G_emb_sha, item_G_emb_sha = self.lightGCN_forward(
            all_embeddings_share,
            self.cross_payload['cross_adj'],
            self.user_num,
            self.s_item_num + self.t_item_num,
        )
        item_G_emb_sha_s, item_G_emb_sha_t = torch.split(item_G_emb_sha, [self.s_item_num, self.t_item_num])
        return user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t

    def get_G_emb(self):
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        share_user = self.share_user_embedding(self.user_index)

        all_emb_s = torch.cat([self.source_user_embedding(self.user_index), source_item], dim=0)
        all_emb_t = torch.cat([self.target_user_embedding(self.user_index), target_item], dim=0)
        lgcn_user_s, lgcn_item_s = self.lightGCN_forward(all_emb_s, self.s_adj, self.user_num, self.s_item_num)
        lgcn_user_t, lgcn_item_t = self.lightGCN_forward(all_emb_t, self.t_adj, self.user_num, self.t_item_num)

        if self.use_prototype_denoise:
            user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t = self.prototype_refinement(
                share_user, source_item, target_item, self.cross_payload
            )
        else:
            user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t = self.global_cross_forward(
                share_user, source_item, target_item
            )

        return lgcn_user_s, lgcn_item_s, lgcn_user_t, lgcn_item_t, user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t

    def lightGCN_forward(self, all_embeddings, norm_adj_matrix, n_users, n_items):
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.sparse_dropout(norm_adj_matrix, self.dropout), all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return torch.split(lightgcn_all_embeddings, [n_users, n_items])

    def forward(
        self,
        user,
        source_pos_item,
        source_neg_item,
        target_pos_item,
        target_neg_item,
        source_pop_item,
        target_pop_item,
    ):
        if self.restore_user_s is not None or self.restore_item_s is not None:
            self.restore_user_s, self.restore_item_s = None, None

        user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t, \
            user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t = self.get_G_emb()

        source_user_feature_spe = user_G_emb_s[user]
        target_user_feature_spe = user_G_emb_t[user]
        user_feature_share = user_G_emb_sha[user]

        source_item_pos_feature_spe = item_G_emb_s[source_pos_item]
        target_item_pos_feature_spe = item_G_emb_t[target_pos_item]
        source_item_neg_feature_spe = item_G_emb_s[source_neg_item]
        target_item_neg_feature_spe = item_G_emb_t[target_neg_item]
        source_item_pos_feature_share = item_G_emb_sha_s[source_pos_item]
        target_item_pos_feature_share = item_G_emb_sha_t[target_pos_item]
        source_item_neg_feature_share = item_G_emb_sha_s[source_neg_item]
        target_item_neg_feature_share = item_G_emb_sha_t[target_neg_item]

        episode_batch = user.shape[0]
        pos_label = torch.ones(episode_batch).long().cuda()
        neg_label = -torch.ones(episode_batch).long().cuda()
        loss_cc_A = self.loss_cos(source_item_pos_feature_spe.detach(), source_item_pos_feature_share, pos_label) + \
            self.loss_cos(source_item_neg_feature_spe.detach(), source_item_pos_feature_share, neg_label)
        loss_cc_B = self.loss_cos(target_item_pos_feature_spe.detach(), target_item_pos_feature_share, pos_label) + \
            self.loss_cos(target_item_neg_feature_spe.detach(), target_item_pos_feature_share, neg_label)
        loss_cc = loss_cc_A + loss_cc_B

        y_S = (torch.ones(episode_batch, 2) / 2.0).cuda()
        y_A = torch.ones(episode_batch).long().cuda()
        y_B = torch.zeros(episode_batch).long().cuda()

        source_spe_score = self.domain_cls(source_user_feature_spe)
        target_spe_score = self.domain_cls(target_user_feature_spe)
        share_score = self.domain_cls(user_feature_share)
        loss_share_kld = self.loss_KLD(F.log_softmax(share_score, dim=1), y_S)
        loss_domain_CLS_A = self.loss_CE(source_spe_score, y_A)
        loss_domain_CLS_B = self.loss_CE(target_spe_score, y_B)
        loss_dom = loss_share_kld + loss_domain_CLS_A + loss_domain_CLS_B

        pop_user_feature = self.pop_encoder(user_feature_share)
        int_user_feature = self.int_encoder(user_feature_share)
        pos_source_int_score = self.predict_dot(int_user_feature, source_item_pos_feature_share)
        pos_target_int_score = self.predict_dot(int_user_feature, target_item_pos_feature_share)
        pos_source_pop_score = self.predict_dot(pop_user_feature, source_item_pos_feature_share)
        pos_target_pop_score = self.predict_dot(pop_user_feature, target_item_pos_feature_share)

        conf_weight_A = torch.exp(source_pop_item.unsqueeze(1))
        conf_weight_B = torch.exp(target_pop_item.unsqueeze(1))
        int_weight_A = torch.exp(torch.ones_like(conf_weight_A) - source_pop_item.unsqueeze(1))
        int_weight_B = torch.exp(torch.ones_like(conf_weight_B) - target_pop_item.unsqueeze(1))

        loss_conf_A = -torch.log(conf_weight_A * torch.exp(pos_source_pop_score / self.temp)).mean() + \
            torch.log(torch.exp(pop_user_feature @ item_G_emb_sha_s.T / self.temp).sum(1) + 1e-8).mean()
        loss_conf_B = -torch.log(conf_weight_B * torch.exp(pos_target_pop_score / self.temp)).mean() + \
            torch.log(torch.exp(pop_user_feature @ item_G_emb_sha_t.T / self.temp).sum(1) + 1e-8).mean()
        loss_int_A = -torch.log(int_weight_A * torch.exp(pos_source_int_score / self.temp)).mean() + \
            torch.log(torch.exp(int_user_feature @ item_G_emb_sha_s.T / self.temp).sum(1) + 1e-8).mean()
        loss_int_B = -torch.log(int_weight_B * torch.exp(pos_target_int_score / self.temp)).mean() + \
            torch.log(torch.exp(int_user_feature @ item_G_emb_sha_t.T / self.temp).sum(1) + 1e-8).mean()
        loss_pd = loss_conf_A + loss_conf_B + loss_int_A + loss_int_B

        source_user_feature_fused = self.emb_fusion(source_user_feature_spe, int_user_feature, pop_user_feature, domain='source')
        target_user_feature_fused = self.emb_fusion(target_user_feature_spe, int_user_feature, pop_user_feature, domain='target')

        regularization_loss = self.reliability_regularization(
            source_user_feature_spe,
            target_user_feature_spe,
            user_feature_share,
            source_user_feature_fused,
            target_user_feature_fused,
            int_user_feature,
            pop_user_feature,
            fusion_mode=self.fusion_mode,
        ) if self.use_reliability_regularization and self.lambda_regularization > 0 else torch.zeros(
            (), device=user.device, dtype=source_user_feature_spe.dtype
        )

        source_item_pos_feature_fused = self.agg_s(torch.cat([source_item_pos_feature_spe, source_item_pos_feature_share], dim=1))
        source_item_neg_feature_fused = self.agg_s(torch.cat([source_item_neg_feature_spe, source_item_neg_feature_share], dim=1))
        target_item_pos_feature_fused = self.agg_t(torch.cat([target_item_pos_feature_spe, target_item_pos_feature_share], dim=1))
        target_item_neg_feature_fused = self.agg_t(torch.cat([target_item_neg_feature_spe, target_item_neg_feature_share], dim=1))

        pos_source_score = self.predict_dot(source_user_feature_fused, source_item_pos_feature_fused)
        neg_source_score = self.predict_dot(source_user_feature_fused, source_item_neg_feature_fused)
        pos_target_score = self.predict_dot(target_user_feature_fused, target_item_pos_feature_fused)
        neg_target_score = self.predict_dot(target_user_feature_fused, target_item_neg_feature_fused)

        loss_bpr_source = torch.mean(torch.nn.functional.softplus(neg_source_score - pos_source_score))
        loss_bpr_target = torch.mean(torch.nn.functional.softplus(neg_target_score - pos_target_score))

        source_user_feature_ego = self.source_user_embedding(user)
        target_user_feature_ego = self.target_user_embedding(user)
        source_item_pos_feature_ego = self.source_item_embedding(source_pos_item)
        target_item_pos_feature_ego = self.target_item_embedding(target_pos_item)
        source_item_neg_feature_ego = self.source_item_embedding(source_neg_item)
        target_item_neg_feature_ego = self.target_item_embedding(target_neg_item)
        user_feature_share_ego = self.share_user_embedding(user)
        reg_loss = (1 / 2) * (
            source_user_feature_ego.norm(2).pow(2)
            + target_user_feature_ego.norm(2).pow(2)
            + source_item_pos_feature_ego.norm(2).pow(2)
            + target_item_pos_feature_ego.norm(2).pow(2)
            + source_item_neg_feature_ego.norm(2).pow(2)
            + target_item_neg_feature_ego.norm(2).pow(2)
            + user_feature_share_ego.norm(2).pow(2)
        ) / float(len(user))
        loss_rec = loss_bpr_source + loss_bpr_target + self.opt['reg_weight'] * reg_loss

        loss = loss_rec + loss_dom + self.opt['lambda1'] * loss_cc + self.opt['lambda2'] * loss_pd
        if self.use_reliability_regularization:
            loss = loss + self.lambda_regularization * regularization_loss
        if self.use_prototype_denoise:
            loss = loss + self.lambda_refine * self.prototype_refinement.last_refinement_loss
            loss = loss + self.lambda_edge_denoise * self.prototype_refinement.last_edge_denoise_loss
        return loss

    def get_evaluate_embedding(self):
        if self.restore_user_s is None or self.restore_item_s is None:
            self.restore_user_s, self.restore_item_s, self.restore_user_t, self.restore_item_t, self.restore_user_sha, \
                self.restore_item_sha_s, self.restore_item_sha_t = self.get_G_emb()

        source_user_feature_spe = self.restore_user_s[self.user_index]
        target_user_feature_spe = self.restore_user_t[self.user_index]
        source_item_feature_spe = self.restore_item_s[self.source_item_index]
        target_item_feature_spe = self.restore_item_t[self.target_item_index]

        user_feature_share = self.restore_user_sha[self.user_index]
        source_item_feature_share = self.restore_item_sha_s[self.source_item_index]
        target_item_feature_share = self.restore_item_sha_t[self.target_item_index]

        pop_user_feature = self.pop_encoder(user_feature_share)
        int_user_feature = self.int_encoder(user_feature_share)

        source_user_feature_fused = self.emb_fusion(source_user_feature_spe, int_user_feature, pop_user_feature, domain='source')
        target_user_feature_fused = self.emb_fusion(target_user_feature_spe, int_user_feature, pop_user_feature, domain='target')

        source_item_feature_fused = self.agg_s(torch.cat([source_item_feature_spe, source_item_feature_share], dim=1))
        target_item_feature_fused = self.agg_t(torch.cat([target_item_feature_spe, target_item_feature_share], dim=1))

        return source_user_feature_fused, source_item_feature_fused, target_user_feature_fused, target_item_feature_fused
