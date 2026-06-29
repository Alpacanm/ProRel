import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityRegularizationLoss(nn.Module):
    def __init__(self, opt):
        super(ReliabilityRegularizationLoss, self).__init__()
        self.alpha = opt.get('regularization_alpha', 1.0)
        self.beta = opt.get('regularization_beta', 0.1)
        self.sigma = opt.get('regularization_sigma', 1.0)
        self.temperature = opt.get('regularization_temp', opt.get('temp', 0.05))
        self.min_hsic_batch = opt.get('min_hsic_batch', 16)
        self.fusion_mode = opt.get('fusion_mode', 'ours')
        self.last_losses = {}

    def _zero(self, x):
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def _info_nce(self, x, y):
        if x.size(0) <= 1:
            return self._zero(x)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        logits = torch.matmul(x, y.t()) / self.temperature
        labels = torch.arange(x.size(0), device=x.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2.0

    def _kernel_matrix(self, x):
        x = F.normalize(x, p=2, dim=1)
        return torch.exp((torch.matmul(x, x.t()) - 1.0) / self.sigma)

    def _hsic_from_kernel(self, kx, ky):
        m = kx.size(0)
        kxy = torch.mm(kx, ky)
        h = torch.trace(kxy) / (m ** 2) + torch.mean(kx) * torch.mean(ky) - 2.0 * torch.mean(kxy) / m
        return h * (m / (m - 1)) ** 2

    def _hsic(self, x, y):
        if x.size(0) < self.min_hsic_batch:
            return self._zero(x)
        return self._hsic_from_kernel(self._kernel_matrix(x), self._kernel_matrix(y))

    def forward(self, spe_s, spe_t, share, fused_s, fused_t, int_emb, pop_emb, fusion_mode=None):
        fusion_mode = self.fusion_mode if fusion_mode is None else fusion_mode

        l_pres_sep = self._info_nce(share, spe_s) + self._info_nce(share, spe_t)
        l_comp_sep = self._hsic(spe_s, share) + self._hsic(spe_t, share) + self._hsic(spe_s, spe_t)

        l_pres_fuse = self._info_nce(fused_s, spe_s) + self._info_nce(fused_t, spe_t)
        if fusion_mode == 'ours':
            l_comp_fuse = self._hsic(fused_s, int_emb) + self._hsic(fused_s, pop_emb) + \
                          self._hsic(fused_t, int_emb) + self._hsic(fused_t, pop_emb)
        else:
            l_comp_fuse = self._zero(spe_s)

        l_pres_cross = self._info_nce(fused_s, share) + self._info_nce(fused_t, share)
        l_comp_cross = self._hsic(fused_s, fused_t)

        l_pres = l_pres_sep + l_pres_fuse + l_pres_cross
        l_comp = l_comp_sep + l_comp_fuse + l_comp_cross
        loss = self.alpha * l_pres + self.beta * l_comp

        self.last_losses = {
            'pres_sep': l_pres_sep.detach(),
            'comp_sep': l_comp_sep.detach(),
            'pres_fuse': l_pres_fuse.detach(),
            'comp_fuse': l_comp_fuse.detach(),
            'pres_cross': l_pres_cross.detach(),
            'comp_cross': l_comp_cross.detach(),
            'pres': l_pres.detach(),
            'comp': l_comp.detach(),
            'loss': loss.detach(),
        }
        return loss
