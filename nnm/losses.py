import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FCoherenceLoss(nn.Module):
    """
    Coherence forecasting loss
    """
    def __init__(self, c_idx_list, n_idx, dim=-1,
                 c_wg=1.0, n_wg=1.0, reg_loss=nn.MSELoss,
                 ):
        super(FCoherenceLoss, self).__init__()
        if c_wg is not None and c_wg <= 0:
            raise ValueError('c_wg must be > 0')

        if n_wg is not None and n_wg <= 0:
            raise ValueError('n_wg must be > 0')

        if c_wg is None and n_wg is None:
            raise ValueError('only one of the weighs can be None not both')

        if not isinstance(c_idx_list, list):
            c_idx_list = [c_idx_list]

        self.c_idx_list = [torch.as_tensor(c_idx_list[i],
                                           dtype=torch.long).clone()
                           for i in range(len(c_idx_list))]
        self.n_idx = torch.as_tensor(n_idx.clone(), dtype=torch.long)
        self.dim = dim
        self.c_wg = c_wg
        self.n_wg = n_wg
        self.c_loss = reg_loss()
        self.n_loss = reg_loss()

    def cuda(self, device=None):
        super(FCoherenceLoss, self).cuda(device)
        self.c_idx_list = [self.c_idx_list[i].cuda(device)
                           for i in range(len(self.c_idx_list))]
        self.n_idx = self.n_idx.cuda(device)

    def cpu(self):
        super(FCoherenceLoss, self).cpu()
        self.c_idx_list = [self.c_idx_list[i].cpu()
                           for i in range(len(self.c_idx_list))]
        self.n_idx = self.n_idx.cpu()

    def forward(self, input, target):
        # coherence loss
        in_mv = []
        ta_mv = []
        for i in range(len(self.c_idx_list)):
            i_inp_st = []
            i_tar_st = []
            i_inp_st = input.index_select(
                                dim=self.dim, index=self.c_idx_list[i]
                                ).mean(dim=self.dim, keepdim=True)
            i_tar_st = target.index_select(
                                dim=self.dim, index=self.c_idx_list[i]
                                ).mean(dim=self.dim, keepdim=True)
            in_mv.append(i_inp_st)
            ta_mv.append(i_tar_st)

        in_mv_t = torch.cat(in_mv, dim=self.dim)
        ta_mv_t = torch.cat(ta_mv, dim=self.dim)
        loss_c = self.c_loss(in_mv_t, ta_mv_t)
        # non coherence loss
        loss_n = self.n_loss(
                        input.index_select(dim=self.dim, index=self.n_idx),
                        target.index_select(dim=self.dim, index=self.n_idx))
        # weighted loss
        c_wg = self.c_wg
        if c_wg is None:
            c_wg = len(self.n_idx) / float(in_mv_t.size(self.dim))

        n_wg = self.n_wg
        if n_wg is None:
            n_wg = len(in_mv_t.size(self.dim)) / float(len(self.n_idx))

        loss = (c_wg * loss_c) + (n_wg * loss_n)
        return loss
