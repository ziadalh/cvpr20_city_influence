import logging

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def _get_torch_tensor(a):
    if torch.is_tensor(a):
        return a

    if type(a) == np.ndarray:
        return torch.from_numpy(a)
    elif type(a) == list:
        return torch.FloatTensor(a)
    else:
        raise ValueError('unknown type {}'.format(type(a)))


class TSDataset(Dataset):
    """
    A dataset class for multi-variate time series
    """
    def __init__(self, yts_lags, yts_target, lags=1, flatten=True):
        """
        yts_lags: (N x T) array
            multi-variate time series where lags are used.
            N is the time series length and T is the number of trajectories
        yts_target: (N x K) array
            multi-variate time series of target forecasts
            N is the time series length and K is the number of targets
        lags: int or list of lists
            the lags to use from each time series. len(lags) == T
            if int then the same sequence of lags (from 1 up to lags)
            is used from each time series
        flatten: bool
            whether to output lags as one flat tensor
        """
        super(TSDataset, self).__init__()
        self.flatten = flatten
        self.yts_lags = _get_torch_tensor(yts_lags).float()
        self.yts_target = _get_torch_tensor(yts_target).float()
        if self.yts_target.dim() == 1:
            self.yts_target = self.yts_target.view(-1, 1)

        if type(lags) == list:
            assert len(lags) == self.yts_lags.size(1)
            lags_t = []
            for li in range(len(lags)):
                lags_t.append(_get_torch_tensor(lags[li]).long())

            self.lags = lags_t
        else:
            lags_t = []
            for i in range(self.yts_lags.size(1)):
                lags_t.append(_get_torch_tensor(list(range(lags))).long()
                              + 1)

            self.lags = lags_t

        self.maxlag = 0
        for li in self.lags:
            if self.maxlag < li.max():
                self.maxlag = li.max()

    def __len__(self):
        ln = self.yts_lags.size(0) - self.maxlag
        return ln

    def __getitem__(self, index):
        idx = index + self.maxlag
        ylags = self.get_lags(index)

        y = self.yts_target[idx, :]

        return ylags, y.float()

    def get_slice(self, index):
        return self.yts_lags[index, :].clone()

    def get_lags(self, index):
        """
        Returns the set of lags relative to an index
        """
        idx = index + self.maxlag
        ylags = []
        for ti in range(self.yts_lags.size(1)):
            ylags.append(self.yts_lags[idx - self.lags[ti], ti])

        if self.flatten:
            ylags = torch.cat(ylags, dim=0)

        return ylags

    def append(self, ys_lags, ys_target=None):
        """
        Adds data points to the dataset
        """
        ys_lags_t = ys_lags
        if not torch.is_tensor(ys_lags):
            ys_lags_t = _get_torch_tensor(ys_lags)

        if ys_lags_t.dim() == 1:
            ys_lags_t = ys_lags_t.unsqueeze(0)

        ys_lags_t = ys_lags_t.type_as(self.yts_lags)
        self.yts_lags = torch.cat([self.yts_lags, ys_lags_t], dim=0)

        if ys_target is not None:
            ys_target_t = ys_target
            if not torch.is_tensor(ys_target):
                ys_target_t = _get_torch_tensor(ys_target)

            if ys_target_t.dim() == 1:
                ys_target_t = ys_target_t.unsqueeze(0)

            ys_target_t = ys_target_t.type_as(self.yts_target)
            self.yts_target = torch.cat([self.yts_target, ys_target_t], dim=0)

    def clone(self):
        c_lags = [l.clone() for l in self.lags]
        return TSDataset(yts_lags=self.yts_lags.clone(),
                         yts_target=self.yts_target.clone(),
                         lags=c_lags,
                         flatten=self.flatten)
