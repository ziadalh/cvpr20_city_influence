import logging
import warnings
import copy
import numbers
from functools import partial
from collections import defaultdict
from types import SimpleNamespace as SN
# import multiprocessing as mp

import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from forecast.baselines import Forecaster
import nnm.factory as nnf
from nnm.trainer import Trainer
import nnm.mlps as mlps
import nnm.losses as losses
from forecast.dataset import TSDataset
from forecast import metrics
import utils.misc as umisc

logger = logging.getLogger(__name__)
# mp.set_sharing_strategy('file_system')


def granger_causality(x1, x2, maxlag, addconst=True,
                      p_thr=0.05, verbose=False):
    """
    Runs granger causality test (x2 granger causes x1?)
    returns minimum p_value, corresponding lag, vote support
    """
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc = sm.tsa.stattools.grangercausalitytests(
                                        np.stack([x1, x2], axis=1),
                                        maxlag=maxlag, addconst=addconst,
                                        verbose=False)
    # get all p_values
    ntests = 4
    l_stat = np.zeros((maxlag, ntests))
    for li in gc.keys():
        pi = 0
        for si in gc[li][0].keys():
            l_stat[int(li)-1][pi] = gc[li][0][si][1]
            pi += 1

    l_vt = (l_stat < p_thr).sum(axis=1)
    if l_vt.sum() > 0:
        # pick one with the largest majority
        lag = np.argmax(l_vt) + 1
        pv = np.min(l_stat[lag-1])
        support = l_vt[lag-1] / float(ntests)
    else:
        # return minimum overall
        ind = np.unravel_index(np.argmin(l_stat, axis=None), l_stat.shape)
        lag = ind[0] + 1
        pv = l_stat[ind[0], ind[1]]
        support = 0.

    return pv, lag, support


def granger_causality_lags_and_indices(
                        yts, yt_target_idx, maxlag, minlag=1,
                        addconst=True, p_thr=0.05, support=0.9,
                        verbose=False):
    """
    A wrapper for granger_causality to get lags for multiple
        influencing variables in yts
    yts: an array (N x T) where n is the number of steps and T number of
        variables
    yt_target_idx: int, the index of the target variable
    """
    n_var = yts.shape[1]
    lag_inf = np.zeros((n_var))
    for vi in range(n_var):
        if vi == yt_target_idx:
            continue

        pv, lag, vi_support = granger_causality(
                                yts[:, yt_target_idx], yts[:, vi],
                                maxlag=maxlag,
                                addconst=addconst, p_thr=p_thr,
                                verbose=verbose)
        lvi = (pv <= p_thr) * (vi_support >= support) * lag
        lag_inf[vi] = lvi

    lag_inf[lag_inf < minlag] = 0
    if lag_inf.sum() > 0:
        selected_vars = np.nonzero(lag_inf)[0].tolist()
        lags = []
        for li in range(len(lag_inf)):
            if lag_inf[li] > 0:
                lags.append(int(lag_inf[li]))

        return selected_vars, lags
    else:
        return [], []


class NN_ICM(Forecaster):
    """
    Influence coherent model for forecasting style trends
    paper: https://arxiv.org/abs/2004.01316
    """
    def __init__(self, lags=[1, 2], nhidden=2,
                 nepochs=500, batch_size=8, ninit=3, nval=4,
                 nproc=1, cuda=False, seed=42):

        super(NN_ICM, self).__init__()
        if lags is not None:
            if isinstance(lags, numbers.Number) and lags < 1:
                raise ValueError('number of lags should be > 0')
            elif isinstance(lags, list) and len(lags) == 0:
                raise ValueError('number of lags should be > 0')

        if nhidden is not None and nhidden < 0:
            raise ValueError('number of hidden units should be >= 0')

        self.lags = lags
        self.nhidden = nhidden
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.nval = nval
        self.ninit = ninit
        self.nproc = nproc
        self.cuda = cuda
        self.seed = seed
        self.optimizer_opts = SN(optimizer='Adam', lr=1e-2, weight_decay=1e-8)
        self.scale_range = (-1, 1)
        self.nworkers = 0
        # metric for corss validation
        fnc = partial(torch.as_tensor, dtype=torch.float,
                      device=torch.device('cpu'))
        self.metric = metrics.get_metric('mae',
                                         output_transform=fnc,
                                         target_transform=fnc)
        if type(lags) == list:
            self.v_lags = lags
        else:
            self.v_lags = list(range(1, lags + 1))

        self.model = None
        self.criterion = None
        self.optimizer = None
        self._idx_coherent = None
        self._idx_noncoherent = None

    @classmethod
    def is_multivar(cls):
        return True

    def _get_criterion(self):
        return losses.FCoherenceLoss(
                            c_idx_list=self._idx_coherent,
                            n_idx=self._idx_noncoherent,
                            c_wg=1.0,
                        )

    def reset(self, mlp_lags):
        """
        Rests the model params
        """
        l_nunints_list = self._get_nunits_list(mlp_lags)
        self.model = mlps.MLPGroup(
                            l_nunints_list, mlp_lags,
                            activation_fnc=torch.nn.Sigmoid,
                            bias=True,
                            )

        self.optimizer = nnf.get_optimizer(self.model.parameters(),
                                           self.optimizer_opts)
        self.criterion = self._get_criterion()
        if self.metric is not None:
            self.metric.reset()

    def _get_influence_lags_and_indices(self, yts):
        nvars = yts.shape[1]
        params = []
        for vi in range(nvars):
            params.append((yts.copy(), vi))

        # setup causality function with common parameters
        gc_fnc = partial(granger_causality_lags_and_indices,
                         maxlag=8, minlag=1,
                         p_thr=0.05, support=0.9,
                         addconst=True, verbose=False)

        with mp.Pool(self.nproc) as p:
            out_ls = p.starmap(gc_fnc, params)

        # list of all lags per variable
        vars_idxs = []
        vars_lags = []
        for vi in range(nvars):
            vars_idxs.append(len(self.v_lags) * [vi] + out_ls[vi][0])
            vars_lags.append(self.v_lags + out_ls[vi][1])

        var_lags_dict = defaultdict(set)
        for vi in range(nvars):
            for vj in range(len(vars_idxs[vi])):
                vij = vars_idxs[vi][vj]
                vij_l = vars_lags[vi][vj]
                var_lags_dict[vij] = var_lags_dict[vij].union(set([vij_l]))

        ds_lags = []
        ds_lag_shift = []
        shift = 0
        for vi in range(nvars):
            ds_lags.append(torch.as_tensor(list(var_lags_dict[vi]),
                                           dtype=torch.long))
            ds_lag_shift.append(shift)
            shift += len(ds_lags[vi])

        mlp_lags = []
        for vi in range(nvars):
            vi_lag_idx = []
            for j in range(len(vars_idxs[vi])):
                vij = vars_idxs[vi][j]
                vij_shift = ds_lag_shift[vij]
                vij_l = vars_lags[vi][j]
                idx = int((ds_lags[vij] == vij_l).nonzero().squeeze())
                vi_lag_idx.append(vij_shift + idx)

            mlp_lags.append(vi_lag_idx)

        return mlp_lags, ds_lags

    def _get_nunits_list(self, mlp_lags):
        hidden_nunits = [1]
        if self.nhidden is not None and self.nhidden > 0:
            hidden_nunits = [self.nhidden] + hidden_nunits

        l_nunints_list = []
        for mi in range(len(mlp_lags)):
            if self.nhidden is None:
                mi_nh = max(1, int((len(mlp_lags[mi]) + 1) / 2.))
                l_nunints_list.append([len(mlp_lags[mi])] + [mi_nh]
                                      + hidden_nunits)
            else:
                l_nunints_list.append([len(mlp_lags[mi])] + hidden_nunits)

        return l_nunints_list

    def _fit(self, dataset, yt_val=None, metric=None,
             nepochs=200, batch_size=8, num_workers=0, verbose=False):
        b_state = None
        b_metric = None
        if yt_val is not None:
            # set dataset for self.forecast to work
            self.ds = dataset
            nval_steps = yt_val.shape[0]
            if metric is None:
                fnc = partial(torch.as_tensor, dtype=torch.float,
                              device=torch.device('cpu'))
                metric = metrics.get_metric('mse',
                                            output_transform=fnc,
                                            target_transform=fnc)

        # training
        self.model.train()
        # get training data loader
        ds_loader = torch.utils.data.DataLoader(
                            dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, sampler=None)
        trainer = Trainer(model=self.model,
                          criterion=self.criterion,
                          optimizer=self.optimizer)
        # set training metric
        trainer.add_metric(metric, training=True)
        for ei in range(nepochs):
            ei_loss, ei_metric = trainer.train(ds_loader, cuda=self.cuda,
                                               epoch=ei, verbose=verbose)
            if yt_val is not None:
                # eval forecast on validation data
                yt_fc = self.forecast(nval_steps, self.cuda)
                metric.reset()
                metric.update(yt_fc, yt_val)
                ei_metric = metric.compute()

            if ei_metric is not None:
                if isinstance(ei_metric, numbers.Number):
                    ei_metric_m = ei_metric
                else:
                    ei_metric_m = ei_metric.mean()
            else:
                # if no metric is specified then use epoch loss
                ei_metric_m = ei_loss

            if b_metric is None or ei_metric_m < b_metric:
                b_state = {'model': copy.deepcopy(self.model.state_dict()),
                           'optim': copy.deepcopy(self.optimizer.state_dict()),
                           'epoch': ei}

                b_metric = ei_metric_m

        return b_state, b_metric

    def fit(self, yts, nepochs=None, batch_size=None, ninit=None):
        umisc.set_random_seed(self.seed)
        mlp_lags, ds_lags = self._get_influence_lags_and_indices(yts)
        b_state = None
        b_metric = None
        nepochs = nepochs if nepochs is not None else self.nepochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        ninit = ninit if ninit is not None else self.ninit
        nworkers = self.nworkers
        n_vars = yts.shape[1]
        n_steps = yts.shape[0]
        lags_l = [ds_lags]
        mlp_l = [mlp_lags]
        if self.nval > 0:
            yt_val = yts[-self.nval:, :].copy()
            yt_tr = yts[:-self.nval, :].copy()
        else:
            yt_val = None
            yt_tr = yts.copy()

        if self.scale_range is not None:
            yt_tr, prm = umisc.mv_minmax_scale(yt_tr, self.scale_range)
            self.scale_param = prm

        yt_tr_in = yt_tr.copy()
        yt_tr_gt = yt_tr.copy()
        self._idx_coherent = torch.arange(yt_tr_gt.shape[1], dtype=torch.long)
        self._idx_noncoherent = torch.arange(yt_tr.shape[1], dtype=torch.long)

        for li in range(len(lags_l)):
            self.lags = lags_l[li]
            n_lags = 0
            for lti in range(len(self.lags)):
                n_lags += len(self.lags[lti])

            for i in range(ninit):
                # init model
                self.reset(mlp_l[li])
                metric = self.metric
                dataset = TSDataset(yt_tr_in, yt_tr_gt,
                                    lags=self.lags, flatten=True)
                # train
                i_state, i_metric = self._fit(dataset=dataset,
                                              yt_val=yt_val,
                                              metric=metric,
                                              nepochs=nepochs,
                                              batch_size=batch_size,
                                              num_workers=nworkers)
                if b_metric is None or i_metric < b_metric:
                    b_state = i_state
                    b_metric = i_metric
                    b_state['n_lags'] = n_lags
                    b_state['lags'] = copy.deepcopy(self.lags)
                    b_state['mlp_lags'] = copy.deepcopy(mlp_l[li])

        self.lags = b_state['lags']
        if self.scale_range is not None:
            yts, _ = umisc.mv_minmax_scale(yts, param=self.scale_param)

        self.ds = TSDataset(yts, yts, lags=self.lags, flatten=True)
        # reset model to make sure it has the correct arch. given selected lags
        self.nlags = n_lags
        self.nvars = n_vars
        self.reset(b_state['mlp_lags'])
        self.model.load_state_dict(b_state['model'])
        self.optimizer.load_state_dict(b_state['optim'])
        self.model.cpu()
        return True

    def forecast(self, nsteps, use_cuda=False, start_step=None):
        if self.model is None:
            raise ValueError('Model is None. Call fit() first')

        self.model.eval()
        f_ds = self.ds.clone()
        if start_step is None:
            start_step = len(f_ds)

        if start_step < 0:
            start_step = len(f_ds) + start_step

        y = []
        for i in range(nsteps):
            # get lags
            lags_i = f_ds.get_lags(start_step + i)
            if use_cuda:
                lags_i = lags_i.cuda()

            # forecast
            yi = self.model(lags_i)
            yi = yi.detach()
            if yi.dim() == 1:
                yi = yi.unsqueeze(0)

            y.append(yi)
            # append forecast to dataset
            f_ds.append(yi)

        y = torch.cat(y, dim=0).cpu().numpy()
        if self.scale_range is not None:
            # reverse the scaling
            y, _ = umisc.mv_minmax_scale(y, param=self.scale_param,
                                         reverse=True)

        return y
