from __future__ import division
import time
import math
import logging
from datetime import datetime

import torch

from forecast.metrics import Metrics

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    A trainer class to train and evaluate a neural model
    for time series forecasting
    """
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_metric = None
        self.val_metric = None
        self.batch_metric = None
        self.training = True
        self.ret_metrics = None

    def cuda(self, activate=True):
        if activate:
            self.model.cuda()
            self.criterion.cuda()
        else:
            self.model.cpu()
            self.criterion.cpu()

    def train_mode(self, activate=True):
        self.training = activate
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def _model_step_train(self, input, target):
        output = self.model(input)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss

    def _tensors_to_cuda(self, input, target, activate=True):
        if activate:
            input, target = input.cuda(), target.cuda()
        else:
            input, target = input.cpu(), target.cpu()

        return input, target

    def train(self, train_loader, max_iter=-1,
              cuda=False, epoch=-1, log_freq=10, plotter=None, verbose=True):
        self.cuda(cuda)
        self.train_mode(True)
        self.reset_metric()
        n_samples = len(train_loader)
        max_iter = n_samples if max_iter <= 0 else min(max_iter, n_samples)
        if log_freq < 1 and log_freq > 0:
            log_freq = math.ceil(log_freq * max_iter)

        logger.debug('log_freq: {}, max_iter: {}, n_samples: {}'.format(
            log_freq, max_iter, n_samples))
        t_estart = datetime.now()
        t_iter_end = time.time()
        losses = 0
        for i, (input, target) in enumerate(train_loader):
            t_data = time.time() - t_iter_end
            if i == max_iter:
                break
            if self.batch_metric is not None:
                self.batch_metric.reset()

            t_iter = time.time()
            input, target = self._tensors_to_cuda(input, target, cuda)
            output, loss = self._model_step_train(input, target)
            losses += loss.item()

            # update metrics / time / print
            self.update_metric(output, target)
            batch_m = ''
            if self.batch_metric is not None:
                self.batch_metric.update(output, target)
                batch_mval = self.batch_metric.compute()
                for mk in batch_mval.keys():
                    batch_m += '{}: {:.2f} '.format(mk, batch_mval[mk])

            t_iter = time.time() - t_iter
            if i % log_freq == 0:
                if verbose:
                    logger.info('E[{}][{:5.2f}%][{:4d}/{}] T[D:{:.3f},M:{:.3f}] '
                                'Loss: {:.5f}\t{}'.format(
                                 epoch, 100.0 * ((i+1) / max_iter), i+1,
                                 max_iter, t_data, t_iter,
                                 loss.item(), batch_m))
                if plotter is not None:
                    plotter.metrics_values({'batch_loss': loss.item()},
                                           (epoch * max_iter) + i+1,
                                           tag='train', std_log=False,
                                           nsml_commit=True)
                    if getattr(self.criterion, 'loss_info', None) is not None:
                        plotter.metrics_values(self.criterion.loss_info,
                                               (epoch * max_iter) + i+1,
                                               tag='train', std_log=False,
                                               nsml_commit=True)

            t_iter_end = time.time()

        if verbose:
            logger.info('E[{}] calculate epoch performance metrics'
                        .format(epoch))

        loss_avg = losses / max_iter
        metric_v = self.compute_metric()
        if verbose:
            logger.info('Epoch {} training finished in {:}. '
                        'Average loss: {:.5f}'.format(
                         epoch, datetime.now() - t_estart, loss_avg))

        return loss_avg, metric_v

    def _model_step_eval(self, input, target):
        output = self.model(input)
        loss = self.criterion(output, target)
        return output, loss

    def eval(self, val_loader, max_iter=-1, cuda=False, epoch=-1, log_freq=10):
        self.cuda(cuda)
        self.train_mode(False)
        self.reset_metric()
        n_samples = len(val_loader)
        max_iter = n_samples if max_iter <= 0 else min(max_iter, n_samples)
        if log_freq < 1 and log_freq > 0:
            log_freq = math.ceil(log_freq * max_iter)

        logger.debug('log_freq: {}'.format(log_freq))
        t_estart = datetime.now()
        t_iter_end = time.time()
        losses = 0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                t_data = time.time() - t_iter_end
                if i == max_iter:
                    break
                input, target = self._tensors_to_cuda(input, target, cuda)
                t_iter = time.time()
                output, loss = self._model_step_eval(input, target)
                losses += loss.item()
                # update metrics / time / print
                self.update_metric(output, target)
                t_iter = time.time() - t_iter
                if i % log_freq == 0:
                    logger.info('E[{}][{:5.2f}%][{:4d}/{}] T[D:{:.3f},M:{:.3f}] '
                                'Loss: {:.5f}'.format(
                                 epoch, 100.0 * ((i+1) / max_iter), i+1,
                                 max_iter, t_data, t_iter,
                                 loss.item()))
                t_iter_end = time.time()

        logger.info('E[{}] calculate epoch performance metrics'.format(epoch))
        loss_avg = losses / max_iter
        metric_v = self.compute_metric()
        logger.info('Epoch {} evaluation finished in {}. '
                    'Average loss: {:.5f}'.format(
                     epoch, datetime.now() - t_estart, loss_avg))
        return loss_avg, metric_v

    def add_metric(self, metric, training=True):
        if training:
            self.train_metric = metric
        else:
            self.val_metric = metric

    def add_batch_metric(self, metric):
        if not isinstance(metric, Metrics):
            raise ValueError('batch metric should be an instance of Metrics')

        self.batch_metric = metric

    def reset_metric(self):
        if self.training and self.train_metric is not None:
            self.train_metric.reset()

        if not self.training and self.val_metric is not None:
            self.val_metric.reset()

    def update_metric(self, output, target):
        if self.training and self.train_metric is not None:
            self.train_metric.update(output, target)

        if not self.training and self.val_metric is not None:
            self.val_metric.update(output, target)

    def compute_metric(self):
        values = None
        if self.training and self.train_metric is not None:
            values = self.train_metric.compute()

        if not self.training and self.val_metric is not None:
            values = self.val_metric.compute()

        return values
