import math
import logging
from abc import ABCMeta, abstractmethod
from functools import partial
import sys

import torch
import torch.nn.functional as F
import numpy as np


logger = logging.getLogger(__name__)

"""
Contains a set of metrics to evaluate forecast model performance
"""


def get_metric(metric_name, axis=0,
               output_transform=lambda x: x,
               target_transform=lambda x: x):
    metric_name = metric_name.lower()
    if metric_name == 'mae':
        return MAE(axis=axis,
                   output_transform=output_transform,
                   target_transform=target_transform)
    elif metric_name == 'mape':
        return MAPE(axis=axis,
                    output_transform=output_transform,
                    target_transform=target_transform)
    else:
        raise ValueError('unknown metric [{}]'.format(metric_name))


class Metric(object):
    """
    Base class for all Metrics.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 output_transform=lambda x: x,
                 target_transform=lambda x: x):
        self._output_transform = output_transform
        self._target_transform = target_transform
        # self.reset()

    @abstractmethod
    def reset(self):
        """
        Resets the metric to to it's initial state.
        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output, target):
        """
        Updates the metric's state using the passed batch output.
        This is called once for each batch.
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.
        This is called at the end of each epoch.
        """
        pass


class Metrics(Metric):
    """
    Class for a group of metrics calculated on the same data
    """
    def __init__(self, summarize=False):
        super(Metrics, self).__init__()
        self.metrics = {}
        self.summarize = summarize

    def add(self, key, metric):
        self.metrics[key] = metric

    def keys(self):
        return self.metrics.keys()

    def reset(self):
        for k in self.metrics.keys():
            self.metrics[k].reset()

    def update(self, output, target):
        for k in self.metrics.keys():
            self.metrics[k].update(output, target)

    def compute(self):
        values = {}
        for k in self.metrics.keys():
            values[k] = self.metrics[k].compute()

        if self.summarize:
            for k in self.metrics.keys():
                if torch.is_tensor(values[k]):
                    values[k] = torch.mean(values[k])

        return values


class TimeSeriesMetric(Metric):
    """
    Time Series Metric
    """
    def __init__(self,
                 axis=0,
                 output_transform=lambda x: x,
                 target_transform=lambda x: x):
        super(TimeSeriesMetric, self).__init__(output_transform,
                                               target_transform)
        # set the axis of the time series
        self.axis = axis
        self.reset()

    def reset(self):
        self.pred = []
        self.targ = []

    def update(self, output, target):
        outp = self._output_transform(output)
        targ = self._target_transform(target)
        self.pred.append(outp)
        self.targ.append(targ)

    @abstractmethod
    def compute(self):
        pass


class MAE(TimeSeriesMetric):
    """
    Mean Absolute Error
    """
    def compute(self):
        pred_t = torch.cat(self.pred, dim=self.axis)
        targ_t = torch.cat(self.targ, dim=self.axis)
        return torch.mean(torch.abs(targ_t - pred_t), dim=self.axis)


class MAPE(TimeSeriesMetric):
    """
    Mean Absolute Percentage Error
    """
    def compute(self):
        pred_t = torch.cat(self.pred, dim=self.axis)
        targ_t = torch.cat(self.targ, dim=self.axis)
        denom = targ_t.sign() * targ_t.abs().clamp(min=1e-8)
        error = torch.abs((targ_t - pred_t)/denom)
        return 100. * torch.mean(error, dim=self.axis)
