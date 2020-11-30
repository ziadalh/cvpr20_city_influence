import logging
import random
import warnings

import numpy as np
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class Forecaster(object):
    """
    Base class for a forecaster model
    """
    def __init__(self):
        self.success = True

    @classmethod
    def is_multivar(cls):
        return False

    def fit(self, yt):
        raise NotImplementedError()

    def forecast(self, nsteps):
        raise NotImplementedError()

    def summary(self, short=False):
        return self.__class__.__name__


class RandomForecaster(Forecaster):
    """
    Forecasts next steps as random number samples from normal distribution
        with mean and std learned from the time series
    """
    def __init__(self, seed=None):
        super(RandomForecaster, self).__init__()
        self.u = 0.
        self.s = 1.
        self.seed = seed

    def fit(self, yt):
        self.u = yt.mean()
        self.s = yt.std()
        return True

    def forecast(self, nsteps):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        yp = np.random.normal(loc=self.u, scale=self.s, size=nsteps)
        return yp


class MeanForecaster(Forecaster):
    """
    Forecasts next steps as the overall mean of the time series
    """
    def __init__(self):
        super(MeanForecaster, self).__init__()
        self.mean_v = 0.
        self.std_v = 1.

    def fit(self, yt):
        self.mean_v = yt.mean()
        self.std_v = yt.std()
        return True

    def forecast(self, nsteps):
        yp = np.repeat(self.mean_v, nsteps)
        return yp


class LastForecaster(Forecaster):
    """
    Forecasts next steps as the last observed value in the time series
    """
    def __init__(self):
        super(LastForecaster, self).__init__()
        self.last = 0

    def fit(self, yt):
        self.last = yt[-1]
        return True

    def forecast(self, nsteps):
        return np.repeat(self.last, nsteps)


class DriftForecaster(Forecaster):
    """
    Forecasts next steps using the main linear drift/trend learned from the
        time series
    """
    def __init__(self):
        super(DriftForecaster, self).__init__()

    def fit(self, yt):
        self.m = (yt[-1] - yt[0]) / float(len(yt) - 1)
        self.yn = yt[-1]
        return True

    def forecast(self, nsteps):
        return self.yn + (self.m * np.arange(1, nsteps+1))


class SeasonalNaiveForecaster(Forecaster):
    """
    Forecasts next steps using the last seasonal values of the time series
    """
    def __init__(self, seasonal_period=12):
        super(SeasonalNaiveForecaster, self).__init__()
        self.m = seasonal_period

    def fit(self, yt):
        self.yt = yt
        return True

    def forecast(self, nsteps):
        d = self.m * (int((nsteps-1)/self.m) + 1)
        return self.yt[-d:-d+nsteps]


class ExponentialSmoothing(Forecaster):
    """
    Exponential Smoothing model.
    Used in the Fashion Forward paper: https://arxiv.org/abs/1705.06394
    """
    def __init__(self):
        super(ExponentialSmoothing, self).__init__()
        self.reset()

    def reset(self):
        self.model = None
        self.fitted = None
        self.success = True
        self.smoothing_level = None
        self.ts_len = 0

    def fit(self, yt, **kwargs):
        """
        kwargs: smoothing_level=None, optimized=True, start_params=None,
                initial_level=None, use_brute=True
        """
        self.reset()
        self.ts_len = len(yt)
        self.model = sm.tsa.SimpleExpSmoothing(yt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = self.model.fit(**kwargs)

        if self.fitted.mle_retvals is not None:
            self.success = self.fitted.mle_retvals.success

        self.smoothing_level = self.fitted.params['smoothing_level']
        logger.debug(self.summary())
        return self.success

    def forecast(self, nsteps):
        if self.model is None:
            logger.error('model is None. use fit() to create a model')
            raise ValueError('model is None')

        start_idx = self.ts_len
        end_idx = start_idx + nsteps - 1
        res = self.model.predict(self.fitted.params,
                                 start=start_idx, end=end_idx)
        return res

    def summary(self, short=False):
        s = 'Exp()'
        if self.fitted is not None:
            s += ': level={:.5f}'.format(self.smoothing_level)
            if not short:
                s += ', aic={:.3f}, bic={:.3f}, sse={:.3f}'.format(
                            self.fitted.aic, self.fitted.bic, self.fitted.sse)
                if self.fitted.mle_retvals is not None:
                    s += ', success={}'.format(self.fitted.mle_retvals.success)

        return s
