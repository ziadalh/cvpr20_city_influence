import logging
import argparse
import os
import os.path as osp
from types import SimpleNamespace as SN
from datetime import datetime as dt
from collections import defaultdict
from functools import partial
import pickle
# import multiprocessing as mp

from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp

import forecast.baselines as fbase
import forecast.neuralnet as fnn
from forecast import metrics
import utils.logging as ulog
import utils.misc as umisc

NCPUS = mp.cpu_count()
np.set_printoptions(precision=4, edgeitems=5)
# mp.set_sharing_strategy('file_system')


def to_float_tensor(x):
    return torch.from_numpy(x).float()


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
                        description='forecast style trends across the world. '
                                    'paper: https://arxiv.org/abs/2004.01316')
    parser.add_argument('--f_traj', type=str, default=None,
                        help='trajectories file path', required=True)
    parser.add_argument('--d_output', type=str, default=None,
                        help='output directory path', required=True)
    parser.add_argument('--nproc', type=int, default=NCPUS,
                        help='number of processes')

    t_start = dt.now()
    opt = parser.parse_args()
    # check files and directories
    if not osp.isfile(opt.f_traj):
        raise ValueError('file was not found: {}'.format(opt.f_traj))

    if not osp.isdir(opt.d_output):
        os.makedirs(opt.d_output)
    else:
        logger.warn('directory already exists: {}'.format(opt.d_output))

    # setup logger
    ulog.setup_logging(root_logfile=os.path.join(opt.d_output, 'output.log'),
                       level='INFO')
    logger.info(opt)
    # load trajectories
    logger.info('load trajectories from file [{}]'.format(opt.f_traj))
    with open(opt.f_traj, 'rb') as f:
        traj_data = pickle.load(f)

    # set variables
    units_traj = traj_data['units_traj']
    units_list = traj_data['units_list']
    units_ids = traj_data['units_ids']
    num_units = len(units_list)
    num_vars = units_traj[0].shape[1]
    time_range = traj_data['time_range']
    num_steps = len(time_range)
    time_res = 'weeks'
    seasonal_periods = 52
    seed = 42
    ntest = 26
    umisc.set_random_seed(seed)
    # make sure all values are positive
    for ui in range(num_units):
        units_traj[ui] = np.clip(units_traj[ui], 0, None)

    # show loaded traj. info.
    logger.info('trends loaded for {} cities with {} styles each'
                .format(num_units, num_vars))

    # setup forecasting metrics
    fmetrics = metrics.Metrics(summarize=False)
    fmetrics.add('mae', metrics.MAE(axis=0,
                                    output_transform=to_float_tensor,
                                    target_transform=to_float_tensor))
    fmetrics.add('mape', metrics.MAPE(axis=0,
                                      output_transform=to_float_tensor,
                                      target_transform=to_float_tensor))
    logger.info('evaluation metrics: {}'.format(list(fmetrics.keys())))

    # time range info
    logger.info('time range: [{} - {}], {} steps in {}'
                .format(time_range[0], time_range[-1],
                        len(time_range), time_res))
    logger.info('forecast test starts at {} for {} steps'
                .format(time_range[-ntest], ntest))

    # forecasters errors
    models_error = defaultdict(lambda: defaultdict(list))

    ##########################################
    # forecast style trajectories [baselines]
    forecaster = {
        'Gaussian': partial(fbase.RandomForecaster, seed=seed),
        'Seasonal': partial(fbase.SeasonalNaiveForecaster,
                            seasonal_period=seasonal_periods),
        'Mean': fbase.MeanForecaster,
        'Last': fbase.LastForecaster,
        'Drift': fbase.DriftForecaster,
        'FashionForward (EXP)': fbase.ExponentialSmoothing,
        }
    logger.info('baseline forecasters: {}'.format(list(forecaster.keys())))
    # for each baseline model
    for fi in forecaster.keys():
        nfails = 0
        fmetrics.reset()
        logger.info('forecasting with [{}]'.format(fi))
        # for each unit
        for ui in tqdm(range(num_units)):
            fmetrics.reset()
            succ_ls, pred_ls = [], []
            # for each variable
            for vi in range(num_vars):
                # fit and forecast a trend
                model = forecaster[fi]()
                succ_ls.append(model.fit(units_traj[ui][:-ntest, vi]))
                pred_ls.append(model.forecast(ntest))

            nfails += num_vars - np.array(succ_ls).sum()
            st_pred = np.stack(pred_ls, axis=-1)
            st_pred = np.clip(st_pred, 0, None)
            # get forecast errors
            fmetrics.update(st_pred, units_traj[ui][-ntest:, :])
            models_error[fi][ui] = fmetrics.compute()

        if nfails > 0:
            logger.info('{} failed in {} (%{:.1f}) of the models'
                        .format(fi, nfails,
                                100. * (nfails/(num_units*num_vars))))

    ################################################
    # forecast style trajectories [influence-based]
    logger.info('influence-based forecaster')
    finame = 'Influence-based (Ours)'
    infl_forecaster = partial(
                        fnn.NN_ICM,
                        lags=[1, 2, seasonal_periods], nproc=opt.nproc)
    mvstyle_error = defaultdict(lambda: defaultdict(list))
    # for each style
    for vi in tqdm(range(num_vars)):
        fmetrics.reset()
        # collect style si from all cities
        sib_traj = np.zeros((len(time_range), num_units))
        for ui in range(num_units):
            sib_traj[:, ui] = units_traj[ui][:, vi]

        # fit and forecast for style si across cities
        f_model = infl_forecaster()
        succ = f_model.fit(sib_traj[:-ntest, :])
        st_pred = f_model.forecast(ntest)
        nfails += 0 if succ else 1

        st_pred = np.clip(st_pred, 0, None)
        if np.isnan(st_pred).sum() > 0:
            logger.warn(('{} nan values found in forecasts!'
                        ' set those to zeros.').format(
                            np.isnan(st_pred).sum()))
            st_pred[np.isnan(st_pred)] = 0

        fmetrics.update(st_pred, sib_traj[-ntest:, :])
        mvstyle_error[finame][vi] = fmetrics.compute()

    for ui in range(num_units):
        ui_ms = defaultdict(list)
        for mi in fmetrics.keys():
            for vi in range(num_vars):
                ui_ms[mi].append(mvstyle_error[finame][vi][mi][ui])

            ui_ms[mi] = np.array(ui_ms[mi])

        models_error[finame][ui] = ui_ms

    ###############################
    # summarize forecasting errors
    fb_error = defaultdict(lambda: defaultdict(float))
    for fi in models_error.keys():
        for mi in fmetrics.keys():
            for ui in range(num_units):
                fb_error[fi][mi] += models_error[fi][ui][mi].mean()

            fb_error[fi][mi] /= float(num_units)

    fb_error_tbl, fb_error_df = umisc.print_table_from_dict(fb_error)
    logger.info('\nForecast errors across city-style trends\n{}'
                .format(fb_error_tbl))

    # save to csv file
    df_all = fb_error_df
    logger.info('save results to csv file...')
    f_csv = os.path.join(opt.d_output, 'models_forecast.csv')
    df_all = df_all.applymap(lambda x: x.item() if torch.is_tensor(x) else x)
    df_all.to_csv(f_csv, sep=',', header=True,
                  index=True, index_label='model', encoding='utf-8')

    logger.info('Finished in {}'.format(dt.now() - t_start))
