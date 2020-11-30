import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def print_table_from_dict(d, transpose=True, column_types=None,
                          floatfmt='.5f', tablefmt='psql'):
    """
    Returns a nice table (as string) for numbers in nested dictionary d.
    Arguments:
        d (dict): data dictionary
            d['model']['metric'] = value
            row labels are the first level of keys
            column labels are the second level of keys
        transpose (bool): transpose the table
        column_types (dict) : set the type of each column
    returns a table as string
    """
    import tabulate as tab
    from tabulate import tabulate
    import pandas as pd
    if tablefmt not in tab.tabulate_formats:
        raise ValueError('tablefmt [{}] is unkown, it must be one of: {}'
                         .format(tablefmt, tab.tabulate_formats))

    df = pd.DataFrame(d)
    if transpose:
        df = df.transpose()

    if column_types is not None:
        df = df.astype(column_types)

    # make a nice table
    tblstr = tabulate(df, headers='keys',
                      tablefmt=tablefmt, floatfmt=floatfmt)
    return tblstr, df


def set_random_seed(seed):
    """
    Sets the random seed for multiple libs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def mv_minmax_scale(yts, value_range=(-1, 1), param=None, reverse=False):
    """
    Scales values in each column of yts to a certain range
    """
    squeeze = False
    if len(yts.shape) == 1:
        yts = yts.reshape(-1, 1)
        squeeze = True

    if param is not None:
        value_range = param['range']
        ys_minmax = param['minmax']
    else:
        ys_minmax = (np.min(yts, axis=0), np.max(yts, axis=0))
        param = {'range': value_range, 'minmax': ys_minmax}

    ys_range = ys_minmax[1] - ys_minmax[0]
    if not reverse:
        ys_range[ys_range == 0] = 1
        ys_std = (yts - ys_minmax[0].reshape([1, -1])) / ys_range.reshape([1, -1])
        ys_sc = ys_std * (value_range[1] - value_range[0]) + value_range[0]
    else:
        ys_std = (yts - value_range[0]) / (value_range[1] - value_range[0])
        ys_sc = ys_std * ys_range.reshape((1, -1)) + ys_minmax[0]

    if squeeze:
        ys_sc = ys_sc.squeeze()

    return ys_sc, param
