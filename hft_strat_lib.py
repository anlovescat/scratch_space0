import sys
sys.path.append('/home/mingyuan/Projects/AndrewStock')
from config import numpy_data_dirs, etf_numpy_data_dir, if_numpy_data_dir
import numpy as np
import datetime
import glob
import os
from hft_signal_lib import *
#import scipy.stats as stats

def get_summary_simple(position, price_diff):
    pnl = np.cumsum(position * price_diff)
    volume = np.sum(np.abs(np.diff(position))) + np.abs(position[0])
    result = np.array([(pnl[-1], np.min(pnl), np.max(pnl),
                        np.min(position), np.max(position), volume)],
                      dtype = [('total_pnl', 'f'), ('min_pnl', 'f'),
                               ('max_pnl', 'f'), ('min_position', 'd'),
                               ('max_position', 'd'), ('volume', 'd')])
    return result



def sim_mv_simple(data, sample_freq = 120 * 5, ticksize=0.2, round_factor=10):
    mid = midpoint(data)
    ind = subsample_nsample(data, sample_freq)
    mid_sample = mid[ind]
    position = np.round(np.round((mid_sample - mid_sample[0]) / 0.2)[:-1] * (-1) / round_factor)
    price_diff = np.diff(mid_sample)
    return get_summary_simple(position, price_diff)


def sim_mv_poslim(data, pos_lim=10, sample_freq = 120 * 5, ticksize=0.2, round_factor=10):
    mid = midpoint(data)
    ind = subsample_nsample(data, sample_freq)
    mid_sample = mid[ind]
    position = np.round(np.round((mid_sample - mid_sample[0]) / 0.2)[:-1] * (-1) / round_factor)
    position[np.where(position > pos_lim)] = pos_lim
    position[np.where(position < -pos_lim)] = -pos_lim
    price_diff = np.diff(mid_sample)
    return get_summary_simple(position, price_diff)
    
def sim_mv_poslim_reset(data, pos_lim=10, sample_freq = 120 * 5, ticksize=0.2, round_factor=10):
    mid = midpoint(data)
    ind = subsample_nsample(data, sample_freq)
    mid_sample = mid[ind]
    position = np.round(np.round((mid_sample - mid_sample[0]) / 0.2)[:-1] * (-1) / round_factor)
    reset_ind = np.where((position > pos_lim) | (position < -pos_lim))[0]
    while(len(reset_ind) > 0) :
        min_ind = np.min(reset_ind)
        position[min_ind:] = position[min_ind:] - position[min_ind]
        reset_ind = np.where((position > pos_lim) | (position < -pos_lim))[0]
    price_diff = np.diff(mid_sample)
    return get_summary_simple(position, price_diff)

def sim_trigger(data, valuation, thresh):
    mid = midpoint(data)
    prediction = valuation - mid
    buy_ind = np.where( prediction >= thresh)
    sell_ind = np.where( prediction <= -thresh)
    trade_act = np.zeros(len(data))
    trade_act[buy_ind] = 1
    trade_act[sell_ind] = -1
    position = np.zeros(len(data))
    position[0] = trade_act[0]
    for ii in range(1, len(data)) :
        if trade_act[ii] == 0:
            position[ii] = position[ii-1]
        else :
            position[ii] = trade_act[ii]
    position = position[:-1]
    price_diff = np.diff(mid)
    return get_summary_simple(position, price_diff)



