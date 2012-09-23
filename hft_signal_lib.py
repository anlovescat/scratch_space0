import sys
sys.path.append('/home/mingyuan/Projects/AndrewStock')
from config import numpy_data_dirs, etf_numpy_data_dir, if_numpy_data_dir
import numpy as np
import datetime
import glob
import os
#import scipy.stats as stats


IFHV = 'IFHV'
IF16 = 'IF16'
IF17 = 'IF17'
IF18 = 'IF18'
IF19 = 'IF19'

ETF1 = '159901'
ETF2 = '159902'
ETF3 = '160706'
ETF4 = '510050'
ETF5 = '510180'

def clean_data(data):
    good_idx = np.where((data['ask1'] > data['bid1']) \
                            & (data['ask1'] > 0.0) \
                            & (data['bid1'] > 0.0) \
                            & (data['bidsize1'] > 0) \
                            & (data['asksize1'] > 0))
    return data[good_idx]

def separate_by_date(data):
    dates = list(np.unique(data['date']))
    dates.sort()
    result = {}
    for adate in dates:
        idx = np.where(data['date'] == adate)
        result[adate.strftime('%Y%m%d')] = data[idx]
    return result


def load_data(product, yearmonth):
    if 'IF' in product:
        folder = if_numpy_data_dir
    else :
        folder = etf_numpy_data_dir
    filename = os.path.join(folder, '_'.join([product, yearmonth])+'.npy')
    data = np.load(filename)
    data = clean_data(data)
    result = separate_by_date(data)
    return result

def midpoint(data):
    return (data['ask1'] + data['bid1']) / 2.0

def booksignal(data, maxlevel, decay):
    assert maxlevel > 0, 'maxlevel needs to be greater than 0'
    assert maxlevel <= 3, 'maxlevel needs to be smaller than 4'
    assert decay > 0, 'decay needs to be greater than 0, usually smaller than 1, but not requited'
    bid_numerator = 0.0
    bid_denominator = 0.0
    ask_numerator = 0.0
    ask_denominator = 0.0
    for ii in range(1, maxlevel+1) :
        bid = 'bid%d'%ii
        bidsize = 'bidsize%d'%ii
        ask = 'ask%d'%ii
        asksize = 'asksize%d'%ii
        bid_numerator += data[bid] * data[bidsize] * (decay ** (ii - 1))
        bid_denominator += data[bidsize] * (decay ** (ii - 1))
        ask_numerator += data[ask] * data[asksize] * (decay ** (ii - 1))
        ask_denominator += data[asksize] * (decay ** (ii - 1))
    bid_price = bid_numerator / bid_denominator
    ask_price = ask_numerator / ask_denominator
    value = (bid_price * ask_denominator + ask_price * bid_denominator) / (ask_denominator + bid_denominator)
    return value

def ema(nparray, decay):
    new_array = [nparray[0]]
    for ii in range(1, len(nparray)) :
        new_array.append(new_array[ii-1] * decay + nparray[ii] * (1.0 - decay))
    return np.array(new_array)

def tradesign(data):
    trade_sign = np.zeros(len(data))
    trade_sign[np.where(data['lastdirection'] == 'B')] = 1.0
    trade_sign[np.where(data['lastdirection'] == 'S')] = -1.0
    return trade_sign

def signedvolume(data):
    mid = midpoint(data)
    delta_mid = np.array([0.0] + list(np.diff(mid)))
    return np.sign(delta_mid) * data['volume']

def signedvolume_ema(data, decay):
    sv = signedvolume(data)
    volume_ema = ema(data['volume'], decay)
    volume_ema[np.where(volume_ema <= 0.0)] = 1.0
    return ema(sv, decay) / volume_ema

def vwap_ema(data, decay):
    volume = data['volume']
    volume[np.where(volume <= 0.0)] = 1.0
    volume_price = data['volume'] * data['lastprice']
    volume_price_cumsum = np.cumsum(volume_price)
    volume_cumsum = np.cumsum(volume_price)
    volume_price_cumsum_ema = ema(volume_price_cumsum, decay)
    volume_cumsum_ema = ema(volume_cumsum, decay)
    return (volume_price_cumsum - volume_price_cumsum_ema) / (volume_cumsum - volume_cumsum_ema)



def tradesignema(data, decay):
    trade_sign = tradesign(data)
    return ema(trade_sign, decay)

def tradesignvolume_ema(data, decay):
    trade_sign = tradesign(data)
    trade_sign_volume = trade_sign * data['volume']
    trade_sign_volume_ema = ema(trade_sign, decay)
    volume_ema = ema(data['volume'], decay)
    volume_ema[np.where(volume_ema <= 0.0)] = 1.0
    return trade_sign_volume_ema / volume_ema

def spread(data):
    return data['ask1'] - data['bid1']


def generate_offset_index(n, offsets, base = None, burn_in = 0):
    assert isinstance(offsets, list), 'offsets must be list'
    max_offsets = max(offsets)
    if base is None:
        ind = range(burn_in, n)
        base_ind = ind[:(n-max_offsets)]
    else :
        ind = base
        base_ind = [ii for ii in ind if (ii < (n-max_offsets)) and (ii >= burn_in)]
    offsets_ind = []
    for offset in offsets:
        offset_ind = list(np.array(base_ind) + offset)
        offsets_ind.append(offset_ind)
    return base_ind, offsets_ind

def subsample_nsample(data, every_nsample):
    ind = range(0, len(data), every_nsample)
    return ind

def subsample_clock(data, clock_list):
    ind = []
    clock_list.sort()
    for time in clock_list:
        idx = np.where(data['time'] == time)[0]
        if len(idx) > 0:
            ind.append(max(idx))
    return ind

def subsample_bigvolume(data, volume_thresh):
    ind = np.where(data['volume'] >= volume_thresh)[0]
    return list(ind)




    


