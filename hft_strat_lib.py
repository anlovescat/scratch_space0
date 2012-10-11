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

def get_summary_simple_stoploss(position, price_diff, stoploss):
    pnl = np.cumsum(position * price_diff)
    idx = np.where(pnl < stoploss)[0]
    if len(idx) > 0:
        stop_idx = np.min(idx)
        position[(stop_idx+1):] = 0.0
        pnl = np.cumsum(position * price_diff)
    volume = np.sum(np.abs(np.diff(position))) + np.abs(position[0])
    result = np.array([(pnl[-1], np.min(pnl), np.max(pnl),
                        np.min(position), np.max(position), volume)],
                      dtype = [('total_pnl', 'f'), ('min_pnl', 'f'),
                               ('max_pnl', 'f'), ('min_position', 'd'),
                               ('max_position', 'd'), ('volume', 'd')])
    return result


def get_summary_simple_drawdown(position, price_diff, drawdown):
    pnl = np.cumsum(position * price_diff)
    running_max = np.maximum.accumulate(pnl)
    idx = np.where(pnl - running_max < drawdown)[0]
    if len(idx) > 0:
        stop_idx = np.min(idx)
        position[(stop_idx+1):] = 0.0
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

def sim_trigger_stoploss(data, valuation, thresh, stoploss):
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
    return get_summary_simple_stoploss(position, price_diff, stoploss)


def sim_trigger_drawdown(data, valuation, thresh, drawdown):
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
    return get_summary_simple_drawdown(position, price_diff, drawdown)

def is_filled(limit_price, side, data_idx, data, latency, time_in_force):
    in_action_begin = data_idx + latency
    in_action_end = min(data_idx + latency + time_in_force, len(data))
    trade_price = data['lastprice']
    trade_price_in_action = trade_price[in_action_begin:in_action_end]
    if side > 0:
        filled = np.any(trade_price_in_action < limit_price)
        if filled:
            fill_ts = np.min(np.where(trade_price_in_action < limit_price)) + in_action_begin
        else :
            fill_ts = in_action_end
        return filled, fill_ts
    else :
        filled = np.any(trade_price_in_action > limit_price)
        if filled:
            fill_ts = np.min(np.where(trade_price_in_action > limit_price)) + in_action_begin
        else :
            fill_ts = in_action_end
        return filled, fill_ts

def get_limit_price(valuation, edge, ticksize):
    bid_price = np.floor((valuation - edge) / ticksize) * ticksize
    ask_price = np.floor((valuation + edge) / ticksize + 0.5) * ticksize
    return bid_price, ask_price

def get_summary_simple_drawdown_limitorder(target_position, data, drawdown,
                                           bid_price, ask_price,
                                           latency, time_in_force, iteration=1000):
    signal = np.array([0] + np.diff(target_position).tolist())
    trade_idx = np.where(np.abs(signal) > 0)[0]
    order_qty = np.zeros(len(signal))
    filled_qty = np.zeros(len(signal))
    filled_price = np.zeros(len(signal))
    prefilled_qty = np.zeros(len(signal))
    prefilled_price = np.zeros(len(signal))
    previous_idx = 0
    buy_order_expiring = []
    sell_order_expiring = []
    #import pdb; pdb.set_trace()
    for idx in trade_idx:
        cur_position = np.sum(filled_qty[:idx])
        side = ((signal[idx] * cur_position) <= 0) * signal[idx]
        buy_order_expiring = [ts for ts in buy_order_expiring if ts > idx]
        sell_order_expiring = [ts for ts in sell_order_expiring if ts > idx]
        if len(buy_order_expiring) >= iteration and side > 0:
            continue
        if len(sell_order_expiring) >= iteration and side < 0:
            continue
        if side == 0:
            continue
        elif side > 0:
            limit_price = bid_price[idx]
            order_qty[idx] = signal[idx] 
            filled, fill_ts = is_filled(limit_price, side, idx+1, data, latency, time_in_force)
            if filled:
                prefilled_qty[idx] = order_qty[idx]
                prefilled_price[idx] = limit_price
                filled_qty[fill_ts] += order_qty[idx]
                filled_price[fill_ts] = limit_price
            buy_order_expiring.append( fill_ts )
        else:
            limit_price = ask_price[idx]
            order_qty[idx] = signal[idx] 
            filled, fill_ts = is_filled(limit_price, side, idx+1, data, latency, time_in_force)
            if filled:
                prefilled_qty[idx] = order_qty[idx]
                prefilled_price[idx] = limit_price
                filled_qty[fill_ts] += order_qty[idx]
                filled_price[fill_ts] = limit_price
            sell_order_expiring.append( fill_ts )

    cum_position = np.cumsum(prefilled_qty)
    result = get_blotter_pnl(order_qty, prefilled_qty, prefilled_price, cum_position, data,
                             drawdown)
    return result

def get_summary_simple_drawdown_limitorder_deprecated(target_position, data, drawdown,
                                           bid_price, ask_price,
                                           latency, time_in_force):
    signal = np.array([0] + np.diff(target_position).tolist())
    trade_idx = np.where(np.abs(signal) > 0)[0]
    order_qty = np.zeros(len(signal))
    filled_qty = np.zeros(len(signal))
    filled_price = np.zeros(len(signal))
    previous_idx = 0
    buy_order_outstanding = False
    buy_order_expiring = 0
    sell_order_outstanding = False
    sell_order_expiring = 0
    for idx in trade_idx:
        cur_position = np.sum(filled_qty[:idx])
        side = ((signal[idx] * cur_position) <= 0) * signal[idx]
        if idx >= buy_order_expiring:
            buy_order_outstanding = False
        if idx >= sell_order_expiring:
            sell_order_outstanding = False
        if buy_order_outstanding and side > 0:
            continue
        if sell_order_outstanding and side < 0:
            continue
        if side == 0:
            continue
        elif side > 0:
            limit_price = bid_price[idx]
            order_qty[idx] = signal[idx] 
            filled, fill_ts = is_filled(limit_price, side, idx, data, latency, time_in_force)
            if filled:
                filled_qty[fill_ts] += order_qty[idx]
                filled_price[fill_ts] = limit_price
                buy_order_outstanding = True
                buy_order_expiring = fill_ts
        else:
            limit_price = ask_price[idx]
            order_qty[idx] = signal[idx] 
            filled, fill_ts = is_filled(limit_price, side, idx, data, latency, time_in_force)
            if filled:
                filled_qty[fill_ts] += order_qty[idx]
                filled_price[fill_ts] = limit_price
                sell_order_outstanding = True
                sell_order_expiring = fill_ts


    cum_position = np.cumsum(filled_qty)
    result = get_blotter_pnl(order_qty, filled_qty, filled_price, cum_position, data,
                             drawdown)
    return result


def get_blotter_pnl(order_qty, filled_qty, filled_price, cum_position, data, drawdown):
    #import pdb; pdb.set_trace()
    mid = midpoint(data)
    cash = np.sum(filled_qty * filled_price) * (-1.0)
    open_cash = cum_position[-1] * mid[-1]
    pnl = cash + open_cash
    pnl_t = np.cumsum(cum_position[:-1] * np.diff(mid))
    spread = np.cumsum((mid - filled_price) * filled_qty)
    pnl_t = spread[1:] + pnl_t
    assert abs(pnl - pnl_t[-1]) < 0.01

    running_max = np.maximum.accumulate(pnl_t)
    idx = np.where(pnl_t - running_max < drawdown)[0]
    if len(idx) > 0:
        stop_idx = np.min(idx)
        cum_position[(stop_idx+1):] = 0.0
        pnl_t = np.cumsum(cum_position[:-1] * np.diff(mid))
        order_qty[(stop_idx+1):] = 0.0
        filled_qty[(stop_idx+1):] = 0.0
        spread = np.cumsum((mid - filled_price) * filled_qty)
        pnl_t = spread[1:] + pnl_t

    order_volume = np.sum(np.abs(order_qty))
    trade_volume = np.sum(np.abs(filled_qty))

    
    result = np.array([(pnl_t[-1], np.min(pnl_t), np.max(pnl_t),
                        np.min(cum_position), np.max(cum_position), trade_volume,
                        order_volume, trade_volume * 1.0 / order_volume)],
                      dtype = [('total_pnl', 'f'), ('min_pnl', 'f'),
                               ('max_pnl', 'f'), ('min_position', int),
                               ('max_position', int), ('volume', int),
                               ('order_volume', int), ('fill_ratio', float)])
    return result
        


def is_filled_simple(bid_limit, ask_limit, trade_price):
    if trade_price < bid_limit:
        return 1
    elif trade_price > ask_limit:
        return -1
    else :
        return 0

def sim_market_making_simple(data, valuation, edge, edge_buffer, latency, ticksize):
    mid = midpoint(data)
    bid_limitprice, ask_limitprice = get_limit_price(valuation, edge, ticksize)
    back_off_factor = 100
    beg_idx = np.min(np.where(~np.isnan(valuation)))
    bid_quote = bid_limitprice[beg_idx]
    ask_quote = ask_limitprice[beg_idx]
    filled_qty = np.zeros(len(mid))
    filled_price = np.zeros(len(mid))
    bid_edge_adjust = 0.0
    ask_edge_adjust = 0.0
    #import pdb;pdb.set_trace()
    for ii in range(beg_idx+1, len(mid)):
        filled = is_filled_simple(bid_quote, ask_quote, data['lastprice'][ii])
        if filled > 0:
            filled_qty[ii] = filled
            filled_price[ii] = bid_quote
            bid_edge_adjust = ticksize * back_off_factor
            ask_edge_adjust = 0.0
        elif filled < 0:
            filled_qty[ii] = filled
            filled_price[ii] = ask_quote
            ask_edge_adjust = ticksize * back_off_factor
            bid_edge_adjust = 0.0
        if valuation[ii] - bid_quote <= edge - edge_buffer:
            bid_quote = bid_limitprice[ii]
        if ask_quote - valuation[ii] <= edge - edge_buffer:
            ask_quote = ask_limitprice[ii]
        bid_quote -= bid_edge_adjust
        ask_quote += ask_edge_adjust
        
    order_qty = filled_qty.copy()
    cum_position = np.cumsum(filled_qty)
    result = get_blotter_pnl(order_qty, filled_qty, filled_price, cum_position, data, drawdown=-100000.0)
    import pdb;pdb.set_trace()
    return result



def sim_trigger_drawdown_limitorder(data, valuation, thresh, drawdown,
                                    edge, latency, time_in_force, ticksize):
    mid = midpoint(data)
    prediction = valuation - mid
    bid_limitprice, ask_limitprice = get_limit_price(valuation, edge, ticksize)
    #bid_limitprice = data['bid1']
    #ask_limitprice = data['ask1']
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
    result = get_summary_simple_drawdown_limitorder(position, 
                                                    data, drawdown,
                                                    bid_limitprice, ask_limitprice,
                                                    latency, time_in_force)
    return result



        
