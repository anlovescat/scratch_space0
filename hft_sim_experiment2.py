from hft_sim_lib import *
from hft_signal_lib import *
from hft_strat_lib import *
import numpy as np
from scipy.optimize import nnls
from collections import defaultdict

cost = 0.15 #- 0.1 * 0.6
drawdown = -1000.0

ticksize = 0.2
edge = 0.9
edge_buffer_frac = 0.5
edge_rem_frac = 0.1
latency = 0

def get_signals(data):
    decays = [0.5, 0.7, 0.8, 0.9]
    decay_names = [str(ff).replace('0.', '_') for ff in decays]
    mid = midpoint(data) 
    book = booksignal(data, 1, 0.9) - mid
    last = data['lastprice'] - mid
    #emabook = [ema(book, ff) - mid for ff in decays]
    vwap = [vwap_ema(data, ff) - mid for ff in decays]
    #trends = [book / ema(book, ff) - 1.0 for ff in decays]
    trsigns = [tradesignema(data, ff) for ff in decays]
    trsvolumes = [tradesignvolume_ema(data, ff) for ff in decays]
    sv = [signedvolume_ema(data, ff) for ff in decays]
    
    signals = {'book': book, 'last': last}
    #signals.update(dict(zip(['bkema'+dn for dn in decay_names], emabook)))
    signals.update(dict(zip(['vwap'+dn for dn in decay_names], vwap)))
    #signals.update(dict(zip(['trends'+dn for dn in decay_names], trends)))
    signals.update(dict(zip(['trsign'+dn for dn in decay_names], trsigns)))
    signals.update(dict(zip(['trsvolume'+dn for dn in decay_names], trsvolumes)))
    signals.update(dict(zip(['signedv'+dn for dn in decay_names], sv)))
    return signals

def construct_valuation(data, beta):
    signals = get_signals(data)
    valuation = midpoint(data)
    for key in signals.keys():
        valuation += signals[key] * beta[key]
    return valuation


def run_regression(result, dates):
    mid = []
    future = []
    sample_freq = 120
    future_horizon = 120 * 0.5
    signals = defaultdict(list)
    for adate in dates:
        data = result[adate]
        signals_tmp = get_signals(data)
        mid_tmp = midpoint(data)
        base_ind, offset_inds = generate_offset_index(len(data), [future_horizon],
                                                      base = subsample_nsample(data, sample_freq),
                                                      burn_in = sample_freq)
        for key in signals_tmp.keys():
            signals[key] += list(signals_tmp[key][base_ind])
        mid +=(list(mid_tmp[base_ind]))
        future +=(list(mid_tmp[offset_inds[0]]))
    names = signals.keys()
    X = np.c_[[np.array(signals[nm]) for nm in names]].T
    Y = np.array(future) - np.array(mid)
    #beta, res, rank, s = np.linalg.lstsq(X, Y)
    beta, rrs = nnls(X, Y)
    valuation = np.dot(X, beta)
    print 'Regression Dates  : %s - %s'%(min(dates), max(dates))
    print 'Regression Results: %s'%(str(dict(zip(names, beta))))
    return names, dict(zip(names, list(beta))), np.abs(valuation).mean()

def optimize_thresh(result, dates, reg_param):
    names, beta, val_std = reg_param
    thresh = edge + np.arange(-0.1, 0.12, 0.02) #np.arange(0.5, 3.5, 0.1) * val_std
    summary = [None] * len(thresh)
    for adate in dates:
        data = result[adate]
        valuation = construct_valuation(data, beta)
        for ii, th in enumerate(thresh):
            if summary[ii] is None:
                summary[ii] = sim_market_making_simple(data, valuation, th, th * edge_buffer_frac, th * edge_rem_frac,
                                                       latency, ticksize)
            else :
                summary[ii] = np.append(summary[ii], sim_market_making_simple(data, valuation, th, th * edge_buffer_frac, th * edge_rem_frac,
                                                                              latency, ticksize))
    for ii in range(len(summary)):
        summary[ii]['total_pnl'] -= (summary[ii]['volume'] * cost)
    avg_pnl = [np.mean(item['total_pnl'] ) for item in summary]
    sharpe  = [np.mean(item['total_pnl'] ) / np.std(item['total_pnl'] ) for item in summary]
    ppv     = [np.sum(item['total_pnl'] ) / np.sum(item['volume']) for item in summary]
    volume  = [np.mean(item['volume']) for item in summary]
    sharpe_constrained = [sp if (vol > 200) and (vol < 2050) else -1e10 for (sp, vol) in zip(sharpe, volume)]
    #print avg_pnl
    #print sharpe
    #print ppv
    idx_max = np.argmax(sharpe_constrained)
    print 'Optimization Dates  :  %s - %s'%(min(dates), max(dates))
    print 'Optimization Results:  idx = %d; thresh = %s'%(idx_max, str(thresh[idx_max]))
    print_summary(summary[idx_max], 'InSample Optimized')
    if sharpe[idx_max] < 0: ## negative stuff, no trading
        return 1e10
    else :
        return thresh[idx_max]

def optimize_in_sample(result, dates):
    reg_dates = dates#[:len(dates) / 2]
    sim_dates = dates#[(len(dates) / 2):]
    reg_param = run_regression(result, reg_dates)
    trade_param = optimize_thresh(result, sim_dates, reg_param)
    param = np.array([tuple([trade_param] + [reg_param[1][nm] for nm in reg_param[0]])],
                     dtype = [('thresh', 'f')] + [(nm, 'f') for nm in reg_param[0]])
    return param

def sim_out_sample(result, dates, param):
    thresh = param['thresh']
    beta = param[0] #list(param[0])[1:]
    pnl_array = None
    for adate in dates:
        data = result[adate]
        valuation = construct_valuation(data, beta)
        if pnl_array is None:
            pnl_array = sim_market_making_simple(data, valuation, thresh, thresh * edge_buffer_frac, thresh * edge_rem_frac,
                                                 latency, ticksize)
        else :
            pnl_array = np.append(pnl_array, sim_market_making_simple(data, valuation, thresh, thresh * edge_buffer_frac, thresh * edge_rem_frac,
                                                                      latency, ticksize))
    pnl_array['total_pnl'] -= (pnl_array['volume'] * cost)
    print 'OutSample Dates : %s - %s'%(min(dates), max(dates))
    print_summary(pnl_array, 'Outsampl Result')
    return pnl_array


rsim = RollSim('trigger', 10, 5)
rsim.TrainFunc = optimize_in_sample
rsim.SimFunc = sim_out_sample

#rsim.YearMonthList = ['201101']

rsim.load_all_data()
rsim.split_dates()
rsim.run()
rsim.print_pnl_summary()


