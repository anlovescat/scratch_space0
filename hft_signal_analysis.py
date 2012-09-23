import hft_signal_lib
reload(hft_signal_lib)
from hft_signal_lib import *
import numpy as np
import datetime
import glob
import os
from collections import defaultdict
### some code to analysis index futures

if len(sys.argv) <= 1:
    yearmonth = '201101'
else :
    yearmonth = sys.argv[1]

result = load_data(IFHV, yearmonth)
dates = result.keys()
dates.sort()

decays = [0.01, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
decay_names = [str(ff).replace('0.', '_') for ff in decays]

burn_in = 60
offsets = range(2, 60, 4)

#data = result[dates[0]]

### signals
def get_signals(data):
    mid = midpoint(data)
    book = booksignal(data, 1, 0.9)
    last = data['lastprice']
    emabook = [ema(book, ff) for ff in decays]
    vwap = [vwap_ema(data, ff) for ff in decays]
    trends = [book / ema(book, ff) for ff in decays]
    trsigns = [tradesignema(data, ff) for ff in decays]
    trsvolumes = [tradesignvolume_ema(data, ff) for ff in decays]
    sv = [signedvolume_ema(data, ff) for ff in decays]
    spr = spread(data)
    
    
    base_ind = subsample_nsample(data, 60)
    base_ind, offsets_ind = generate_offset_index(len(data), offsets, base_ind, burn_in)
    
    signals = {'mid': mid, 'book': book, 'spr': spr, 'last': last}
    signals.update(dict(zip(['bkema'+dn for dn in decay_names], emabook)))
    signals.update(dict(zip(['vwap'+dn for dn in decay_names], vwap)))
    signals.update(dict(zip(['trends'+dn for dn in decay_names], trends)))
    signals.update(dict(zip(['trsign'+dn for dn in decay_names], trsigns)))
    signals.update(dict(zip(['trsvolume'+dn for dn in decay_names], trsvolumes)))
    signals.update(dict(zip(['signedv'+dn for dn in decay_names], sv)))
    return signals, base_ind, offsets_ind

def get_correlation(signals, base_ind, offsets_ind):
    futures = [signals['mid'][offind] - signals['mid'][base_ind] for offind in offsets_ind]
    new_signals = {}
    for key in signals.keys():
        if key == 'mid' or key == 'spr':
            continue
        elif key == 'book' or 'bkema' in key or key == 'last' or 'vwap' in key:
            new_signals[key] = (signals[key] - signals['mid'])[base_ind]
        elif 'trends' in key:
            new_signals[key] = (signals[key] - 1.0)[base_ind]
        else :
            new_signals[key] = signals[key][base_ind]
    def get_cor_impl(sig, futures):
        result = [np.mean(sig * fut) / np.std(sig) / np.std(fut) for fut in futures]
        return result
    correlation = {}
    for key in new_signals.keys():
        correlation[key] = get_cor_impl(new_signals[key], futures)
    return correlation

#correlations = get_correlation(signals, base_ind, offsets_ind)

def print_correlations(correlations):
    correlation_avg = [(key, correlations[key].mean(axis=0) * 100) for key in correlations.keys()]
    correlation_max = [(key, correlations[key].mean(axis=0).mean() * 100) for key in correlations.keys()]
    correlation_max = sorted(correlation_max, key = lambda x : x[1])
    for signal, value in correlation_max:
        print signal, '\t', value
    print
    
    trend = [item for item in correlation_max if 'trend' in item[0]]
    book = [item for item in correlation_max if 'book' in item[0]]
    last = [item for item in correlation_max if 'last' in item[0]]
    emabook = [item for item in correlation_max if 'bkema' in item[0]]
    vwap = [item for item in correlation_max if 'vwap' in item[0]]
    trsign = [item for item in correlation_max if 'trsign' in item[0]]
    trsvolume = [item for item in correlation_max if 'trsvolume' in item[0]]
    signedv = [item for item in correlation_max if 'signedv' in item[0]]

    print 'Trend signals'
    print trend[0]
    print dict(correlation_avg)[trend[0][0]]
    print trend[-1]
    print dict(correlation_avg)[trend[-1][0]]
    print 

    print 'book signals'
    print book[0]
    print dict(correlation_avg)[book[0][0]]
    #print trend[-1]
    #print dict(correlation_avg)[trend[-1][0]]
    print 

    print 'last signals'
    print last[0]
    print dict(correlation_avg)[last[0][0]]
    #print trend[-1]
    #print dict(correlation_avg)[trend[-1][0]]
    print 


    print 'emabook signals'
    print emabook[0]
    print dict(correlation_avg)[emabook[0][0]]
    print emabook[-1]
    print dict(correlation_avg)[emabook[-1][0]]
    print 

    print 'vwap signals'
    print vwap[0]
    print dict(correlation_avg)[vwap[0][0]]
    print vwap[-1]
    print dict(correlation_avg)[vwap[-1][0]]
    print 

    print 'Trsign signals'
    print trsign[0]
    print dict(correlation_avg)[trsign[0][0]]
    print trsign[-1]
    print dict(correlation_avg)[trsign[-1][0]]
    print 

    print 'Trsvolume signals'
    print trsvolume[0]
    print dict(correlation_avg)[trsvolume[0][0]]
    print trsvolume[-1]
    print dict(correlation_avg)[trsvolume[-1][0]]
    print 

    print 'Signedv signals'
    print signedv[0]
    print dict(correlation_avg)[signedv[0][0]]
    print signedv[-1]
    print dict(correlation_avg)[signedv[-1][0]]
    print 
    
    

correlations = defaultdict(list)

for adate in dates:
    print "processing %s"%adate
    data = result[adate]
    signals, base_ind, offsets_ind = get_signals(data)
    tmp_cor = get_correlation(signals, base_ind, offsets_ind)
    for key, value in tmp_cor.items():
        correlations[key].append(value)

for key in correlations.keys():
    correlations[key] = np.array(correlations[key])

print 'DONE!'

print_correlations(correlations)


        






    


