import hft_strat_lib
reload(hft_strat_lib)
from hft_strat_lib import *
from hft_signal_lib import *
from hft_sim_lib import *
from optparse import OptionParser
from collections import defaultdict
### some code to analysis index futures


parser = OptionParser()

parser.add_option('--yearmonth', dest='yearmonth', default='201101')
parser.add_option('--begin-time', dest='begin_time', default='09:00')
parser.add_option('--end-time', dest='end_time', default='16:00')


parser.add_option('--freq-min', dest='freq_min', default=5)
parser.add_option('--pos-lim', dest='pos_lim', default=10)





(options, args) = parser.parse_args()

yearmonth = options.yearmonth
freq_min = int(options.freq_min)
pos_lim = int(options.pos_lim)

product = IFHV

npoint_per_min = 120
ticksize = 0.2

begin_time = options.begin_time
end_time = options.end_time


result = load_data(product, yearmonth)
dates = result.keys()
dates.sort()



def print_option(options):
    print "Options are:"
    print str(options)

def time_filter(data, begin_time, end_time):
    time = np.array([dt.strftime('%H:%M') for dt in data['time']])
    idx = np.where((time >= begin_time) & (time <= end_time))
    return data[idx]


def sim_mv_all_month(dates, result, freq_min=5, pos_lim=10):

    data = result[dates[0]]
    data = time_filter(data, begin_time, end_time)
    pnl_simple = sim_mv_simple(data, sample_freq = npoint_per_min * freq_min, 
                               ticksize = ticksize)
    pnl_poslim = sim_mv_poslim(data, pos_lim=pos_lim, sample_freq = npoint_per_min * freq_min, 
                               ticksize = ticksize)
    pnl_reset  = sim_mv_poslim_reset(data, pos_lim=pos_lim, sample_freq = npoint_per_min * freq_min, 
                                     ticksize = ticksize)

    for adate in dates[1:]:
        data = result[adate]
        data = time_filter(data, begin_time, end_time)
        pnl_simple_tmp = sim_mv_simple(data, sample_freq = npoint_per_min * freq_min, 
                                   ticksize = ticksize)
        pnl_poslim_tmp = sim_mv_poslim(data, pos_lim=pos_lim, sample_freq = npoint_per_min * freq_min, 
                                   ticksize = ticksize)
        pnl_reset_tmp  = sim_mv_poslim_reset(data, pos_lim=pos_lim, 
                                             sample_freq = npoint_per_min * freq_min, 
                                             ticksize = ticksize)
        pnl_simple = np.append(pnl_simple, pnl_simple_tmp)
        pnl_poslim = np.append(pnl_poslim, pnl_poslim_tmp)
        pnl_reset = np.append(pnl_reset, pnl_reset_tmp)
    
    return pnl_simple, pnl_poslim, pnl_reset

def sim_trigger_all_month(dates, result, beta, thresh):

    data = result[dates[0]]
    data = time_filter(data, begin_time, end_time)
    valuation = booksignal(data, 1, 0.9) + (tradesignema(data, 0.7) * beta)
    pnl_signal = sim_trigger(data, valuation, thresh)

    for adate in dates[1:]:
        data = result[adate]
        data = time_filter(data, begin_time, end_time)
        valuation = booksignal(data, 1, 0.9) + (tradesignema(data, 0.7) * beta)
        pnl_signal_tmp = sim_trigger(data, valuation, thresh)
        pnl_signal = np.append(pnl_signal, pnl_signal_tmp)
    
    return pnl_signal
    

print(options)

print "simulating..."

"""
pnl_arrays = sim_mv_all_month(dates, result, freq_min, pos_lim)
pnl_zip = zip(['mv_simple', 'mv_poslim', 'mv_reset'], pnl_arrays)
"""
thresh = [3.0, 3.05, 3.1, 3.15]
pnl_arrays = [sim_trigger_all_month(dates, result, 3.0, th) for th in thresh]
pnl_zip = zip(thresh, pnl_arrays)


print "done!"

for name, pnl_array in pnl_zip:
    print_summary(pnl_array, name)







