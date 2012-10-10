from mfipy.data_loader import readCSV
from mfipy.yahoo_data_loader import getYahooData
from mfipy.roll_sim import RollSim
import os.path as op
import datetime
import numpy as np
from matplotlib.mlab import rec_summarize, rec_groupby, rec2txt

BASE_DIR = '/home/mingyuan/Projects/DZH_data/Daily'


def rearrange(data):
    isocal = np.array([dt.isocalendar() for dt in data['Date']])
    isoweek = np.array([tp[0] * 100 + tp[1] for tp in isocal])
    unique_isoweek = np.unique(isoweek)
    new_data = []
    for isw in unique_isoweek:
        idx = np.where(isoweek == isw)
        row = [np.nan] * 10
        for ii in idx[0]:
            week = isocal[ii][2]
            if week > 5 or week < 0:
                print "incorrect weekday %d! (%s)" % (week, str(data['Date'][ii]))
                continue
            row[2 * (week - 1)] = data['Open'][ii]
            row[2 * (week - 1) + 1] = data['Close'][ii]
        row = [isocal[ii][0], isocal[ii][1]] + row
        new_data.append(tuple(row))
    new_data = np.array(new_data, dtype = [('year', int), ('weekoftheyear', int),
                                           ('mon_open', float), ('mon_close', float),
                                           ('tue_open', float), ('tue_close', float),
                                           ('wed_open', float), ('wed_close', float),
                                           ('thu_open', float), ('thu_close', float),
                                           ('fri_open', float), ('fri_close', float)])
    return unique_isoweek, new_data
    

def load_file(filename):
    assert 'daily_' in filename
    ticker = filename.split('_')[1]
    full_path = op.join(BASE_DIR, filename)
    data = readCSV(full_path, date_name = 'Date', date_format='%Y%m%d')
    data = np.sort(data, 0, order=['Date'])
    data = data[np.where(data['Date'] > datetime.date(2004, 1, 1))]
    iso_week, new_data = rearrange(data)
    return ticker, iso_week, new_data

def load_yahoo_data(ticker, date1=datetime.date(2004, 1, 1), date2=datetime.date.today()):
    data = getYahooData(ticker, date1, date2)
    data = np.sort(data, 0, order=['Date'])
    data = data[np.where(data['Date'] > datetime.date(2004, 1, 1))]
    iso_week, new_data = rearrange(data)
    return ticker, iso_week, new_data
    
def summarize_simple(array):
    rarray = np.diff(array) / array[:-1]
    rarray = rarray[~np.isnan(rarray)]
    return np.mean((rarray * 100 ) ** 2)
    
def estimate_var(new_data, groupbyyear=False):
    summary_list = [(nm, summarize_simple, nm+'_var') for nm in new_data.dtype.names \
                        if nm not in ['year', 'weekoftheyear']]
    if groupbyyear:
        return rec_groupby(new_data, ['year'], summary_list)
    else :
        return rec_groupby(new_data, [], summary_list)

def analyze_yahoo_data(ticker, date1=datetime.date(2004, 1, 1), date2=datetime.date.today()):
    ticker, iso_week, data = load_yahoo_data(ticker, date1, date2)
    result_all = estimate_var(data)
    result_year = estimate_var(data, True)
    print rec2txt(result_all)
    print rec2txt(result_year)
    

class WphRollSim(RollSim):
    def __init__(self, name, train_n, sim_n):
        super(WphRollSim, self).__init__(name, train_n, sim_n)
    
    def load_all_data(self, ticker, date1=datetime.date(2004, 1, 1), date2=datetime.date.today()):
        ticker, iso_week, data = load_yahoo_data(ticker, date1, date2)
        self._ticker = ticker
        self._result = data
        self._dates = iso_week.tolist()

def subset_result_for_dates(result, dates):
    new_result = None
    result_dates = result['year'] * 100 + result['weekoftheyear']
    for adate in dates:
        if new_result is None:
            new_result = result[np.where(result_dates == adate)]
        else :
            new_result = np.append(new_result, result[np.where(result_dates == adate)])
    return new_result

def train_func(result, dates):
    new_result = subset_result_for_dates(result, dates)
    summary = estimate_var(new_result)
    summary_array = np.array(summary[0].tolist())
    idx_max = np.argmax(summary_array)
    idx_min = np.argmax(-1.0 * summary_array)
    param = np.array([(summary.dtype.names[idx_max], idx_max, summary.dtype.names[idx_min], idx_min)],
                     dtype = [('max_var_name', 'S20'), ('max_var_ind', int), 
                              ('min_var_name', 'S20'), ('min_var_ind', int)])
    return param

def generate_pnl(result, param, long_only=False):
    idx_max = param['max_var_ind'] + 2
    idx_min = param['min_var_ind'] + 2
    pnl = []
    volume = []
    last_row = result[0].tolist()
    #import pdb; pdb.set_trace()
    if idx_max > idx_min:
        for ii in range(1, len(result)-1):
            row = result[ii].tolist()
            next_row = result[ii+1].tolist()
            pos1 = (row[idx_min] - last_row[idx_max]) / last_row[idx_max]
            ret1 = (row[idx_max] - row[idx_min]) / row[idx_min]
            pos2 = (row[idx_max] - row[idx_min]) / row[idx_min] * (-1.0)
            ret2 = (next_row[idx_min] - row[idx_max]) / row[idx_max]
            if long_only:
                pos1 = max(0.0, pos1)
                pos2 = max(0.0, pos2)
            if np.isnan(pos1 * ret1 + pos2 * ret2) :
                pnl.append(0.0)
                volume.append(0.0)
            else :
                pnl.append(pos1 * ret1 + pos2 * ret2)
                volume.append(abs(pos1) + abs(pos2))
    else :
        for ii in range(1, len(result)-1):
            row = result[ii].tolist()
            next_row = result[ii+1].tolist()
            pos1 = (row[idx_max] - last_row[idx_min]) / last_row[idx_min] * (-1.0)
            ret1 = (row[idx_min] - row[idx_max]) / row[idx_max]
            pos2 = (row[idx_min] - row[idx_max]) / row[idx_max]
            ret2 = (next_row[idx_max] - row[idx_min]) / row[idx_min]
            if long_only:
                pos1 = max(0.0, pos1)
                pos2 = max(0.0, pos2)
            if np.isnan(pos1 * ret1 + pos2 * ret2) :
                pnl.append(0.0)
                volume.append(0.0)
            else :
                pnl.append(pos1 * ret1 + pos2 * ret2)
                volume.append(abs(pos1) + abs(pos2))
    pnl.append(0.0)
    volume.append(0.0)
    return pnl, volume

def sim_func(result, dates, param):
    result_dates = result['year'] * 100 + result['weekoftheyear']
    min_dates = min(dates)
    expanded_dates = [result_dates[result_dates.tolist().index(min_dates)-1]] + dates
    new_result = subset_result_for_dates(result, expanded_dates)
    pnl, volume = generate_pnl(new_result, param, long_only=True)
    pnl_array = zip(dates, pnl, volume, volume, volume)
    pnl_array = np.array(pnl_array, dtype = [('date', int), ('total_pnl', float),
                                             ('volume', float), ('max_position', float),
                                             ('min_position', float)])
    return pnl_array


def run_simulation_single(ticker):
    rsim = WphRollSim('test_weekly_phase', 25, 25)
    rsim.TrainFunc = train_func
    rsim.SimFunc = sim_func
    
    rsim.load_all_data(ticker)
    rsim.split_dates()
    rsim.run()
    rsim.print_pnl_summary()
    exclude_top2trade = np.argsort(rsim.pnl_array['total_pnl'])[:-2]
    idx = np.sort(exclude_top2trade)
    total_pnl = rsim.pnl_array['total_pnl'][idx]
    volume = rsim.pnl_array['volume'][idx]
    avg_pnl = total_pnl.sum() / volume.sum()
    sharpe = total_pnl.mean() / total_pnl.std() * np.sqrt(50)
    nweek = rsim.pnl_array.shape[0]
    return np.array([(ticker, sharpe, avg_pnl, nweek)], dtype = [('ticker', 'S9'), ('sharpe', float), ('avg_pnl', float), ('nweek', int)])

def analyze_dzh_file(filename):
    ticker, iso_week, data = load_file(filename)
    result_all = estimate_var(data)
    result_year = estimate_var(data, True)
    print rec2txt(result_all)
    print rec2txt(result_year)
    

if __name__ == '__main__': 
    rsim = WphRollSim('test_weekly_phase', 25, 25)
    rsim.TrainFunc = train_func
    rsim.SimFunc = sim_func
    
    rsim.load_all_data('500018.SS')
    rsim.split_dates()
    rsim.run()
    rsim.print_pnl_summary()

    import pylab as pl
    pl.plot(np.cumsum(rsim.pnl_array['total_pnl']))
    pl.show()



#filename = 'daily_000029_20120623215017.csv'
#filename = 'daily_600000_20121001134258.csv'
#filename = 'daily_000538_20120928150033.csv'
#filename = 'daily_000099_20120928150501.csv'
#filename = 'daily_000799_20120928145811.csv'
#filename = 'daily_002230_20120928145038.csv'
#filename = 'daily_600016_20120928144733.csv'
#filename = 'daily_600036_20121001134436.csv'
#filename = 'daily_601009_20121001134859.csv'
#filename = 'daily_601169_20121001134635.csv'
#filename = 'daily_601857_20120928145224.csv'

#filename = 'daily_601998_20121001134751.csv'
#ticker, iso_week, data = load_file(filename)
#result_all = estimate_var(data)
#result_year = estimate_var(data, True)
#print rec2txt(result_all)
#print rec2txt(result_year)





