from mfipy.roll_sim import RollSim, print_summary
from mfipy.data_loader import readCSV
import numpy as np
import os.path as op
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from collections import defaultdict
from matplotlib import mlab
import datetime
from sklearn.decomposition import PCA
from sklearn.grid_search import IterGrid


TRAINED_MODEL = {}

back_range = range(10, 16, 1)
front_range = range(10, 16, 1)
factor_range = [4]
param_space = {'back': back_range, 'front': front_range}
#param_list = list(IterGrid(param_space))
param_list = [{'back':item[0], 'front':item[1]} for item in zip(back_range, front_range)]

def open_to_open_ret(rec, name=''):
    ret = np.diff(rec['Open']) / rec['Open'][:-1]
    dates = rec['Date'][1:]
    new_rec = np.array(zip(dates, ret), dtype=[('Date', datetime.date),
                                               ('Ret_'+name, float)])
    return new_rec

def recs_inner_join(key, recs, postfixes):
    new_rec = recs[0]
    for ii in range(1, len(recs)):
        r1postfix='1'
        r2postfix='2'
        new_rec = mlab.rec_join(key, new_rec, recs[ii], jointype='inner', 
                                defaults=None, r1postfix=r1postfix, r2postfix=r2postfix)
    return new_rec
    
def to_matrix(rec):
    new_rec = mlab.rec_drop_fields(rec, ['Date'])
    new_mat = np.c_[[new_rec[nm] for nm in new_rec.dtype.names]].T
    return new_mat


class Mft2RollSim(RollSim):
    def __init__(self, name, train_n, sim_n):
        super(Mft2RollSim, self).__init__(name, train_n, sim_n)
        
    def load_all_data(self, filenames, ret_func, min_date='20040101'):
        result = {}
        dates = []
        for fname in filenames:
            assert 'daily_' in fname
            ticker = op.split(fname)[-1].split('_')[1]
            rec = readCSV(fname, date_name='Date', date_format='%Y%m%d')
            rec = np.sort(rec, 0, order=['Date'])
            result[ticker] = ret_func(rec, ticker)
            
        tickers, recs = zip(*result.items())
        self._tickers = tickers
        self._result = recs_inner_join('Date', recs, 
                                       postfixes=tickers)
        self._result = np.sort(self._result, 0, order=['Date'])
        self._dates = [dt.strftime('%Y%m%d') for dt in self._result['Date'] if dt.strftime('%Y%m%d') >= min_date ]
        print "Tickers: "
        print "        ", ", ".join(result.keys())

def subset_result_for_dates(result, dates):
    new_result = None
    for adate in dates:
        if new_result is None:
            new_result = result[np.where(result['Date'] == adate)]
        else :
            new_result = np.append(new_result, result[np.where(result['Date'] == adate)])
    return new_result

def auto_correlation_score(mat, loading, back, front):
    factor = np.dot(mat, loading)
    nn = len(factor)
    back_weights = np.ones(back) * 1.0 / back
    front_weights = np.ones(front) * 1.0 / front
    factor_back = np.convolve(factor, back_weights)[(back-1):(nn-front)]
    factor_front = np.convolve(factor, front_weights)[(back+front-1):nn]
    value = np.mean(factor_back * factor_front) / np.std(factor_back) / np.std(factor_front)
    return value

def pos_trading_score(mat, loading, back, front, mean_revert=True):
    factor = np.dot(mat, loading)
    nn = len(factor)
    back_weights = np.ones(back) * 1.0 / back
    front_weights = np.ones(front) * 1.0 / front
    factor_back = np.convolve(factor, back_weights)[(back-1):(nn-front)]
    
    pos_loading = np.copy(loading)
    pos_loading[np.where(loading < 0)] = 0.0
    pos_factor_front = np.convolve(np.dot(mat, pos_loading), front_weights)[(back+front-1):nn]
    neg_loading = np.copy(loading) * (-1.0)
    neg_loading[np.where(loading > 0)] = 0.0
    neg_factor_front = np.convolve(np.dot(mat, neg_loading), front_weights)[(back+front-1):nn]
    
    if mean_revert:
        trade_factor = np.copy(pos_factor_front)
        trade_factor[np.where(factor_back) > 0] = neg_factor_front[np.where(factor_back) > 0]
        value = np.mean(np.fabs(factor_back) * trade_factor) / np.std(factor_back) / np.std(trade_factor)
    else :
        trade_factor = np.copy(pos_factor_front)
        trade_factor[np.where(factor_back) < 0] = neg_factor_front[np.where(factor_back) < 0]
        value = np.mean(np.fabs(factor_back) * trade_factor) / np.std(factor_back) / np.std(trade_factor)
    return value


def generating_pos_position(mat, loading, dates, back, front, mean_revert=True):
    factor = np.dot(mat, loading)
    nn = len(factor)
    back_weights = np.ones(back) * 1.0 / back
    factor_back = np.convolve(factor, back_weights)[(back-1):nn]
    date_used = dates[(back-1):nn]

    pos_loading = np.copy(loading)
    pos_loading[np.where(loading < 0)] = 0.0
    neg_loading = np.copy(loading)
    neg_loading[np.where(loading > 0)] = 0.0
    pos_loading *= 10.0
    neg_loading *= -10.0
    
    position = []
    for ii in range(front-1, len(date_used)):
        cur_pos = np.zeros(len(pos_loading))
        for jj in range(front):
            signal = factor_back[ii-jj]
            if mean_revert:
                if signal > 0:
                    cur_pos = cur_pos + signal * neg_loading
                else :
                    cur_pos = cur_pos + signal * (-1.0) * pos_loading
            else :
                if signal > 0:
                    cur_pos = cur_pos + signal * pos_loading
                else :
                    cur_pos = cur_pos + signal * (-1.0) * neg_loading
        position.append(cur_pos)
    date_trade = date_used[(front-1):]
    return date_trade[:-1], position[:-1]


def train_func(result, dates):
    dates_dt = [datetime.datetime.strptime(dt, '%Y%m%d').date() for dt in dates]
    new_result = subset_result_for_dates(result, dates_dt)
    new_result_mat = to_matrix(new_result)
    pca = PCA()
    pca.fit(new_result_mat)

    est = []
    for ind in factor_range:
        for param in param_list:
            for mv in [True]:
                value = pos_trading_score(new_result_mat, pca.components_[ind], param['back'], param['front'], mv)
                est.append((ind, param['back'], param['front'], mv, value))
    est = np.array(est, dtype=[('factor_ind', int), ('back', int), ('front', int), ('mean_revert', bool), ('value', float)])
    model_name = max(dates)
    ind_max = np.argmax(est['value'])
    TRAINED_MODEL[model_name] = {'factor_ind': est['factor_ind'][ind_max],
                                 'back': est['back'][ind_max],
                                 'front': est['front'][ind_max],
                                 'mean_revert': est['mean_revert'][ind_max],
                                 'est': est,
                                 'loadings': pca.components_,
                                 'dates': dates}
    param = np.array([(model_name, est['factor_ind'][ind_max], est['back'][ind_max], est['front'][ind_max], est['mean_revert'][ind_max],
                       est['value'][ind_max])],
                     dtype = [('model_name', 'S8'), ('factor_ind', int), ('back', int), ('front', int), ('mean_revert', bool),
                              ('value', float)])
    print "optimized param"
    print param
    return param


def sim_func(result, dates, param):
    back = param['back'][0]
    front = param['front'][0]
    model_name = param['model_name'][0]
    pre_dates = TRAINED_MODEL[model_name]['dates']
    all_dates = pre_dates[(-back-front):] + dates
    all_dates_dt = [datetime.datetime.strptime(dt, '%Y%m%d').date() for dt in all_dates]
    new_result = subset_result_for_dates(result, all_dates_dt)
    new_result_mat = to_matrix(new_result)
    trade_dates, position = generating_pos_position(new_result_mat, TRAINED_MODEL[model_name]['loadings'][param['factor_ind'][0]], 
                                                    all_dates, back, front, mean_revert=param['mean_revert'][0])
    future_return = new_result_mat[(back+front-1):,:]
    pnl_mat = np.array(position) * future_return
    pnl = np.sum(pnl_mat, axis=1)
    pos = np.sum(position, axis=1)
    pnl_array = np.array(zip(dates, pnl, pos, pos, pos), dtype=[('date', 'S8'), ('total_pnl', float),
                                                                ('volume', float), ('max_position', float),
                                                                ('min_position', float)])
    print print_summary(pnl_array, 'OutSample')
    return pnl_array


rsim = Mft2RollSim('test_pca', 100, 100)
rsim.TrainFunc = train_func
rsim.SimFunc = sim_func

#fileroot = '/home/mingyuan/Projects/DZH_data/Daily'
#filelist = [op.join(fileroot, 'daily_000029_20120623215017.csv')]
filelist = [
    '/home/mingyuan/Projects/DZH_data/Daily/daily_600016_20120928144733.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_002230_20120928145038.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_601857_20120928145224.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000002_20120928145525.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000799_20120928145811.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000538_20120928150033.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000060_20120928150309.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000099_20120928150501.csv',
    ]

rsim.load_all_data(filelist, open_to_open_ret, min_date='20040101')
#param = train_func(rsim._result, rsim._dates[:300])
#sim_func(rsim._result, rsim._dates[300:400], param)
rsim.split_dates()
rsim.run()
rsim.print_pnl_summary()
