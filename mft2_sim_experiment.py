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

mv_back_range = range(11, 12, 1)
mv_front_range = range(11, 12, 1)
mv_factor_range = [4]
mv_param_space = {'back': mv_back_range, 'front': mv_front_range}
#mv_param_list = list(IterGrid(mv_param_space))
mv_param_list = [{'back':item[0], 'front':item[1]} for item in zip(mv_back_range, mv_front_range)]


tf_back_range = range(1, 2, 1)
tf_front_range = range(1, 2, 1)
tf_factor_range = [3]
tf_param_space = {'back': tf_back_range, 'front': tf_front_range}
#tf_param_list = list(IterGrid(tf_param_space))
tf_param_list = [{'back':item[0], 'front':item[1]} for item in zip(tf_back_range, tf_front_range)]


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
    neg_loading[np.where(loading < 0)] = 0.0
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

    mv_est = []
    mv = True
    for ind in mv_factor_range:
        for param in mv_param_list:
            value = pos_trading_score(new_result_mat, pca.components_[ind], param['back'], param['front'], mv)
            mv_est.append((ind, param['back'], param['front'], mv, value))
    mv_est = np.array(mv_est, dtype=[('factor_ind', int), ('back', int), ('front', int), ('mean_revert', bool), ('value', float)])
    tf_est = []
    mv = False
    for ind in tf_factor_range:
        for param in tf_param_list:
            value = pos_trading_score(new_result_mat, pca.components_[ind], param['back'], param['front'], mv)
            tf_est.append((ind, param['back'], param['front'], mv, value)) 
    tf_est = np.array(tf_est, dtype=[('factor_ind', int), ('back', int), ('front', int), ('mean_revert', bool), ('value', float)])   
    model_name = max(dates)
    mv_ind_max = np.argmax(mv_est['value'])
    tf_ind_max = np.argmax(tf_est['value'])
    TRAINED_MODEL[model_name] = {'mv_factor_ind': mv_est['factor_ind'][mv_ind_max],
                                 'mv_back': mv_est['back'][mv_ind_max],
                                 'mv_front': mv_est['front'][mv_ind_max],
                                 'mv_est': mv_est,
                                 'tf_factor_ind': tf_est['factor_ind'][tf_ind_max],
                                 'tf_back': tf_est['back'][tf_ind_max],
                                 'tf_front': tf_est['front'][tf_ind_max],
                                 'tf_est': tf_est,
                                 'loadings': pca.components_,
                                 'dates': dates}
    param = np.array([(model_name, mv_est['factor_ind'][mv_ind_max], mv_est['back'][mv_ind_max], mv_est['front'][mv_ind_max], mv_est['value'][mv_ind_max],
                       tf_est['factor_ind'][tf_ind_max], tf_est['back'][tf_ind_max], tf_est['front'][tf_ind_max], tf_est['value'][tf_ind_max])],
                     dtype = [('model_name', 'S8'), ('mv_factor_ind', int), ('mv_back', int), ('mv_front', int), ('mv_value', float),
                              ('tf_factor_ind', int), ('tf_back', int), ('tf_front', int), ('tf_value', float)])
    print "optimized param"
    print param
    return param


def sim_func(result, dates, param):
    mv_back = param['mv_back'][0]
    mv_front = param['mv_front'][0]
    tf_back = param['tf_back'][0]
    tf_front = param['tf_front'][0]
    model_name = param['model_name'][0]
    pre_dates = TRAINED_MODEL[model_name]['dates']
    mv_all_dates = pre_dates[(-mv_back-mv_front):] + dates
    mv_all_dates_dt = [datetime.datetime.strptime(dt, '%Y%m%d').date() for dt in mv_all_dates]
    mv_new_result = subset_result_for_dates(result, mv_all_dates_dt)
    mv_new_result_mat = to_matrix(mv_new_result)
    mv_trade_dates, mv_position = generating_pos_position(mv_new_result_mat, TRAINED_MODEL[model_name]['loadings'][param['mv_factor_ind'][0]], 
                                                          mv_all_dates, mv_back, mv_front, mean_revert=True)
    tf_all_dates = pre_dates[(-tf_back-tf_front):] + dates
    tf_all_dates_dt = [datetime.datetime.strptime(dt, '%Y%m%d').date() for dt in tf_all_dates]
    tf_new_result = subset_result_for_dates(result, tf_all_dates_dt)
    tf_new_result_mat = to_matrix(tf_new_result)

    tf_trade_dates, tf_position = generating_pos_position(tf_new_result_mat, TRAINED_MODEL[model_name]['loadings'][param['tf_factor_ind'][0]], 
                                                          tf_all_dates, tf_back, tf_front, mean_revert=True)

    mv_position = np.array(mv_position)
    tf_position = np.array(tf_position)
    #if param['mv_value'] < 0:
    #mv_position = np.zeros(mv_position.shape)
    #if param['tf_value'] < 0:
    #tf_position = np.zeros(tf_position.shape)
    position = mv_position + tf_position
    future_return = mv_new_result_mat[(mv_back+mv_front-1):,:]
    pnl_mat = position * future_return
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
sector_filelist = [
    '/home/mingyuan/Projects/DZH_data/Daily/daily_600016_20120928144733.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_002230_20120928145038.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_601857_20120928145224.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000002_20120928145525.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000799_20120928145811.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000538_20120928150033.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000060_20120928150309.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_000099_20120928150501.csv',
    ]

bank_filelist = [
    '/home/mingyuan/Projects/DZH_data/Daily/daily_600016_20121001134109.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_600000_20121001134258.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_600036_20121001134436.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_601169_20121001134635.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_601998_20121001134751.csv',
    '/home/mingyuan/Projects/DZH_data/Daily/daily_601009_20121001134859.csv',
    ]

filelist = sector_filelist 

rsim.load_all_data(filelist, open_to_open_ret, min_date='20040101')
rsim.split_dates()
rsim.run()
rsim.print_pnl_summary()
