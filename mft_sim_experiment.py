from mfipy.roll_sim import RollSim, print_summary
from mfipy.data_loader import readCSV
import numpy as np
import os.path as op
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from collections import defaultdict

HISTORY = 1
FUTURE = 1

TRAINED_MODEL = {}

Trade_Thresh = 0.
Cost = 0.01

Kernel = 'linear'

if Kernel == 'rbg':
    C_range = 10.0 ** np.arange(2, 4, 1)
    gamma_range = 10.0 ** np.arange(-3, 1, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)
elif Kernel == 'linear' :
    C_range = 10.0 ** np.arange(2, 4, 1)
    param_grid = dict(C=C_range)

CLASSIFY = False

def neg_mse(y_act, y_pred):
    return (-1.0) * mean_squared_error(y_act, y_pred)

class MftRollSim(RollSim):
    def __init__(self, name, train_n, sim_n):
        super(MftRollSim, self).__init__(name, train_n, sim_n)
        
    def load_all_data(self, filenames, min_date='20040101'):
        result = {}
        dates = []
        for fname in filenames:
            assert 'daily_' in fname
            ticker = op.split(fname)[-1].split('_')[1]
            rec = readCSV(fname, date_name='Date', date_format='%Y%m%d')
            rec = np.sort(rec, 0, order=['Date'])
            result[ticker] = rec
            dates = np.append(dates, rec['Date'])

        self._result = result
        self._dates = [dt.strftime('%Y%m%d') for dt in np.unique(dates) if dt.strftime('%Y%m%d') >= min_date ]
        self._dates.sort()
        print "Tickers: "
        print "        ", ", ".join(result.keys())

def subset_results_for_dates(result, dates):
    new_result = {}
    for ticker in result.keys():
        ind = np.where([item.strftime('%Y%m%d') in dates for item in result[ticker]['Date']])
        new_result[ticker] = result[ticker][ind]
    return new_result


def get_signal(results, with_y=True, vol_normal=None, x_norm=None, normalize_y=False):
    # time zero is open at day ii
    signals = {}
    prepared_x    = []
    tickers       = []
    dates         = []
    prepared_y    = []
    new_future = FUTURE if with_y else 0
    if vol_normal is None:
        vol_normalization = {}
    else : 
        vol_normalization = vol_normal
    for ticker in results.keys():
        data = results[ticker]
        if vol_normal is None:
            vol_std = data['Volume'].std() * 10.0  # scaled to have std of 0.1
            vol_mean = data['Volume'].mean()
            vol_normalization[ticker] = {'vol_mean': vol_mean, 'vol_std': vol_std}
        else :
            vol_std = vol_normalization[ticker]['vol_std']
            vol_mean = vol_normalization[ticker]['vol_mean']
        for ii in range(HISTORY, len(data) - new_future, 1):
            #open_to_open = [(data['Open'][ii-jj] - data['Open'][ii-jj-1])/data['Open'][ii-jj-1] for jj in range(HISTORY)]
            open_to_close = [(data['Close'][ii-jj-1] - data['Open'][ii-jj-1])/data['Open'][ii-jj-1] for jj in range(HISTORY)]
            #open_to_high =  [(data['High'][ii-jj-1] - data['Open'][ii-jj-1])/data['Open'][ii-jj-1] for jj in range(HISTORY)]
            #open_to_low = [(data['Low'][ii-jj-1] - data['Open'][ii-jj-1])/data['Open'][ii-jj-1] for jj in range(HISTORY)]
            close_to_open = [(data['Open'][ii-jj] - data['Close'][ii-jj-1])/data['Close'][ii-jj-1] for jj in range(HISTORY)]
            std_volume = [(data['Volume'][ii-jj-1] - vol_mean) / vol_std for jj in range(HISTORY)]
            if with_y:
                future_open_to_open = [(data['Open'][ii+jj] - data['Open'][ii]) / data['Open'][ii] for jj in [FUTURE]]
                prepared_y += future_open_to_open
            x_tmp = open_to_close + close_to_open + std_volume #open_to_open + open_to_close + open_to_high + open_to_low + close_to_open
            prepared_x.append(x_tmp)
            tickers.append(ticker)
            dates.append(data['Date'][ii+FUTURE])
    names =  ['otc-' + str(jj) for jj in range(HISTORY)] + \
        ['cto-' + str(jj) for jj in range(HISTORY)] + \
        ['svol-' + str(jj) for jj in range(HISTORY)]
    prepared_x = np.array(prepared_x)
    prepared_y = np.array(prepared_y)
    if x_norm is None:
        x_norm = [(prepared_x[:,ii].mean(), prepared_x[:,ii].std()) for ii in range(prepared_x.shape[1])]
    prepared_x = np.array([(prepared_x[:,ii] - x_norm[ii][0]) / x_norm[ii][1] for ii in range(len(x_norm))]).T
    if normalize_y and with_y:
        prepared_y = (prepared_y - prepared_y.mean()) / prepared_y.std()
    return names, prepared_x, prepared_y, vol_normalization, tickers, dates, x_norm
    
def classify(yy):
    if yy > Trade_Thresh:
        return 1
    elif yy < -Trade_Thresh:
        return -1
    else :
        return 0

def train_func(results, dates):
    new_results = subset_results_for_dates(results, dates)
    #import pdb; pdb.set_trace()
    names, X, Y, vol_normalization, tickers, eff_dates, x_norm = get_signal(new_results, with_y=True, normalize_y=True)
    print "Training set characteristics", X.shape
    
    if not CLASSIFY:
        grid = GridSearchCV(svm.SVR(kernel=Kernel, verbose=False), param_grid=param_grid, cv=KFold(n=len(Y), k=3, shuffle=True),
                            score_func = neg_mse)#cv=StratifiedKFold(y=Y_class, k=3))
        grid.fit(X, Y)
        C = grid.best_estimator_.C
        if Kernel == 'rbg': 
            gamma = grid.best_estimator_.gamma
            print 'C=%f, gamma=%f'%(C, gamma)
            estimator = svm.SVR(kernel=Kernel, C=C, gamma=gamma, verbose=False)
        elif Kernel == 'linear':
            print 'C=%f'%C
            estimator = svm.SVR(kernel=Kernel, C=C, verbose=False)
        
        estimator.fit(X, Y)
        trained_name = max(eff_dates).strftime('%Y%m%d')
        y_pred = estimator.predict(X)
        #import pdb; pdb.set_trace()
        print 'R2_score = %f'%(r2_score(Y, y_pred))
    else :
        Y_class = np.array([classify(yy) for yy in Y])
        grid = GridSearchCV(svm.SVC(kernel=Kernel, verbose=False, class_weight='auto'), param_grid=param_grid, 
                            cv=StratifiedKFold(y=Y_class, k=3))
        grid.fit(X, Y_class)
        C = grid.best_estimator_.C
        if Kernel == 'rbg': 
            gamma = grid.best_estimator_.gamma
            print 'C=%f, gamma=%f'%(C, gamma)
            estimator = svm.SVC(kernel=Kernel, C=C, gamma=gamma, verbose=False, class_weight='auto')
        elif Kernel == 'linear':
            print 'C=%f'%C
            estimator = svm.SVC(kernel=Kernel, C=C, verbose=False, class_weight='auto')
        estimator.fit(X, Y_class)
        trained_name = max(eff_dates).strftime('%Y%m%d')
        y_pred = estimator.predict(X)
        print classification_report(Y_class, y_pred)

    TRAINED_MODEL[trained_name] = {'names': names,
                                   'y_pred_mean': y_pred.mean(),
                                   'y_pred_std': y_pred.std(),
                                   'x_norm': x_norm,
                                   'model': estimator,
                                   'dates': dates, #[dt.strftime('%Y%m%d') for dt in dates],
                                   'vol_normalization': vol_normalization}
    if Kernel == 'rbg':
        return np.array([(trained_name, C, gamma, y_pred.mean(), y_pred.std())], 
                        dtype=[('model', 'S8'), ('C', 'f'), ('gamma', 'f'), ('y_pred_mean', 'f'), ('y_pred_std', 'f')])

    elif Kernel == 'linear':
        return np.array([(trained_name, C, y_pred.mean(), y_pred.std())], 
                        dtype=[('model', 'S8'), ('C', 'f'), ('y_pred_mean', 'f'), ('y_pred_std', 'f')])


def reorganize_pnl(pnl, date_list, dates):
    pnl_dict = defaultdict(list)
    for ii, adate in enumerate(date_list):
        if adate in dates:
            pnl_dict[adate].append(pnl[ii])
    pnl_array = None
    for adate in dates:
        if adate in pnl_dict.keys():
            total_pnl = np.sum(pnl_dict[adate])
            volume = len(pnl_dict[adate])
        else :
            total_pnl = 0.0
            volume = 0
        pnl_tmp = np.array([(total_pnl, volume, volume, volume)], dtype = 
                           [('total_pnl', 'f'), ('volume', 'd'), ('max_position', 'd'),
                            ('min_position', 'd')])
        if pnl_array is None:
            pnl_array = pnl_tmp
        else :
            pnl_array = np.append(pnl_array, pnl_tmp)
    return pnl_array

def sim_func(results, dates, param):
    model_name = param['model'][0]
    x_norm = TRAINED_MODEL[model_name]['x_norm']
    all_dates = TRAINED_MODEL[model_name]['dates'] + dates
    all_results = subset_results_for_dates(results, all_dates)
    names, X, Y, vol_normal, full_tickers, full_dates, x_norm = get_signal(all_results, 
                                                                           with_y=True, 
                                                                           vol_normal=TRAINED_MODEL[model_name]['vol_normalization'],
                                                                           x_norm=x_norm,
                                                                           normalize_y=False)
    full_dates = [dt.strftime('%Y%m%d') for dt in full_dates]
    model = TRAINED_MODEL[model_name]['model']
    if not CLASSIFY:
        y_pred_mean = param['y_pred_mean']
        y_pred_std = param['y_pred_std']
        #import pdb; pdb.set_trace()

        prediction = model.predict(X)
        print prediction[-10:]
        trade_idx = np.where(prediction > (Trade_Thresh * y_pred_std + y_pred_mean) )
    else :
        prediction = model.predict(X)
        print prediction[-10:]
        trade_idx = np.where(prediction > 0)
    pnl = np.array(Y)[trade_idx] - Cost
    full_dates = np.array(full_dates)[trade_idx]
    new_pnl =  reorganize_pnl(pnl, full_dates, dates)
    print_summary(new_pnl, 'OutSample')
    return new_pnl




rsim = MftRollSim('test_svr', 300, 10)
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

rsim.load_all_data(filelist, min_date='20040101')
rsim.split_dates()
rsim.run()
rsim.print_pnl_summary()

np.savez(file='mft.npz', param_array = rsim.param_array, pnl_array = rsim.pnl_array)

