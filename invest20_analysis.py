from mfipy.yahoo_data_loader import getYahooData
from mfipy.mean_variance import mean_estimation, cov_estimation, MeanVariance
import datetime
import numpy as np


def get_return(dat):
    dates = dat['Date'][1:]
    ret = np.diff(dat['Close']) / dat['Close'][:-1]
    new_dat = np.array(zip(dates, ret), dtype = [('Date', datetime.date),
                                                 ('Return', float)])
    return new_dat


stockList = ['FDX', 'UPS', 'OHI', 'HCN', 'XOM', 'CVX']

date1 = datetime.date(1990, 1, 1)
date2 = datetime.date(2012, 10, 10)

dataList = [getYahooData(ticker, date1, date2) for ticker in stockList]
cdataList = [get_return(dat) for dat in dataList]

average_return = mean_estimation([dat['Return'] for dat in cdataList], use_stein=False)
covariance = cov_estimation(cdataList, 'Date', pair_wise=True)

average_return_mod = average_return
#average_return_mod[0] *= 1.3
#average_return_mod[1] *= 1.3
#average_return_mod[2] *= 1.0

mv = MeanVariance()
mv.nameList = stockList
mv.meanList = average_return_mod
mv.covMat = covariance
mv.fixedPosition = {0: 0.378, 1:0.622}
mv.targetRisk = 0.02
#mv.run('zeroone', True)
mv.run('hedge|positive')
print mv.printResult()




