from mfipy.data_loader import readCSV
import numpy as np
import datetime

def get_sig(data):
    volume = data['Volume']
    signed = np.sign(data['Close'] - data['Open'])
    return np.sum(volume * signed) / np.sum(volume)

def get_fut_c2c(data, fut_data):
    idx0 = np.argmax(data['Datetime'])
    price0 = data['Close'][idx0]
    idx1 = np.argmax(fut_data['Datetime'])
    price1 = fut_data['Close'][idx1]
    return (price1 - price0) / price0


def get_fut_c2o(data, fut_data):
    idx0 = np.argmax(data['Datetime'])
    price0 = data['Close'][idx0]
    idx1 = np.argmax(datetime.datetime.max - fut_data['Datetime'])
    price1 = fut_data['Open'][idx1]
    return (price1 - price0) / price0

def get_fut_o2c(data, fut_data):
    idx0 = np.argmax(datetime.datetime.max - fut_data['Datetime'])
    price0 = fut_data['Open'][idx0]
    idx1 = np.argmax(fut_data['Datetime'])
    price1 = fut_data['Close'][idx1]
    return (price1 - price0) / price0

def get_fut_o2o(data, fut_data):
    idx0 = np.argmax(datetime.datetime.max - fut_data['Datetime'])
    price0 = fut_data['Open'][idx0]
    dates = [dt.date() for dt in fut_data['Datetime']]
    max_date = max(dates)
    fut_data_max_date = fut_data[np.where(np.array(dates) == max_date)]
    idx1 = np.argmax(datetime.datetime.max - fut_data_max_date['Datetime'])
    price1 = fut_data_max_date['Open'][idx1]
    return (price1 - price0) / price0


def get_data(filename, sig_funcs, fut_func):
    data = readCSV(filename, datetime_name = 'Datetime', 
                   datetime_format = '%Y%m%d%H%M')
    data = np.sort(data, 0, order=['Datetime'])
    dates = np.array([dt.date() for dt in data['Datetime']])
    unique_dates = np.unique(dates)
    print len(unique_dates)
    print unique_dates
    result = []
    for ii in range(1, len(unique_dates)-1):
        tmp = data[np.where(dates == unique_dates[ii-1])]
        sig = [sf(tmp) for sf in sig_funcs]
        fut_tmp = data[np.where((dates == unique_dates[ii]) | (dates == unique_dates[ii+1]))]
        fut = fut_func(tmp, fut_tmp)
        row = sig + [fut]
        result.append(row)
    return np.array(result)

def get_sig_data(filename, sig_funcs):
    data = readCSV(filename, datetime_name = 'Datetime', 
                   datetime_format = '%Y%m%d%H%M')
    data = np.sort(data, 0, order=['Datetime'])
    dates = np.array([dt.date() for dt in data['Datetime']])
    unique_dates = np.unique(dates)
    print len(unique_dates)
    print unique_dates
    result = []
    for ii in range(0, len(unique_dates)):
        tmp = data[np.where(dates == unique_dates[ii])]
        sig = [sf(tmp) for sf in sig_funcs]
        row = sig
        result.append(row)
    return np.array(result)


if __name__ == '__main__':
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000692_20121015170412.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_002371_20121015193925.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_300075_20121015195144.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_002366_20121015195300.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000903_20121015195601.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000522_20121015195809.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000799_20121015195941.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600537_20121015201309.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600356_20121015201453.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_601989_20121015201552.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600760_20121015201739.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600011_20121015202214.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_300087_20121015202321.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600894_20121015203006.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_300086_20121016115020.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000596_20121016115141.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_002292_20121016115317.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000537_20121016115500.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_002610_20121016115905.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_002274_20121016120019.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_000663_20121016120149.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600844_20121016121925.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600199_20121016122031.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600249_20121016122149.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600559_20121016122359.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600420_20121016123142.csv'
    #filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_603000_20121016123552.csv'
    filename = '/home/mingyuan/Projects/DZH_data/Daily/m5_600011_20121016130251.csv'
    data = get_data(filename, [get_sig], get_fut_o2o)
    print np.corrcoef(data[:,0], data[:,1])
    print data[:,1][np.where(data[:,0] > 0.2)].mean()
    print data[:,1][np.where(data[:,0] < -0.2)].mean()

    sig_data = get_sig_data(filename, [get_sig])
    print sig_data

