import numpy as np
from weekly_phase import run_simulation_single
from mfipy.dzh_ticker_info import load_ticker_info


ticker_info = load_ticker_info()
## require positive pe or nan pe with non nan volume
good_idx = (np.isnan(ticker_info['pe_ratio']) & (~np.isnan(ticker_info['volume'])))
good_idx|= (ticker_info['pe_ratio'] > 0)
good_ticker_info = ticker_info[good_idx]

good_tickers = good_ticker_info['ticker']

summary = None

for ticker in good_tickers:
    try: 
        tmp = run_simulation_single(ticker)
        if summary is None:
            summary = tmp
        else :
            summary = np.append(summary, tmp)
        print
        print
        print
        print " "* 20, "TICKER %s IS DONE!!!"%ticker
        print 
        print
        print
    except:
        print
        print
        print
        print " "* 20, "TICKER %s FAILED!!!"%ticker
        print 
        print
        print


