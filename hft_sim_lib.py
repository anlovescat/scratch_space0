from hft_signal_lib import *
from hft_strat_lib import *

def print_summary(pnl_array, name):
    if len(pnl_array) > 1:
        string = "\n"
        string+= "pnl summary for %s\n"%name
        string+= "Average Pnl  :   %.2f\n"%pnl_array['total_pnl'].mean()
        string+= "Std Pnl      :   %.2f\n"%pnl_array['total_pnl'].std()
        string+= "Daily Sharpe :   %.4f\n"%(pnl_array['total_pnl'].mean() / pnl_array['total_pnl'].std())
        string+= "Min Daily Pnl:   %.2f\n"%pnl_array['total_pnl'].min()
        string+= "Max Daily Pnl:   %.2f\n"%pnl_array['total_pnl'].max()
        string+= "Daily Volume :   %.2f\n"%pnl_array['volume'].mean()
        string+= "Prof per vol :   %.4f\n"%(pnl_array['total_pnl'].sum() / pnl_array['volume'].sum())
        string+= "Max Position :   %.0f\n"%pnl_array['max_position'].max()
        string+= "Min Position :   %.0f\n"%pnl_array['min_position'].min()
    else :
        string = "\n"
        string+= "pnl summary for %s\n"%name
        string+= "Average Pnl  :   %.2f\n"%pnl_array['total_pnl'].mean()
        string+= "Daily Volume :   %.2f\n"%pnl_array['volume'].mean()
        string+= "Prof per vol :   %.4f\n"%(pnl_array['total_pnl'].sum() / pnl_array['volume'].sum())
        string+= "Max Position :   %.0f\n"%pnl_array['max_position'].max()
        string+= "Min Position :   %.0f\n"%pnl_array['min_position'].min()
    print string



class RollSim(object):
    def __init__(self, name, train_n, sim_n):
        self.StratName = name
        self.TrainDays = train_n
        self.SimDays = sim_n
        
        self.Product = IFHV
        self.YearMonthList = ['201101', '201102', '201103', '201104',
                              '201105', '201106', '201107', '201108',
                              '201109', '201110', '201111', '201112']
        self.TrainFunc = None
        self.SimFunc = None

        self._result = {}
        self._dates = []
        
        self.pnl_array = None
        self.param_array = None
        self.pnl_array_dates = []
        self.param_array_dates = []
        
    def load_all_data(self):
        for yearmonth in self.YearMonthList:
            self._result.update(load_data(self.Product, yearmonth))
        self._dates = self._result.keys()
        self._dates.sort()

    def split_dates(self):
        one_period = self.TrainDays + self.SimDays
        assert one_period <= len(self._dates)
        date_pairs = []
        for ii in range(one_period-1, len(self._dates), self.SimDays):
            train_dates = self._dates[(ii - one_period + 1): (ii - one_period + 1 + self.TrainDays)]
            sim_dates = self._dates[(ii - one_period + 1 + self.TrainDays):(ii+1)]
            date_pairs.append((train_dates, sim_dates))
        return date_pairs

    def run(self):
        assert self.TrainFunc is not None
        assert self.SimFunc is not None
        pnl_array = None
        param_array = None
        self.pnl_array_dates = []
        self.param_array_dates = []
        date_pairs = self.split_dates()
        for train_dates, sim_dates in date_pairs:
            print "Training on %s"%(", ".join(train_dates))
            param = self.TrainFunc(self._result, train_dates)
            print "Simulate on %s"%(", ".join(sim_dates))
            pnl_tmp = self.SimFunc(self._result, sim_dates, param)
            print "Done!"
            if pnl_array is None:
                pnl_array = pnl_tmp
            else :
                pnl_array = np.append(pnl_array, pnl_tmp)
            if param_array is None:
                param_array = param
            else :
                param_array = np.append(param_array, param)
            self.pnl_array_dates += sim_dates
            self.param_array_dates.append(train_dates[-1])
        self.pnl_array = pnl_array
        self.param_array = param_array

    def print_pnl_summary(self):
        assert self.pnl_array is not None
        print 'TrainDays ==== ', self.TrainDays
        print 'SimDays   ==== ', self.SimDays
        print_summary(self.pnl_array, self.StratName)        
        
        
                               
