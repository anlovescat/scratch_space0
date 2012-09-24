from hft_sim_lib import *
from hft_signal_lib import *
from hft_strat_lib import *


def run_regression(result, dates):
    mid = []
    vwap = [] #0.1
    trsign = [] #0.7
    signedv = [] #0.7
    bkema = [] #0.05
    trsvol = [] #0.7
    last = []
    
    for adate in dates:
        data = result[adate]
        
    
    pass

def optimize_thresh(result, dates, reg_param):
    pass

def optimize_in_sample(result, dates):
    pass

def sim_out_sample(result, dates, param):
    pass


rsim = RollSim('trigger', 5, 2)
rsim.TrainFunc = optimize_in_sample
rsim.SimFunc = sim_out_sample

rsim.load_all_data()
rsim.split_dates()
rsim.run()
rsim.print_pnl_summary()


