import numpy as np
from quantlab.data import make_demo_data
from quantlab.strategies import sma_crossover
from quantlab.engine import run_backtest, performance

def test_demo_data_is_valid():
    x=make_demo_data(300); assert len(x)==300; assert x.Close.gt(0).all()
def test_backtest_no_future_signal():
    x=make_demo_data(300); s=sma_crossover(x); bt=run_backtest(x,s); assert bt.position.iloc[0]==0; assert len(bt)==len(x)
def test_metrics_are_finite():
    x=make_demo_data(300); bt=run_backtest(x,sma_crossover(x)); m=performance(bt); assert np.isfinite(m['CAGR']); assert np.isfinite(m['Max Drawdown'])
