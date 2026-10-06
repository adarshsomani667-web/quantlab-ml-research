from __future__ import annotations
import numpy as np, pandas as pd

def regime_label(df):
    vol=df['Close'].pct_change().rolling(30).std()*np.sqrt(252); trend=df['Close'].pct_change(60)
    return pd.Series(np.select([(vol>vol.median())&(trend<0),(vol>vol.median())&(trend>=0),(vol<=vol.median())&(trend<0)],['High Vol / Downtrend','High Vol / Uptrend','Low Vol / Downtrend'],'Low Vol / Uptrend'),index=df.index)

def bootstrap_summary(bt,n=1000,seed=42):
    from .engine import monte_carlo
    sim=monte_carlo(bt['strategy_return'],n,seed)
    if sim.empty:return {}
    return {'median_return':sim['Total Return'].median(),'p05_return':sim['Total Return'].quantile(.05),'p95_return':sim['Total Return'].quantile(.95),'median_drawdown':sim['Max Drawdown'].median(),'p05_drawdown':sim['Max Drawdown'].quantile(.05)}

def annual_returns(bt):
    x=bt.set_index('Date')['strategy_return'].groupby(lambda d:d.year).apply(lambda r:(1+r).prod()-1); return x
