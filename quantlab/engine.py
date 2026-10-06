from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def run_backtest(df, signal, fee_bps=5.0, slippage_bps=2.0):
    x = df.copy().reset_index(drop=True)
    s = pd.Series(signal).reindex(x.index).clip(0, 1).fillna(0.0)
    position = s.shift(1).fillna(0.0)
    asset_return = x['Close'].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = turnover * cost_rate
    strategy_return = position * asset_return - costs
    equity = (1 + strategy_return).cumprod()
    benchmark = (1 + asset_return).cumprod()
    return pd.DataFrame({
        'Date': x['Date'], 'position': position, 'asset_return': asset_return,
        'turnover': turnover, 'cost': costs, 'strategy_return': strategy_return,
        'equity': equity, 'benchmark': benchmark
    })

def performance(bt):
    r = bt['strategy_return'].dropna()
    if len(r) < 2: return {}
    eq = (1+r).cumprod(); years = len(r)/TRADING_DAYS
    total = eq.iloc[-1]-1
    cagr = eq.iloc[-1]**(1/years)-1 if years else np.nan
    vol = r.std(ddof=1)*np.sqrt(TRADING_DAYS)
    sharpe = r.mean()/r.std(ddof=1)*np.sqrt(TRADING_DAYS) if r.std(ddof=1)>0 else np.nan
    neg = r[r<0]
    downside = neg.std(ddof=1)
    sortino = r.mean()/downside*np.sqrt(TRADING_DAYS) if pd.notna(downside) and downside>0 else np.nan
    dd = eq/eq.cummax()-1
    max_dd = dd.min()
    calmar = cagr/abs(max_dd) if max_dd<0 else np.nan
    win = (r>0).mean()
    avg_turnover = bt['turnover'].mean()
    return {'Total Return':total,'CAGR':cagr,'Volatility':vol,'Sharpe':sharpe,'Sortino':sortino,
            'Max Drawdown':max_dd,'Calmar':calmar,'Win Rate':win,'Avg Daily Turnover':avg_turnover}

def drawdown(bt):
    eq=bt['equity']; return pd.DataFrame({'Date':bt['Date'],'Drawdown':eq/eq.cummax()-1})

def monte_carlo(r, n=500, seed=42):
    rng=np.random.default_rng(seed); r=np.asarray(r.dropna());
    if len(r)<20: return pd.DataFrame()
    out=[]
    for i in range(n):
        sample=rng.choice(r,size=len(r),replace=True); eq=np.cumprod(1+sample); dd=eq/np.maximum.accumulate(eq)-1
        out.append((eq[-1]-1,dd.min()))
    return pd.DataFrame(out,columns=['Total Return','Max Drawdown'])
