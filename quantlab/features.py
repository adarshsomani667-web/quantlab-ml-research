from __future__ import annotations
import numpy as np, pandas as pd

def rsi(c,p=14):
    d=c.diff(); gain=d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean(); loss=-d.clip(upper=0).ewm(alpha=1/p,adjust=False).mean(); rs=gain/loss.replace(0,np.nan); return 100-100/(1+rs)

def build_features(df):
    x=df.copy(); c=x['Close']; r=c.pct_change()
    x['ret_1d']=r; x['ret_5d']=c.pct_change(5); x['ret_20d']=c.pct_change(20); x['ret_60d']=c.pct_change(60)
    x['vol_20d']=r.rolling(20).std()*np.sqrt(252); x['vol_60d']=r.rolling(60).std()*np.sqrt(252)
    for p in (10,20,50,100): x[f'price_sma{p}']=c/c.rolling(p).mean()-1
    x['rsi_14']=rsi(c); x['range_pct']=(x['High']-x['Low'])/c; x['volume_z']=(x['Volume']-x['Volume'].rolling(30).mean())/x['Volume'].rolling(30).std()
    x['target_return']=c.pct_change().shift(-1); x['target']=(x['target_return']>0).astype(int)
    return x
