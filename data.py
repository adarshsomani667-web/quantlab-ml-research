from __future__ import annotations
import numpy as np, pandas as pd

def make_demo_data(n=1500, seed=7):
    rng=np.random.default_rng(seed); dates=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=n); t=np.arange(n)
    regime=np.where((t//180)%2==0,0.00028,-0.00002); cycle=.0005*np.sin(t/34); vol=np.where((t//150)%3==1,.014,.008)
    returns=regime+cycle+rng.normal(0,vol); close=100*np.exp(np.cumsum(returns)); open_=close*(1+rng.normal(0,.002,n))
    high=np.maximum(open_,close)*(1+rng.uniform(.0005,.012,n)); low=np.minimum(open_,close)*(1-rng.uniform(.0005,.012,n)); volume=rng.lognormal(15,.35,n).astype(int)
    return clean_ohlcv(pd.DataFrame({'Date':dates,'Open':open_,'High':high,'Low':low,'Close':close,'Volume':volume}))

def clean_ohlcv(df):
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=[c[0] if isinstance(c,tuple) else c for c in x.columns]
    x=x.rename(columns={c:str(c).title() for c in x.columns})
    if 'Date' not in x.columns and (isinstance(x.index,pd.DatetimeIndex) or x.index.name): x=x.reset_index().rename(columns={x.index.name or 'index':'Date'})
    req=['Date','Open','High','Low','Close','Volume']; missing=[c for c in req if c not in x.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    x['Date']=pd.to_datetime(x['Date'],errors='coerce')
    for c in req[1:]: x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=req).sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    if len(x)<100: raise ValueError('Need at least 100 valid trading rows.')
    return x

def load_yahoo(ticker,start,end):
    import yfinance as yf
    raw=yf.download(ticker,start=start,end=end,auto_adjust=False,progress=False)
    if raw.empty: raise ValueError('No market data returned. Check ticker and date range.')
    return clean_ohlcv(raw.reset_index())
