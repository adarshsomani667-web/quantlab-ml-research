from __future__ import annotations
import numpy as np, pandas as pd

def buy_hold(df): return pd.Series(1.,index=df.index)
def sma_crossover(df,fast=20,slow=50): return (df.Close.rolling(fast).mean()>df.Close.rolling(slow).mean()).astype(float).fillna(0)
def momentum(df,lookback=20): return (df.Close.pct_change(lookback)>0).astype(float).fillna(0)
def mean_reversion(df,window=20,z=-1.0):
    m=df.Close.rolling(window).mean(); s=df.Close.rolling(window).std(); score=(df.Close-m)/s
    return (score<z).astype(float).fillna(0)
def breakout(df,window=55): return (df.Close>df.Close.rolling(window).max().shift(1)).astype(float).fillna(0)
