from __future__ import annotations
import io, textwrap
import numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
from quantlab.data import make_demo_data, load_yahoo, clean_ohlcv
from quantlab.features import build_features
from quantlab.strategies import buy_hold, sma_crossover, momentum, mean_reversion, breakout
from quantlab.engine import run_backtest, performance, drawdown
from quantlab.ml import walk_forward_ml
from quantlab.research import regime_label, bootstrap_summary, annual_returns

st.set_page_config(page_title='QuantLab — Quantitative Research Lab',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:1.2rem}.hero{padding:1.4rem 1.6rem;border:1px solid rgba(128,128,128,.25);border-radius:18px;background:linear-gradient(120deg,rgba(80,100,140,.10),rgba(255,255,255,.02));margin-bottom:1rem}.hero h1{font-size:2.5rem;margin:0}.muted{opacity:.7}.tag{display:inline-block;padding:.25rem .55rem;border:1px solid rgba(128,128,128,.25);border-radius:999px;margin-right:.3rem;font-size:.8rem}.section{margin-top:1rem}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_demo(): return make_demo_data()
@st.cache_data(show_spinner=False)
def get_yahoo(ticker,start,end): return load_yahoo(ticker,str(start),str(end))

with st.sidebar:
    st.markdown('## ◈ QuantLab')
    st.caption('Quantitative research & strategy laboratory')
    source=st.radio('Market data',['Built-in research dataset','Yahoo Finance','Upload CSV'])
    if source=='Yahoo Finance':
        ticker=st.text_input('Ticker','SPY').upper().strip(); start=st.date_input('Start',pd.Timestamp('2014-01-01').date()); end=st.date_input('End',pd.Timestamp.today().date())
    elif source=='Upload CSV':
        ticker='UPLOAD'; uploaded=st.file_uploader('OHLCV CSV',type=['csv'])
    else: ticker='DEMO'
    st.divider(); st.markdown('### Backtest assumptions')
    fee=st.slider('Commission (bps)',0,50,5); slip=st.slider('Slippage (bps)',0,50,2)
    st.divider(); st.markdown('### ML assumptions')
    train_frac=st.slider('Training fraction',0.50,0.85,0.70,0.05); threshold=st.slider('Long probability threshold',0.50,0.75,0.55,0.01)
    st.caption('Signals are shifted one trading day. Costs are charged on turnover. ML uses chronological out-of-sample testing.')

try:
    if source=='Built-in research dataset': df=get_demo()
    elif source=='Yahoo Finance': df=get_yahoo(ticker,start,end)
    else:
        if uploaded is None: st.info('Upload a CSV with Date, Open, High, Low, Close, Volume. The built-in dataset is ready to explore now.'); st.stop()
        df=clean_ohlcv(pd.read_csv(uploaded))
except Exception as e: st.error(str(e)); st.stop()

feat=build_features(df); regimes=regime_label(df)
strategies={'Buy & Hold':buy_hold(df),'SMA 20/50':sma_crossover(df),'Momentum 20D':momentum(df),'Mean Reversion':mean_reversion(df),'Breakout 55D':breakout(df)}
results={}; bts={}
for name,sig in strategies.items(): bts[name]=run_backtest(df,sig,fee,slip); results[name]=performance(bts[name])
ml_signal,ml_stats=walk_forward_ml(feat,train_frac,threshold); bts['ML Random Forest']=run_backtest(df,ml_signal,fee,slip); results['ML Random Forest']=performance(bts['ML Random Forest'])

st.markdown('''<div class="hero"><div class="hero-top"><div><div class="eyebrow">QUANTITATIVE RESEARCH PLATFORM</div><h1>QuantLab</h1><div class="muted">Research whether market patterns survive costs, risk controls and out-of-sample validation.</div></div><div class="status">● RESEARCH MODE</div></div><div style="margin-top:.9rem"><span class="tag">BACKTESTING</span><span class="tag">MACHINE LEARNING</span><span class="tag">RISK ENGINE</span><span class="tag">ROBUSTNESS</span></div></div>''',unsafe_allow_html=True)

st.markdown('''<style>.hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}.eyebrow{font-size:.72rem;letter-spacing:.14em;font-weight:700;opacity:.65;margin-bottom:.35rem}.status{font-size:.72rem;letter-spacing:.08em;padding:.42rem .65rem;border:1px solid rgba(128,128,128,.28);border-radius:999px}.kpi{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:1rem 1.05rem;background:rgba(128,128,128,.035);min-height:105px}.kpi-label{font-size:.68rem;letter-spacing:.1em;font-weight:700;opacity:.62}.kpi-value{font-size:1.65rem;font-weight:750;margin-top:.35rem}.kpi-sub{font-size:.72rem;opacity:.62;margin-top:.18rem}.dash-title{font-size:1.05rem;font-weight:750;margin:.8rem 0 .5rem}</style>''',unsafe_allow_html=True)

best=max(results,key=lambda k:results[k]['Sharpe'] if pd.notna(results[k]['Sharpe']) else -99)
best_r=results[best]
cols=st.columns(5)
kpis=[('BEST SHARPE',f"{best_r['Sharpe']:.2f}",best,'risk-adjusted leader'),('BEST CAGR',f"{best_r['CAGR']:.1%}",best,'annualized growth'),('MAX DRAWDOWN',f"{best_r['Max Drawdown']:.1%}",best,'peak-to-trough'),('ML TEST ACCURACY',f"{ml_stats['Accuracy']:.1%}",'Random Forest','unseen test period'),('DATASET',f"{len(df):,}",'observations',f"{df.Date.min().date()} → {df.Date.max().date()}")]
for col,(lab,val,sub,foot) in zip(cols,kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub} · {foot}</div></div>',unsafe_allow_html=True)
st.caption(f"{ticker} · {fee} bps commission + {slip} bps slippage · signals shifted one trading day · chronological ML testing")

with st.container(border=True):
    st.markdown('<div class="dash-title">Out-of-Sample Equity Curves</div>',unsafe_allow_html=True)
    e=go.Figure()
    for n,bt in bts.items():
        e.add_trace(go.Scatter(x=bt.Date,y=bt.equity,name=n,line=dict(width=3 if n in [best,'ML Random Forest'] else 1.5)))
    e.update_layout(height=430,yaxis_title='Growth of $1',hovermode='x unified',legend=dict(orientation='h',y=1.08),margin=dict(t=40,b=20))
    st.plotly_chart(e,use_container_width=True)

left,right=st.columns([1.55,1])
with left:
    st.markdown('<div class="dash-title">Strategy Scorecard</div>',unsafe_allow_html=True)
    score=pd.DataFrame(results).T[['CAGR','Sharpe','Sortino','Max Drawdown','Volatility','Win Rate']].copy()
    for col in ['CAGR','Max Drawdown','Volatility','Win Rate']: score[col]=score[col].map(lambda x:f'{x:.1%}')
    for col in ['Sharpe','Sortino']: score[col]=score[col].map(lambda x:f'{x:.2f}' if pd.notna(x) else '—')
    st.dataframe(score,use_container_width=True,height=250)
with right:
    st.markdown('<div class="dash-title">ML Research Snapshot</div>',unsafe_allow_html=True)
    a,b=st.columns(2); a.metric('Accuracy',f"{ml_stats['Accuracy']:.1%}"); b.metric('ROC AUC',f"{ml_stats['ROC AUC']:.2f}")
    imp=ml_stats['Feature Importance'].sort_values(ascending=True).tail(5)
    f=go.Figure(go.Bar(x=imp.values,y=imp.index,orientation='h')); f.update_layout(height=220,margin=dict(l=10,r=10,t=10,b=10),xaxis_title='Importance',yaxis_title='')
    st.plotly_chart(f,use_container_width=True)

left2,right2=st.columns(2)
with left2:
    st.markdown('<div class="dash-title">Drawdown Monitor</div>',unsafe_allow_html=True)
    dd=drawdown(bts[best]); f=go.Figure(go.Scatter(x=dd.Date,y=dd.Drawdown,fill='tozeroy',name=best)); f.update_layout(height=260,yaxis_tickformat='.0%',yaxis_title='Drawdown',margin=dict(t=15,b=20)); st.plotly_chart(f,use_container_width=True)
with right2:
    st.markdown('<div class="dash-title">Risk / Return Frontier</div>',unsafe_allow_html=True)
    rr=pd.DataFrame({n:{'Volatility':v['Volatility'],'CAGR':v['CAGR']} for n,v in results.items()}).T
    f=go.Figure(go.Scatter(x=rr.Volatility,y=rr.CAGR,text=rr.index,mode='markers+text',textposition='top center',marker=dict(size=10))); f.update_layout(height=260,xaxis_tickformat='.0%',yaxis_tickformat='.0%',xaxis_title='Annualized volatility',yaxis_title='CAGR',margin=dict(t=15,b=20)); st.plotly_chart(f,use_container_width=True)

tab1,tab2,tab3,tab4,tab5=st.tabs(['Overview','Strategy Lab','ML Lab','Risk & Robustness','Research Notes'])
with tab1:
    st.subheader('Price & equity curves')
    p=go.Figure(); p.add_trace(go.Scatter(x=df.Date,y=df.Close,name='Close')); p.update_layout(height=380,yaxis_title='Price',hovermode='x unified'); st.plotly_chart(p,use_container_width=True)
    e=go.Figure();
    for n,bt in bts.items(): e.add_trace(go.Scatter(x=bt.Date,y=bt.equity,name=n))
    e.update_layout(height=460,yaxis_title='Growth of $1',hovermode='x unified'); st.plotly_chart(e,use_container_width=True)
    st.subheader('Strategy scorecard')
    score=pd.DataFrame(results).T
    show=score.copy()
    for col in ['Total Return','CAGR','Volatility','Max Drawdown','Win Rate','Avg Daily Turnover']: show[col]=show[col].map(lambda x:f'{x:.2%}')
    for col in ['Sharpe','Sortino','Calmar']: show[col]=show[col].map(lambda x:f'{x:.2f}' if pd.notna(x) else '—')
    st.dataframe(show,use_container_width=True)

with tab2:
    st.subheader('Compare simple hypotheses')
    st.write('The goal is not to find the prettiest backtest. It is to compare hypotheses under identical execution assumptions.')
    chosen=st.multiselect('Strategies to display',list(bts),default=list(bts.keys()))
    fig=go.Figure()
    for n in chosen: fig.add_trace(go.Scatter(x=bts[n].Date,y=bts[n].equity,name=n))
    fig.update_layout(height=500,yaxis_title='Growth of $1',hovermode='x unified'); st.plotly_chart(fig,use_container_width=True)
    annual=pd.DataFrame({n:annual_returns(bts[n]) for n in chosen}); st.dataframe(annual.style.format('{:.2%}'),use_container_width=True)
    st.download_button('Download strategy returns CSV',annual.to_csv().encode(),file_name='quantlab_strategy_returns.csv',mime='text/csv')

with tab3:
    st.subheader('Out-of-sample machine learning lab')
    a,b,c,d=st.columns(4); a.metric('Accuracy',f"{ml_stats['Accuracy']:.1%}"); b.metric('Precision',f"{ml_stats['Precision']:.1%}"); c.metric('Recall',f"{ml_stats['Recall']:.1%}"); d.metric('ROC AUC',f"{ml_stats['ROC AUC']:.2f}")
    left,right=st.columns(2)
    with left:
        imp=ml_stats['Feature Importance'].sort_values(); f=go.Figure(go.Bar(x=imp.values,y=imp.index,orientation='h')); f.update_layout(height=500,title='Feature importance'); st.plotly_chart(f,use_container_width=True)
    with right:
        cm=ml_stats['Confusion']; f=go.Figure(go.Heatmap(z=cm,x=['Predicted Down','Predicted Up'],y=['Actual Down','Actual Up'],text=cm,texttemplate='%{text}')); f.update_layout(height=500,title='Out-of-sample confusion matrix'); st.plotly_chart(f,use_container_width=True)
    pred=ml_stats['Predictions'][['Date','target','prediction','prob_up']].copy(); pred['signal']=pred.prob_up.ge(threshold).astype(int); st.dataframe(pred.tail(30),use_container_width=True)
    st.info('Research discipline: the ML model is trained only on earlier observations and evaluated on later unseen observations. Accuracy alone is not treated as evidence of profitability.')

with tab4:
    st.subheader('Risk & robustness')
    selected=st.selectbox('Strategy',list(bts),index=list(bts).index('ML Random Forest'))
    bt=bts[selected]
    left,right=st.columns(2)
    with left:
        dd=drawdown(bt); f=go.Figure(go.Scatter(x=dd.Date,y=dd.Drawdown,fill='tozeroy',name='Drawdown')); f.update_layout(height=400,yaxis_tickformat='.0%',title='Drawdown'); st.plotly_chart(f,use_container_width=True)
    with right:
        rr=pd.DataFrame({n:{'Volatility':v['Volatility'],'CAGR':v['CAGR']} for n,v in results.items()}).T; f=go.Figure(go.Scatter(x=rr.Volatility,y=rr.CAGR,text=rr.index,mode='markers+text',textposition='top center')); f.update_layout(height=400,xaxis_tickformat='.0%',yaxis_tickformat='.0%',title='Risk / return frontier',xaxis_title='Annualized volatility',yaxis_title='CAGR'); st.plotly_chart(f,use_container_width=True)
    st.subheader('Bootstrap robustness check')
    boot=bootstrap_summary(bt,1000)
    if boot:
        q1,q2,q3,q4=st.columns(4); q1.metric('Median simulated return',f"{boot['median_return']:.1%}"); q2.metric('5th pct return',f"{boot['p05_return']:.1%}"); q3.metric('95th pct return',f"{boot['p95_return']:.1%}"); q4.metric('Median max drawdown',f"{boot['median_drawdown']:.1%}")
        st.caption('Bootstrap resamples daily strategy returns. It is a robustness diagnostic, not a forecast or guarantee.')
    st.subheader('Market regimes')
    reg=pd.DataFrame({'Regime':regimes,'Return':df.Close.pct_change()}).dropna().groupby('Regime').agg(Observations=('Return','size'),Mean_Daily_Return=('Return','mean'),Volatility=('Return','std')).sort_values('Mean_Daily_Return',ascending=False); st.dataframe(reg.style.format({'Mean_Daily_Return':'{:.3%}','Volatility':'{:.3%}'}),use_container_width=True)

with tab5:
    st.subheader('What this project is actually testing')
    mlc=results['ML Random Forest']; bh=results['Buy & Hold']
    if mlc['CAGR']>bh['CAGR'] and mlc['Max Drawdown']>bh['Max Drawdown']: conclusion='The ML strategy beats Buy & Hold on both CAGR and maximum drawdown in this sample. This is evidence for further investigation, not proof of a persistent edge.'
    elif mlc['CAGR']>bh['CAGR']: conclusion='The ML strategy has higher CAGR than Buy & Hold, but its risk profile is not uniformly better. The next question is whether the result survives other periods and assumptions.'
    else: conclusion='The ML strategy does not beat Buy & Hold on CAGR in this sample. That is a valuable research result: model complexity does not automatically create investment value.'
    st.info(conclusion)
    st.markdown('''**Core research question**  
Can machine-learning pattern detection create economically meaningful out-of-sample performance after transaction costs and risk are considered?

**Controls against common research errors**
- Chronological train/test split rather than random shuffling.
- One-day signal lag to reduce look-ahead bias.
- Explicit commission and slippage assumptions.
- Benchmark comparison against Buy & Hold.
- Multiple simple strategies as competing hypotheses.
- Drawdown, volatility and risk-adjusted metrics rather than return alone.
- Bootstrap robustness diagnostics.

**Next research questions**
1. Does the result survive across assets and market regimes?
2. Does walk-forward retraining improve stability?
3. How sensitive are results to the probability threshold?
4. Does the apparent edge survive higher transaction costs?
5. Is the improvement statistically and economically meaningful?
''')
    report=f'''QUANTLAB — RESEARCH REPORT\n{'='*40}\nAsset: {ticker}\nPeriod: {df.Date.min().date()} to {df.Date.max().date()}\nObservations: {len(df):,}\nExecution costs: {fee} bps commission + {slip} bps slippage\nML train fraction: {train_frac:.0%}\nML threshold: {threshold:.2f}\n\nML OUT-OF-SAMPLE\nAccuracy: {ml_stats['Accuracy']:.4f}\nPrecision: {ml_stats['Precision']:.4f}\nRecall: {ml_stats['Recall']:.4f}\nROC AUC: {ml_stats['ROC AUC']:.4f}\nTrain rows: {ml_stats['Train Rows']}\nTest rows: {ml_stats['Test Rows']}\n\nSTRATEGY RESULTS\n{pd.DataFrame(results).T.to_string()}\n\nINTERPRETATION\n{conclusion}\n\nThis is a hypothetical research backtest, not investment advice. Historical performance does not guarantee future results.\n'''
    st.download_button('Download full research report',report.encode(),file_name='QuantLab_research_report.txt',mime='text/plain')
