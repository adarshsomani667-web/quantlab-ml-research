# QuantLab — Quantitative Research Lab

**Version 2.0 — portfolio/research edition**

QuantLab is a research-first algorithmic trading platform designed to answer a harder question than “can I make a backtest profitable?”:

> **Does an apparent market pattern survive realistic execution costs, out-of-sample testing and risk analysis?**

## Why this is a serious project

QuantLab combines financial statistics, machine learning, software engineering and experimental design. It deliberately includes controls for common backtesting mistakes instead of hiding them.

### Included

- Historical OHLCV ingestion: built-in dataset, Yahoo Finance, or CSV upload
- Five transparent baseline strategies
- Random Forest out-of-sample classifier
- Chronological train/test split
- One-day signal lag
- Commission + slippage modelling
- CAGR, volatility, Sharpe, Sortino, Calmar, drawdown and win rate
- Risk/return frontier
- Bootstrap robustness diagnostic
- Market-regime classification
- Feature importance and confusion matrix
- Annual strategy-return analysis
- CSV and research-report export
- Unit tests

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The built-in research dataset means the app works without an external market-data account.

## Data format for CSV upload

Required columns:

`Date, Open, High, Low, Close, Volume`

## Research methodology

The ML label is tomorrow's direction. Features are constructed from current/past observations. The model is trained chronologically and evaluated on later observations. Signals are shifted one trading day in the execution engine. Trading costs are charged whenever exposure changes.

These choices do not eliminate every source of bias. They make the assumptions visible and give the researcher a foundation for stronger experiments.

## Suggested research paper

A strong accompanying paper can investigate:

**“Do machine-learning trading signals provide economically meaningful out-of-sample value after costs and risk?”**

Report the hypothesis, data, feature design, competing strategies, validation method, results, failure cases, robustness tests and limitations. A negative result is valid research if the methodology is sound.

## Limitations

This is an educational/research platform. It is not a broker, investment adviser or production trading system. It does not model market impact, borrow costs, taxes, corporate actions perfectly, survivorship bias across a full universe, or live execution latency. Yahoo Finance availability and historical data quality can vary.

## License

MIT
