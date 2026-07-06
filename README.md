# FreeEdge Dashboard

FreeEdge Dashboard is a free Streamlit research dashboard for exploring transparent, research-backed trading ideas using public market data from `yfinance`.

It is designed for education, idea generation, and disciplined backtesting. It is not a signal service, not financial advice, and not a money printer.

## What is included

### 1. Live RSI(2) mean-reversion signals

- Daily-bar RSI(2) readings for liquid stocks and ETFs.
- Oversold and overbought detection.
- Price and RSI charts.
- Customizable RSI period, oversold threshold, and overbought threshold.
- Useful primarily for swing-trading research, not scalping.

### 2. Full backtest engine

- Long-only mean-reversion backtest.
- Next-bar execution assumption for more realistic testing.
- Adjustable RSI period, entry level, exit level, starting capital, commission, and slippage.
- Equity curve, drawdown, CAGR, max drawdown, Sharpe, win rate, profit factor, exposure, and trade log.

### 3. Pairs trading / stat-arb lite explorer

- Price-ratio z-score explorer for common ETF and stock pairs.
- Custom pair input.
- Live z-score signal.
- Simple historical spread mean-reversion backtest.
- Based on the broad research tradition around statistical arbitrage and relative-value pairs trading.

### 4. Options and volatility insights

- Nearest-expiry options chain viewer through Yahoo Finance / yfinance when available.
- ATM-ish calls and puts.
- Implied-volatility display from the chain.
- 20-day and 60-day realized/historical volatility calculations.
- Educational comparison of IV versus HV, with heavy tail-risk caveats.

## Important disclaimers

This app is for education and research only. It is not investment advice, financial advice, trading advice, or a recommendation to buy, sell, short, hedge, or trade any security, option, ETF, or derivative.

Backtests are simplified. Real trading results can be materially worse because of bid/ask spreads, commissions, slippage, taxes, liquidity, borrow constraints, survivorship bias, regime change, overfitting, behavioral mistakes, and execution errors.

No strategy is guaranteed. Edges decay. Historical performance does not imply future performance. You can lose substantial money. Paper trade first and use conservative position sizing.

Options involve additional risks, including total premium loss, assignment, early exercise, volatility crush, gap risk, liquidity risk, margin risk, and tail events.

## Local setup

```bash
git clone https://github.com/danielknapp21-ux/freeedge-trading-dashboard.git
cd freeedge-trading-dashboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Free deployment on Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Connect GitHub.
3. Select this repository.
4. Use branch `main`.
5. Use entrypoint `app.py`.
6. Deploy.

The app will auto-update when new commits are pushed to `main`.

## Suggested workflow

1. Start with the Live Signals tab using SPY, QQQ, IWM, DIA, XLF, XLK, XLE, NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA.
2. Backtest multiple regimes separately, such as 2015-2019, 2020, 2022, and 2023-present.
3. Avoid optimizing one ticker and one period until the backtest looks perfect.
4. Use tiny position sizes or paper trading before considering real money.
5. Keep notes on every test and avoid cherry-picking.

## Research context

The dashboard focuses on transparent, commonly researched market anomalies and frameworks:

- Short-term equity/ETF mean reversion using extreme RSI-style overreaction measures.
- Relative-value / pairs trading based on mean-reverting spreads or ratios.
- Volatility risk premium concepts comparing implied volatility and realized volatility.

These ideas have historical and academic support, but real-world profitability depends heavily on costs, risk control, instrument choice, implementation quality, and market regime.

## Data limitations

- Data comes from Yahoo Finance through `yfinance`.
- Daily bars are the primary intended timeframe.
- Options data availability varies by ticker and market.
- Free public data may be delayed, incomplete, adjusted, or unavailable.
- Intraday/scalping workflows are intentionally limited.

## License

Educational use. Add a formal license if you plan to distribute publicly.
