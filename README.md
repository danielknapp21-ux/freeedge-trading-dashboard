# FreeEdge Dashboard - Proven Edge Trading Tools (Free & Open Source)

A fully functional, free Streamlit web application for stocks, ETFs, and options trading analysis. Focuses on **data-backed, non-mainstream quantitative edges** rather than hype or black-box signals.

**Core Strategies Included (with historical research backing):**
- **Short-term RSI(2) Mean Reversion Swing Strategy**: Inspired by Larry Connors research and multiple independent backtests. Captures short-term overreactions in liquid instruments. High win-rate potential in ranging/swing environments, low market exposure.
- **Pairs Trading / Statistical Arbitrage Lite**: Classic quant relative value approach. Trades the spread/z-score between correlated assets (e.g., sector ETFs, SPY/QQQ). Academic literature (e.g., Gatev, Goetzmann, Rouwenhorst 2006 and many follow-ups) documents profitability before costs in many pairs due to mean-reversion of relative prices.
- **Volatility & Options Insights**: Basic tools to explore IV proxies vs realized vol for premium selling ideas (volatility risk premium is a well-studied edge in options).

**NO "foolproof" claims.** Markets are competitive. All edges are probabilistic, regime-dependent, and subject to decay, costs, slippage, and black swans. This is an **educational/research tool** to help you explore, backtest ideas rigorously, and understand risk metrics. 

**CRITICAL DISCLAIMERS (Read Fully):**
- **NOT financial, investment, or trading advice.** 
- Trading and investing involve **substantial risk of loss**, including possible loss of principal. Past performance (backtests or live) does not guarantee future results.
- Backtests are simplified: they typically use close prices, ignore or approximate commissions/slippage/spreads/taxes/borrow costs/liquidity constraints/partial fills/psychological factors. Real trading results will differ (usually worse).
- No strategy is "proven" to work forever. You must forward-test (paper trade) extensively in current market conditions.
- Position sizing, risk management (stops, max drawdown limits, diversification), and broker execution quality are YOUR responsibility.
- Options trading involves additional risks (leverage, expiration, assignment).
- Data from yfinance/Yahoo is delayed and for informational purposes only. Not real-time execution quality.
- Always do your own due diligence. Consider consulting a licensed financial advisor. The authors/contributors assume no liability.

Use this tool responsibly. Start small, paper trade, understand every number on the screen.

## Features
- **Live Signals Tab**: Real-time (delayed) price, RSI(2)/RSI(14), current signal for swing entry/exit on any ticker. Customizable thresholds.
- **Backtest Engine Tab**: Full historical backtester for the RSI mean reversion strategy. Adjustable params (RSI period, oversold/overbought levels, exit level, start/end dates, initial capital, commission %). Outputs:
  - Interactive equity curve (Plotly)
  - Key metrics: CAGR, Total Return, Max Drawdown, Sharpe Ratio (rf=0), Win Rate, Profit Factor, # Trades, Avg Win/Loss, Exposure %
  - Simple trade log summary
- **Pairs Trading Explorer Tab**: Select or enter two tickers. Computes rolling z-score of price ratio/spread. Live signal + historical backtest of the pairs strategy (long/short the spread when z-score extreme). Shows performance metrics and spread/z-score charts. Great for relative value ideas on ETFs.
- **Options & Volatility Tab**: For any ticker, view available options expirations and sample chain (ATM focus). Computes historical realized volatility (HV) as proxy. Highlights potential premium selling opportunities when HV low relative to typical IV environment (user should verify live IV on broker platform). Educational only.
- **Fully Customizable & Transparent**: Change strategy params, dates, capital. All calculations visible in code.
- **Free Forever**: 100% free data (yfinance), free hosting/deploy options.

## How to Run Locally (Free)
1. Clone/fork this repo.
2. `cd free_trading_edge_dashboard`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
5. Open browser to the local URL shown.

## Deploy as Free Public Webapp (Recommended)
**Easiest (Streamlit Cloud - Free tier sufficient for this):**
1. Push/fork this repo to your public GitHub account.
2. Go to https://share.streamlit.io/
3. Sign in with GitHub.
4. Click "New app" → select this repo → main branch → `app.py` as entrypoint.
5. Deploy. It will be live at a share.streamlit.io URL you control. Free, auto-updates on git push.
6. (Optional) Add secrets if needed later, but none required here.

**Alternatives**: Hugging Face Spaces (Gradio/Streamlit), Render.com free tier, Railway.app hobby, Vercel (with adjustments), or self-host.

GitHub Pages alone won't work well (static only); Streamlit Cloud makes the Python backend + interactive charts work for free.

## Why These Strategies? (Brief Research Context)
- **RSI(2) Mean Reversion**: Short-term RSI extremes often precede reversals due to overreaction/microstructure. Multiple practitioner books (Connors) and independent quant sites show positive expectancy in backtests on indices/ETFs/stocks, especially with tight risk. Low holding periods = lower drawdown risk per trade.
- **Pairs/Stat Arb**: Relative prices of economically linked assets tend to mean-revert. Extensive finance literature documents profits from simple distance or cointegration methods on US equities/ETFs, though capacity limited and costs matter. This "lite" version uses z-score for accessibility.
- **Vol Premium**: Selling options premium when implied vol is elevated vs. realized has positive edge on average (insurance selling), documented in many volatility trading papers. App helps screen for candidates.

These are **not** the most common retail TA (no default 14-period RSI crossover hype). They require discipline and are "quant-ish".

## Limitations & Roadmap (Community Welcome)
- Data: Daily bars primarily (yfinance limits intraday history). For true scalping/intraday, integrate paid low-latency feed later (Polygon, etc.).
- No live broker execution or alerts (add via webhooks/Twilio later if desired).
- Backtests don't include dividends reinvestment perfectly in all cases or corporate actions edge cases (yfinance adjusted close helps).
- Pairs hedge ratio is simplified (ratio-based); full OLS/ cointegration test can be added.
- No machine learning or alternative data (future?).
- Mobile experience basic.

Pull requests for improvements, more robust backtesting (e.g. vectorbt integration), additional filters (volume, ATR, regime via ADX), or better pairs cointegration welcome!

## Tech Stack
- Python + Streamlit (UI + caching)
- yfinance (free market data: prices, options chains)
- pandas/numpy/scipy (analysis)
- plotly (interactive charts)

## License
MIT or similar - free to use/modify for personal/educational/commercial? Check code. No warranty.

**Start here**: Deploy it, paper trade the signals on 1-2 liquid tickers (SPY, QQQ, IWM recommended for starters), run backtests on different regimes (bull, bear, sideways), compare to buy-and-hold. Learn what works for YOUR risk tolerance.

Trade smart. Risk little per idea. Compound the process, not just returns.

Questions? Open GitHub issue. Good luck!