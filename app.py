import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FreeEdge Dashboard | RSI MR + Pairs + Options",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

POPULAR_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "DIA", "XLF", "XLK", "XLE",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
]

PAIR_PRESETS = {
    "SPY / QQQ": ("SPY", "QQQ"),
    "XLK / XLF": ("XLK", "XLF"),
    "IWM / SPY": ("IWM", "SPY"),
    "XLE / XLF": ("XLE", "XLF"),
    "GLD / SLV": ("GLD", "SLV"),
    "TLT / IEF": ("TLT", "IEF"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker: str, start=None, end=None, period="max"):
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as exc:
        st.error(f"Error downloading {ticker}: {exc}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker: str):
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None, None, None, None
        nearest = expirations[0]
        chain = t.option_chain(nearest)
        cols = ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
        return expirations, chain.calls[cols], chain.puts[cols], nearest
    except Exception:
        return None, None, None, None


def calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_hv(df: pd.DataFrame, window: int = 20) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def rsi_mean_reversion_signals(df, rsi_period=2, oversold=10, exit_level=50):
    out = df.copy()
    out["rsi"] = calculate_rsi(out["close"], rsi_period)
    out["signal"] = 0
    out["position"] = 0
    in_position = False
    for i in range(1, len(out)):
        rsi_val = out["rsi"].iloc[i]
        if not in_position and rsi_val <= oversold:
            out.loc[out.index[i], "signal"] = 1
            in_position = True
        elif in_position and rsi_val >= exit_level:
            out.loc[out.index[i], "signal"] = -1
            in_position = False
        out.loc[out.index[i], "position"] = 1 if in_position else 0
    return out


def calculate_metrics(equity, trades, buy_hold_returns, initial_capital):
    equity = pd.Series(equity).dropna()
    if len(equity) < 2:
        return {"error": "Insufficient data"}
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = max(days / 365.25, 0.01)
    cagr = ((equity.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100
    daily_rets = equity.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    drawdown = (equity - equity.expanding().max()) / equity.expanding().max() * 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
    avg_win = np.mean([t["return_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["return_pct"] for t in losses]) if losses else 0
    bh_cagr = ((1 + buy_hold_returns).prod() ** (252 / max(len(buy_hold_returns), 1)) - 1) * 100
    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(drawdown.min(), 2),
        "sharpe_ratio": round(sharpe, 3),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "num_trades": len(trades),
        "bh_cagr_approx": round(bh_cagr, 2),
    }


def backtest_rsi_strategy(df, initial_capital=10000.0, commission=0.001, rsi_period=2, oversold=10, exit_level=50):
    data = rsi_mean_reversion_signals(df, rsi_period, oversold, exit_level)
    data["trade_signal"] = data["signal"].shift(1).fillna(0)
    data["returns"] = data["close"].pct_change().fillna(0)
    data["equity"] = initial_capital
    data["position"] = 0
    capital = initial_capital
    shares = 0.0
    entry_price = 0.0
    entry_date = None
    trades = []
    exposure_days = 0
    for i in range(1, len(data)):
        sig = data["trade_signal"].iloc[i]
        price = float(data["close"].iloc[i])
        if shares > 0 and sig == -1:
            exit_value = shares * price * (1 - commission)
            pnl = exit_value - (shares * entry_price)
            capital = exit_value
            trades.append({
                "entry_date": entry_date,
                "exit_date": data.index[i],
                "entry_price": round(entry_price, 2),
                "exit_price": round(price, 2),
                "pnl": round(pnl, 2),
                "return_pct": round((price / entry_price - 1) * 100, 2),
                "direction": "long",
            })
            shares = 0.0
        elif shares == 0 and sig == 1:
            entry_price = price
            entry_date = data.index[i]
            shares = (capital * (1 - commission)) / price
            capital = shares * entry_price
        if shares > 0:
            exposure_days += 1
            data.loc[data.index[i], "equity"] = shares * price
            data.loc[data.index[i], "position"] = 1
        else:
            data.loc[data.index[i], "equity"] = capital
    if shares > 0:
        price = float(data["close"].iloc[-1])
        exit_value = shares * price * (1 - commission)
        pnl = exit_value - (shares * entry_price)
        capital = exit_value
        trades.append({
            "entry_date": entry_date,
            "exit_date": data.index[-1],
            "entry_price": round(entry_price, 2),
            "exit_price": round(price, 2),
            "pnl": round(pnl, 2),
            "return_pct": round((price / entry_price - 1) * 100, 2),
            "direction": "long",
        })
        data.loc[data.index[-1], "equity"] = capital
    metrics = calculate_metrics(data["equity"], trades, data["returns"], initial_capital)
    metrics["exposure_pct"] = round(exposure_days / max(len(data), 1) * 100, 1)
    metrics["trades_df"] = pd.DataFrame(trades)
    metrics["final_equity"] = round(float(data["equity"].iloc[-1]), 2)
    return data, metrics, data["equity"]


def pairs_zscore(df1, df2, window=30):
    ratio = df1["close"] / df2["close"]
    mean = ratio.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = ratio.rolling(window, min_periods=max(window // 2, 2)).std()
    return ratio, (ratio - mean) / std


def backtest_pairs(df1, df2, z_window=30, entry_z=2.0, exit_z=0.5, initial_capital=10000.0, commission=0.001):
    ratio, zscore = pairs_zscore(df1, df2, z_window)
    df = pd.DataFrame({"ratio": ratio, "zscore": zscore}).dropna()
    capital = initial_capital
    position = 0
    entry_ratio = 0.0
    entry_date = None
    trades = []
    equity = []
    dates = []
    for i in range(len(df)):
        z = float(df["zscore"].iloc[i])
        r = float(df["ratio"].iloc[i])
        date = df.index[i]
        if position == 0:
            if z > entry_z:
                position = -1
                entry_ratio = r
                entry_date = date
            elif z < -entry_z:
                position = 1
                entry_ratio = r
                entry_date = date
        elif abs(z) < exit_z:
            pnl_ratio = (r - entry_ratio) * position
            pnl = capital * pnl_ratio * 0.5
            pnl -= abs(pnl) * commission
            capital += pnl
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / max(capital - pnl, 1) * 100, 2),
                "direction": "long_spread" if position == 1 else "short_spread",
            })
            position = 0
        equity.append(capital)
        dates.append(date)
    eq_curve = pd.Series(equity, index=dates)
    metrics = calculate_metrics(eq_curve, trades, pd.Series([0] * len(eq_curve)), initial_capital)
    metrics["trades_df"] = pd.DataFrame(trades)
    metrics["final_equity"] = round(capital, 2)
    return df, metrics, eq_curve, zscore


def metric_grid(metrics):
    cols = st.columns(4)
    cols[0].metric("Total Return", f"{metrics.get('total_return_pct', 0)}%")
    cols[1].metric("CAGR", f"{metrics.get('cagr_pct', 0)}%")
    cols[2].metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0)}%")
    cols[3].metric("Sharpe", metrics.get("sharpe_ratio", 0))
    cols = st.columns(4)
    cols[0].metric("Win Rate", f"{metrics.get('win_rate_pct', 0)}%")
    cols[1].metric("Profit Factor", metrics.get("profit_factor", 0))
    cols[2].metric("Trades", metrics.get("num_trades", 0))
    cols[3].metric("Exposure", f"{metrics.get('exposure_pct', 'N/A')}%")


st.title("📊 FreeEdge Dashboard")
st.markdown("**Free, open-source trading edge explorer** — RSI mean reversion, pairs/stat arb lite, and options/volatility insights.")

with st.expander("⚠️ READ THIS FIRST — Important Disclaimers & Risk Warnings", expanded=True):
    st.error(
        """
        **THIS IS NOT FINANCIAL ADVICE. TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.**

        No strategy is foolproof or guaranteed. Backtests are educational illustrations only and can overstate performance.
        Past results do not predict future results. Options add leverage, assignment, volatility, and tail risks.
        Paper trade first. Position size conservatively. You are responsible for your own decisions.
        """
    )

st.sidebar.header("⚙️ Global Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
commission_pct = st.sidebar.slider("Commission + Slippage %", 0.0, 1.0, 0.1, 0.05) / 100

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Signals", "📈 Backtest RSI", "🔗 Pairs Explorer", "📋 Options & Vol", "📖 About",
])

with tab1:
    st.header("Live RSI Mean-Reversion Signals")
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.selectbox("Ticker", POPULAR_TICKERS, index=0)
        custom = st.text_input("Custom ticker", "")
        ticker = custom.strip().upper() or ticker
        period_choice = st.selectbox("History Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=3)
        rsi_period = st.slider("RSI Period", 2, 14, 2)
        oversold = st.slider("Oversold Entry", 5, 30, 10)
        exit_rsi = st.slider("Exit RSI", 40, 70, 50)
        run = st.button("🔄 Fetch Signal", type="primary")
    with col2:
        if run:
            df = download_data(ticker, period=period_choice)
            if df is None or len(df) < 50:
                st.error("Insufficient data. Try another ticker or longer period.")
            else:
                sig = rsi_mean_reversion_signals(df, rsi_period, oversold, exit_rsi)
                latest = sig.iloc[-1]
                st.metric("Current Price", f"${latest['close']:.2f}")
                st.metric(f"RSI({rsi_period})", f"{latest['rsi']:.1f}")
                if latest["rsi"] <= oversold:
                    st.success("🟢 Oversold mean-reversion setup detected.")
                elif latest["position"] == 1 and latest["rsi"] >= exit_rsi:
                    st.warning("🔴 Exit condition detected.")
                elif latest["position"] == 1:
                    st.info("🟡 Existing long state under this rule set.")
                else:
                    st.info("⚪ No current signal.")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=sig.index, y=sig["close"], name="Close"), row=1, col=1)
                fig.add_trace(go.Scatter(x=sig.index, y=sig["rsi"], name="RSI"), row=2, col=1)
                fig.add_hline(y=oversold, line_dash="dash", row=2, col=1)
                fig.add_hline(y=exit_rsi, line_dash="dash", row=2, col=1)
                fig.update_layout(height=500, title=f"{ticker} Price + RSI")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Data as of {sig.index[-1].strftime('%Y-%m-%d')}. Delayed public data.")

with tab2:
    st.header("RSI Mean-Reversion Backtest")
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.selectbox("Backtest Ticker", POPULAR_TICKERS, index=0, key="bt_t")
        custom = st.text_input("Custom ticker", "", key="bt_custom")
        ticker = custom.strip().upper() or ticker
        start = st.date_input("Start Date", value=datetime(2015, 1, 1))
        end = st.date_input("End Date", value=datetime.now())
        rsi_period = st.slider("RSI Period", 2, 7, 2, key="bt_rsi")
        oversold = st.slider("Oversold Entry", 5, 25, 10, key="bt_os")
        exit_rsi = st.slider("Exit RSI", 40, 65, 50, key="bt_exit")
        run = st.button("🚀 Run Backtest", type="primary")
    with col2:
        if run:
            df = download_data(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df is None or len(df) < 100:
                st.error("Insufficient data for backtest.")
            else:
                result, metrics, equity = backtest_rsi_strategy(df, initial_capital, commission_pct, rsi_period, oversold, exit_rsi)
                metric_grid(metrics)
                fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, name="Strategy Equity"))
                fig.update_layout(title="Equity Curve", height=380, yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig, use_container_width=True)
                if not metrics["trades_df"].empty:
                    st.dataframe(metrics["trades_df"].tail(25), use_container_width=True)
                else:
                    st.info("No trades generated with these parameters.")

with tab3:
    st.header("Pairs Trading / Stat-Arb Lite Explorer")
    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.selectbox("Pair", list(PAIR_PRESETS.keys()) + ["Custom"])
        if preset == "Custom":
            t1 = st.text_input("Ticker 1", "SPY").upper().strip()
            t2 = st.text_input("Ticker 2", "QQQ").upper().strip()
        else:
            t1, t2 = PAIR_PRESETS[preset]
        z_win = st.slider("Z-score Window", 20, 80, 30, 5)
        entry_z = st.slider("Entry |Z|", 1.5, 3.5, 2.0, 0.1)
        exit_z = st.slider("Exit |Z|", 0.2, 1.0, 0.5, 0.1)
        run = st.button("🔄 Analyze Pair", type="primary")
    with col2:
        if run:
            df1 = download_data(t1, period="5y")
            df2 = download_data(t2, period="5y")
            if df1 is None or df2 is None:
                st.error("Failed to download one or both tickers.")
            else:
                idx = df1.index.intersection(df2.index)
                df1, df2 = df1.loc[idx], df2.loc[idx]
                ratio, zscore = pairs_zscore(df1, df2, z_win)
                latest_z = zscore.dropna().iloc[-1]
                st.metric(f"Current {t1}/{t2} Z-score", f"{latest_z:.2f}")
                if latest_z > entry_z:
                    st.warning(f"Potential short spread: short {t1}, long {t2}.")
                elif latest_z < -entry_z:
                    st.success(f"Potential long spread: long {t1}, short {t2}.")
                else:
                    st.info("No current extreme deviation.")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio"), row=1, col=1)
                fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"), row=2, col=1)
                fig.add_hline(y=entry_z, line_dash="dash", row=2, col=1)
                fig.add_hline(y=-entry_z, line_dash="dash", row=2, col=1)
                fig.add_hline(y=0, row=2, col=1)
                fig.update_layout(height=520, title=f"{t1} / {t2} Pair Analysis")
                st.plotly_chart(fig, use_container_width=True)
                _, metrics, eq, _ = backtest_pairs(df1, df2, z_win, entry_z, exit_z, initial_capital, commission_pct)
                st.subheader("Simplified Pairs Backtest")
                metric_grid(metrics)
                st.plotly_chart(go.Figure(go.Scatter(x=eq.index, y=eq.values, name="Pairs Equity")).update_layout(height=300), use_container_width=True)

with tab4:
    st.header("Options Chain & Volatility Insights")
    ticker = st.selectbox("Ticker", POPULAR_TICKERS, index=0, key="opt_t")
    custom = st.text_input("Custom ticker", "", key="opt_custom")
    ticker = custom.strip().upper() or ticker
    if st.button("📋 Load Options + HV", type="primary"):
        expirations, calls, puts, nearest = get_options_chain(ticker)
        df = download_data(ticker, period="1y")
        if df is not None:
            current = df["close"].iloc[-1]
            hv20 = calculate_hv(df, 20).iloc[-1]
            hv60 = calculate_hv(df, 60).iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"${current:.2f}")
            c2.metric("20-day HV", f"{hv20:.1f}%")
            c3.metric("60-day HV", f"{hv60:.1f}%")
        if calls is not None and puts is not None:
            st.subheader(f"Nearest Expiry: {nearest}")
            current = df["close"].iloc[-1] if df is not None else calls["strike"].median()
            left, right = st.columns(2)
            with left:
                st.write("**Calls near ATM**")
                st.dataframe(calls[(calls["strike"] > current * 0.9) & (calls["strike"] < current * 1.1)].head(10), use_container_width=True)
            with right:
                st.write("**Puts near ATM**")
                st.dataframe(puts[(puts["strike"] > current * 0.9) & (puts["strike"] < current * 1.1)].head(10), use_container_width=True)
            st.caption("Yahoo/yfinance options data may be delayed or stale. Verify all quotes with a broker.")
        else:
            st.warning("Options chain unavailable for this ticker.")
        st.info("IV materially above realized volatility can indicate expensive premium, but short-vol strategies carry severe tail risk.")

with tab5:
    st.header("About & Usage")
    st.markdown(
        """
        FreeEdge Dashboard is a transparent research tool for exploring RSI mean reversion, pairs/stat-arb style relative value,
        and volatility-risk-premium concepts. It is built for learning and testing, not blind signal following.

        Recommended workflow:
        1. Check liquid ETFs first.
        2. Backtest across multiple market regimes.
        3. Avoid curve-fitting one ticker or one period.
        4. Paper trade before risking capital.
        5. Improve the code as your process becomes more rigorous.

        Limitations: free delayed public data, simplified execution assumptions, no broker integration, and no guarantee of future results.
        """
    )

st.markdown("---")
st.caption("FreeEdge Dashboard • Data: Yahoo Finance via yfinance • Educational only • Not financial advice")
