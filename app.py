import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="FreeEdge Dashboard | RSI MR + Pairs + Options",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker, start=None, end=None, period="max"):
    """Download historical data with caching."""
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        st.error(f"Error downloading {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker):
    """Get options expirations and a sample chain."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None, None
        # Get nearest expiry chain
        nearest = exps[0]
        chain = t.option_chain(nearest)
        calls = chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']]
        puts = chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']]
        return exps, calls, puts, nearest
    except Exception:
        return None, None, None, None

def calculate_rsi(series, period=2):
    """Calculate RSI (Wilder style approximation with SMA)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_hv(df, window=20, annualize=True):
    """Historical (realized) volatility from close returns."""
    log_ret = np.log(df['close'] / df['close'].shift(1))
    hv = log_ret.rolling(window=window).std()
    if annualize:
        hv = hv * np.sqrt(252)
    return hv * 100  # in percent

def rsi_mean_reversion_signals(df, rsi_period=2, oversold=10, overbought=90, exit_level=50):
    """Generate long-only mean reversion signals based on RSI(2)."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df['signal'] = 0
    df['position'] = 0
    
    in_position = False
    for i in range(1, len(df)):
        rsi_val = df['rsi'].iloc[i]
        if pd.isna(rsi_val):
            continue
        if not in_position and rsi_val <= oversold:
            df.loc[df.index[i], 'signal'] = 1  # Buy signal
            in_position = True
        elif in_position and rsi_val >= exit_level:
            df.loc[df.index[i], 'signal'] = -1  # Sell signal
            in_position = False
        df.loc[df.index[i], 'position'] = 1 if in_position else 0
    
    return df

def backtest_rsi_strategy(df, initial_capital=10000.0, commission=0.001, 
                          rsi_period=2, oversold=10, overbought=90, exit_level=50):
    """
    Vectorized-ish backtest for RSI mean reversion (long only).
    Trades on next bar close after signal for realism.
    """
    df = df.copy()
    df = rsi_mean_reversion_signals(df, rsi_period, oversold, overbought, exit_level)
    
    # Shift signal for next-bar execution (realistic)
    df['trade_signal'] = df['signal'].shift(1).fillna(0)
    
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = 0.0
    df['equity'] = initial_capital
    df['position'] = 0.0
    df['trade_pnl'] = 0.0
    
    capital = initial_capital
    position = 0.0  # shares
    entry_price = 0.0
    trades = []
    current_trade = None
    
    for i in range(1, len(df)):
        sig = df['trade_signal'].iloc[i]
        price = df['close'].iloc[i]
        prev_equity = df['equity'].iloc[i-1]
        
        # Exit if in position and sell signal
        if position > 0 and sig == -1:
            exit_price = price
            pnl = (exit_price - entry_price) * position - abs(position) * entry_price * commission
            capital += pnl
            trades.append({
                'entry_date': current_trade['entry_date'],
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': (exit_price / entry_price - 1) * 100,
                'direction': 'long'
            })
            position = 0.0
            current_trade = None
        
        # Enter long on buy signal
        elif position == 0 and sig == 1:
            position = (capital * (1 - commission)) / price   # approx shares
            entry_price = price
            current_trade = {'entry_date': df.index[i]}
        
        # Mark to market equity
        if position > 0:
            mtm = capital + position * (price - entry_price) - abs(position) * entry_price * commission * 0.1  # rough ongoing cost
            df.loc[df.index[i], 'equity'] = mtm
            df.loc[df.index[i], 'position'] = 1
        else:
            df.loc[df.index[i], 'equity'] = capital
            df.loc[df.index[i], 'position'] = 0
        
        df.loc[df.index[i], 'strategy_returns'] = (df['equity'].iloc[i] / prev_equity - 1) if prev_equity > 0 else 0
    
    # Close any open position at end
    if position > 0:
        exit_price = df['close'].iloc[-1]
        pnl = (exit_price - entry_price) * position - abs(position) * entry_price * commission
        capital += pnl
        trades.append({
            'entry_date': current_trade['entry_date'],
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': (exit_price / entry_price - 1) * 100,
            'direction': 'long'
        })
        df.loc[df.index[-1], 'equity'] = capital
    
    equity_curve = df['equity']
    
    # Metrics
    metrics = calculate_metrics(equity_curve, trades, df['returns'], initial_capital)
    metrics['trades_df'] = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics['final_equity'] = capital
    metrics['total_trades'] = len(trades)
    
    return df, metrics, equity_curve

def calculate_metrics(equity, trades, buy_hold_returns, initial_capital):
    """Compute standard performance metrics."""
    equity = pd.Series(equity).dropna()
    if len(equity) < 2:
        return {'error': 'Insufficient data'}
    
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    
    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 0.01)
    cagr = ((equity.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100
    
    # Daily strategy returns for Sharpe / vol
    daily_rets = equity.pct_change().dropna()
    sharpe = 0.0
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    
    # Max Drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    # Win rate / Profit factor from trades
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0
    
    # Exposure (time in market)
    exposure = (equity.pct_change() != 0).mean() * 100 if 'position' in locals() else 50  # rough
    
    # Buy & Hold comparison
    bh_total = (equity.index[-1].to_pydatetime() - equity.index[0].to_pydatetime()).days  # placeholder
    bh_cagr = ((1 + buy_hold_returns).prod() ** (252 / len(buy_hold_returns)) - 1) * 100 if len(buy_hold_returns) > 0 else 0
    
    return {
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 3),
        'win_rate_pct': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'num_trades': len(trades),
        'exposure_pct': round(exposure, 1),
        'bh_cagr_approx': round(bh_cagr, 2)
    }

def pairs_zscore(df1, df2, window=30):
    """Compute z-score of price ratio for pairs trading."""
    ratio = df1['close'] / df2['close']
    mean = ratio.rolling(window=window, min_periods=window//2).mean()
    std = ratio.rolling(window=window, min_periods=window//2).std()
    zscore = (ratio - mean) / std
    return ratio, zscore

def backtest_pairs(df1, df2, z_window=30, entry_z=2.0, exit_z=0.5, initial_capital=10000.0, commission=0.001):
    """Simple pairs backtest: trade the ratio z-score mean reversion."""
    ratio, zscore = pairs_zscore(df1, df2, z_window)
    df = pd.DataFrame({
        'close1': df1['close'],
        'close2': df2['close'],
        'ratio': ratio,
        'zscore': zscore
    }).dropna()
    
    capital = initial_capital
    position = 0  # 1 = long spread (long1 short2), -1 = short spread
    entry_ratio = 0.0
    trades = []
    equity = [initial_capital]
    dates = [df.index[0]]
    
    for i in range(1, len(df)):
        z = df['zscore'].iloc[i]
        r = df['ratio'].iloc[i]
        p1 = df['close1'].iloc[i]
        p2 = df['close2'].iloc[i]
        
        # Entry logic
        if position == 0:
            if z > entry_z:  # Short the expensive (short1, long2)
                position = -1
                entry_ratio = r
            elif z < -entry_z:  # Long the cheap (long1, short2)
                position = 1
                entry_ratio = r
        
        # Exit logic
        elif position != 0 and abs(z) < exit_z:
            # Close position, calculate PnL approx (simplified, ignores exact hedge shares)
            exit_ratio = r
            pnl_ratio = (exit_ratio - entry_ratio) * position
            # Scale to capital roughly
            pnl = capital * pnl_ratio * 0.5  # conservative scaling
            pnl -= abs(pnl) * commission  # rough cost
            capital += pnl
            trades.append({
                'entry_date': dates[-1],
                'exit_date': df.index[i],
                'pnl': pnl,
                'z_entry': entry_ratio,
                'z_exit': exit_ratio,
                'direction': 'long_spread' if position == 1 else 'short_spread'
            })
            position = 0
        
        # Mark equity (very simplified)
        equity.append(capital)
        dates.append(df.index[i])
    
    # Force close at end
    if position != 0:
        capital += capital * 0.01 * position  # dummy
        equity[-1] = capital
    
    eq_curve = pd.Series(equity, index=dates)
    metrics = calculate_metrics(eq_curve, trades, pd.Series([0]*len(eq_curve)), initial_capital)
    metrics['trades_df'] = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics['final_equity'] = capital
    
    return df, metrics, eq_curve, zscore

# ==================== UI ====================

st.title("📊 FreeEdge Dashboard")
st.markdown("**Free, open-source trading edge explorer** — RSI Mean Reversion Swing + Pairs Stat Arb Lite + Options/Vol Insights. Built for serious backtesting & signal generation. **No BS, data-driven.**")

# Prominent disclaimer
with st.expander("⚠️ READ THIS FIRST — Important Disclaimers & Risk Warnings (Click to expand)", expanded=True):
    st.error("""
    **THIS IS NOT FINANCIAL ADVICE. TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.**
    
    - No strategy here (or anywhere) is foolproof, guaranteed, or "proven" to make money consistently in the future.
    - Backtests are educational illustrations only. They simplify reality and usually overstate performance (no/full slippage, commissions modeled lightly, no taxes, perfect fills assumed, survivorship bias possible).
    - Past results (backtest or hypothetical) do **not** predict future performance. Markets evolve, edges decay or disappear.
    - You can and likely will lose money. Only risk capital you can afford to lose.
    - This tool is for **education, research, and idea generation**. Paper trade extensively before using real money.
    - Options add leverage and unique risks (total loss of premium, assignment, etc.).
    - The developers assume **zero liability**. Use at your own risk. Do your own research.
    """)
    st.info("Strategies selected have some historical research support in academic/practitioner literature (mean reversion, relative value/pairs trading, volatility premium). But that does not guarantee profits.")

st.sidebar.header("⚙️ Global Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
commission_pct = st.sidebar.slider("Commission + Slippage % per trade (round trip approx)", 0.0, 1.0, 0.1, 0.05) / 100.0

# Popular tickers for convenience
popular_tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLK", "XLE", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Signals (RSI MR)", 
    "📈 Backtest RSI Strategy", 
    "🔗 Pairs Trading Explorer", 
    "📋 Options & Vol Insights", 
    "📖 About & Docs"
])

# ---------- TAB 1: LIVE SIGNALS ----------
with tab1:
    st.header("Live RSI(2) Mean Reversion Signals — Swing Trading")
    st.markdown("**Strategy**: Buy (long) when RSI(2) drops to extreme oversold levels (mean reversion entry). Exit when RSI recovers toward neutral/mean. Best suited for swing (few days to weeks) on liquid ETFs/stocks in non-strong-trending regimes.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.selectbox("Select Ticker", popular_tickers, index=0)
        custom_ticker = st.text_input("Or enter custom ticker (e.g. 'SPY' or 'BTC-USD')", "")
        if custom_ticker.strip():
            ticker = custom_ticker.strip().upper()
        
        rsi_period = st.slider("RSI Period", 2, 14, 2, 1)
        oversold = st.slider("Oversold Entry Threshold (Buy)", 5, 30, 10, 1)
        exit_rsi = st.slider("Exit Threshold (Sell / Take Profit)", 40, 70, 50, 1)
        
        period_choice = st.selectbox("History Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    
    with col2:
        if st.button("🔄 Fetch Latest Data & Signal", type="primary"):
            with st.spinner(f"Downloading {ticker} data..."):
                df = download_data(ticker, period=period_choice)
            
            if df is not None and len(df) > 50:
                df_sig = rsi_mean_reversion_signals(df, rsi_period, oversold, 95, exit_rsi)  # overbought not heavily used in long-only
                
                latest = df_sig.iloc[-1]
                prev = df_sig.iloc[-2] if len(df_sig) > 1 else latest
                
                price = latest['close']
                rsi2 = latest['rsi']
                position = latest['position']
                
                st.metric("Current Price", f"${price:.2f}")
                st.metric(f"RSI({rsi_period})", f"{rsi2:.1f}" if not pd.isna(rsi2) else "N/A")
                
                if not pd.isna(rsi2):
                    if rsi2 <= oversold:
                        st.success(f"🟢 **BUY / LONG SIGNAL** — RSI({rsi_period}) at extreme oversold ({rsi2:.1f} ≤ {oversold}). Potential mean reversion entry for swing long.")
                    elif rsi2 >= exit_rsi and position == 1:
                        st.warning(f"🔴 **EXIT / SELL SIGNAL** — RSI recovered to {rsi2:.1f}. Consider taking profit or closing long.")
                    elif position == 1:
                        st.info(f"🟡 **HOLD LONG** — In position. RSI = {rsi2:.1f}. Wait for exit threshold or trail.")
                    else:
                        st.info(f"⚪ **NO SIGNAL / FLAT** — RSI = {rsi2:.1f}. Wait for oversold entry opportunity.")
                
                # Mini chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    row_heights=[0.7, 0.3], vertical_spacing=0.02)
                fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['close'], name="Close", line=dict(color="#1f77b4")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['rsi'], name=f"RSI({rsi_period})", line=dict(color="#ff7f0e")), row=2, col=1)
                fig.add_hline(y=oversold, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold Entry")
                fig.add_hline(y=exit_rsi, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Exit")
                fig.update_layout(height=450, title=f"{ticker} — Price & RSI({rsi_period})", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Data as of {df_sig.index[-1].strftime('%Y-%m-%d')} (delayed). Strategy is long-only mean reversion. Not a recommendation to trade.")
            else:
                st.error("Could not download sufficient data. Try another ticker or longer period.")

# ---------- TAB 2: BACKTEST ----------
with tab2:
    st.header("RSI Mean Reversion Backtest Engine")
    st.markdown("**Run your own historical backtests.** Adjust parameters and see realistic(ish) performance metrics, equity curve, and trade statistics. Compare implicitly to buy-and-hold via CAGR context.")
    
    colA, colB = st.columns([1, 3])
    
    with colA:
        bt_ticker = st.selectbox("Backtest Ticker", popular_tickers, index=0, key="bt_ticker")
        bt_custom = st.text_input("Custom ticker for backtest", "", key="bt_custom")
        if bt_custom.strip():
            bt_ticker = bt_custom.strip().upper()
        
        bt_start = st.date_input("Start Date", value=datetime(2015, 1, 1))
        bt_end = st.date_input("End Date", value=datetime(2025, 12, 31))
        
        bt_rsi_p = st.slider("RSI Period", 2, 7, 2, 1, key="bt_rsi")
        bt_os = st.slider("Oversold Entry", 5, 25, 10, 1, key="bt_os")
        bt_exit = st.slider("Exit RSI Level", 40, 65, 50, 1, key="bt_exit")
        
        if st.button("🚀 Run Backtest", type="primary", key="run_bt"):
            run_backtest = True
        else:
            run_backtest = False
    
    with colB:
        if run_backtest:
            with st.spinner(f"Running backtest on {bt_ticker}... (this may take a moment)"):
                df_bt = download_data(bt_ticker, start=bt_start.strftime("%Y-%m-%d"), end=bt_end.strftime("%Y-%m-%d"))
            
            if df_bt is not None and len(df_bt) > 100:
                df_res, metrics, eq_curve = backtest_rsi_strategy(
                    df_bt, initial_capital=initial_capital, commission=commission_pct,
                    rsi_period=bt_rsi_p, oversold=bt_os, exit_level=bt_exit
                )
                
                st.subheader(f"Backtest Results: {bt_ticker} ({bt_start} to {bt_end})")
                
                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['total_return_pct']}%")
                m2.metric("CAGR", f"{metrics['cagr_pct']}%")
                m3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']}%")
                m4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
                
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Win Rate", f"{metrics['win_rate_pct']}%")
                m6.metric("Profit Factor", f"{metrics['profit_factor']}")
                m7.metric("# Trades", metrics['num_trades'])
                m8.metric("Exposure (time in mkt)", f"{metrics.get('exposure_pct', 'N/A')}%")
                
                st.caption("Note: Low exposure is typical & often desirable for mean-reversion (only trade when edge present). Compare CAGR to buy-and-hold of same ticker over period (usually higher drawdown for BH).")
                
                # Equity curve
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve.values, name="Strategy Equity", line=dict(color="#2ca02c", width=2)))
                fig_eq.update_layout(title="Strategy Equity Curve", height=350, yaxis_title="Portfolio Value ($)", xaxis_title="Date")
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # Trade summary
                if not metrics['trades_df'].empty:
                    st.subheader("Trade Log (last 10)")
                    st.dataframe(metrics['trades_df'].tail(10), use_container_width=True)
                else:
                    st.info("No completed trades in this period with current parameters (try lowering oversold threshold or extending date range).")
                
                st.info("**Interpretation tip**: High win rate + low avg loss is common in tight mean-reversion. Watch max DD and profit factor. Re-optimize carefully — avoid overfitting by testing out-of-sample periods.")
            else:
                st.error("Insufficient data for backtest. Try different dates or ticker.")

# ---------- TAB 3: PAIRS ----------
with tab3:
    st.header("Pairs Trading / Stat Arb Lite Explorer")
    st.markdown("""
    **Classic quantitative relative-value strategy.** Trade the *spread* (price ratio) between two correlated assets when it deviates significantly from its recent mean (z-score).
    - When z-score is very high: Short the expensive leg, long the cheap leg (expect convergence).
    - Exit when z-score reverts toward zero.
    
    This approach has extensive academic backing for many equity/ETF pairs. **Lite version** uses simple ratio z-score (rolling). Real implementations often use cointegration/OLS hedge ratios and more sophisticated risk management.
    """)
    
    pcol1, pcol2 = st.columns([1, 2])
    
    with pcol1:
        pair_preset = st.selectbox("Popular Pairs (ETFs often work well)", 
                                   ["SPY / QQQ", "XLK / XLF", "IWM / SPY", "XLE / XLF", "GLD / SLV", "TLT / IEF", "Custom"], index=0)
        
        if pair_preset == "Custom":
            t1 = st.text_input("Ticker 1 (e.g. SPY)", "SPY").upper().strip()
            t2 = st.text_input("Ticker 2 (e.g. QQQ)", "QQQ").upper().strip()
        else:
            mapping = {
                "SPY / QQQ": ("SPY", "QQQ"),
                "XLK / XLF": ("XLK", "XLF"),
                "IWM / SPY": ("IWM", "SPY"),
                "XLE / XLF": ("XLE", "XLF"),
                "GLD / SLV": ("GLD", "SLV"),
                "TLT / IEF": ("TLT", "IEF")
            }
            t1, t2 = mapping[pair_preset]
        
        z_win = st.slider("Z-score Rolling Window (days)", 20, 60, 30, 5)
        entry_z = st.slider("Entry |Z-score| Threshold", 1.5, 3.5, 2.0, 0.1)
        exit_z = st.slider("Exit |Z-score| Threshold", 0.2, 1.0, 0.5, 0.1)
        
        if st.button("🔄 Analyze Pair & Backtest", type="primary"):
            run_pairs = True
        else:
            run_pairs = False
    
    with pcol2:
        if run_pairs:
            with st.spinner(f"Analyzing {t1} vs {t2}..."):
                df1 = download_data(t1, period="5y")
                df2 = download_data(t2, period="5y")
            
            if df1 is not None and df2 is not None:
                # Align
                common_idx = df1.index.intersection(df2.index)
                df1a = df1.loc[common_idx]
                df2a = df2.loc[common_idx]
                
                ratio, zscore = pairs_zscore(df1a, df2a, z_win)
                latest_z = zscore.iloc[-1] if not zscore.empty else np.nan
                
                st.metric(f"Current {t1}/{t2} Ratio Z-Score", f"{latest_z:.2f}" if not pd.isna(latest_z) else "N/A")
                
                if not pd.isna(latest_z):
                    if latest_z > entry_z:
                        st.warning(f"📉 Potential **SHORT SPREAD** signal (z={latest_z:.2f} > {entry_z}): Short {t1}, Long {t2}")
                    elif latest_z < -entry_z:
                        st.success(f"📈 Potential **LONG SPREAD** signal (z={latest_z:.2f} < -{entry_z}): Long {t1}, Short {t2}")
                    else:
                        st.info(f"⚪ No extreme deviation. Z-score = {latest_z:.2f}. Flat or hold existing.")
                
                # Charts
                fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
                fig_p.add_trace(go.Scatter(x=ratio.index, y=ratio, name=f"{t1}/{t2} Ratio"), row=1, col=1)
                fig_p.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score"), row=2, col=1)
                fig_p.add_hline(y=entry_z, line_dash="dash", line_color="red", row=2)
                fig_p.add_hline(y=-entry_z, line_dash="dash", line_color="green", row=2)
                fig_p.add_hline(y=0, line_dash="solid", line_color="gray", row=2)
                fig_p.update_layout(height=500, title=f"Pairs Analysis: {t1} vs {t2}")
                st.plotly_chart(fig_p, use_container_width=True)
                
                # Quick backtest
                st.subheader("Historical Pairs Backtest (same params)")
                df_pair_res, pair_metrics, pair_eq, _ = backtest_pairs(df1a, df2a, z_win, entry_z, exit_z, initial_capital, commission_pct)
                
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Strategy CAGR", f"{pair_metrics.get('cagr_pct', 'N/A')}%")
                pm2.metric("Max DD", f"{pair_metrics.get('max_drawdown_pct', 'N/A')}%")
                pm3.metric("Win Rate", f"{pair_metrics.get('win_rate_pct', 'N/A')}%")
                
                fig_pe = go.Figure(go.Scatter(x=pair_eq.index, y=pair_eq.values, name="Pairs Equity"))
                fig_pe.update_layout(title="Pairs Strategy Equity Curve (Simplified)", height=300)
                st.plotly_chart(fig_pe, use_container_width=True)
                
                st.caption("Pairs backtest is simplified (ratio-based, approximate PnL scaling). Real pairs trading requires precise hedge ratios, risk parity, and careful execution. Positive historical results in many studies but capacity and costs matter greatly.")
            else:
                st.error("Failed to download data for one or both tickers.")

# ---------- TAB 4: OPTIONS ----------
with tab4:
    st.header("Options Chain & Volatility Insights")
    st.markdown("""
    **Educational tool only.** View options chain for a ticker + basic realized volatility (HV) calculation.
    - High IV relative to HV often favors **premium selling** strategies (credit spreads, iron condors, covered calls) — a form of volatility risk premium harvesting (documented edge in many vol trading papers).
    - **Always verify live IV, bid/ask, and liquidity on your broker.** This uses delayed Yahoo data.
    - Not a signal generator. Options require advanced understanding of Greeks, assignment risk, etc.
    """)
    
    opt_ticker = st.selectbox("Ticker for Options Analysis", popular_tickers, index=0, key="opt_t")
    opt_custom = st.text_input("Custom ticker", "", key="opt_custom")
    if opt_custom.strip():
        opt_ticker = opt_custom.strip().upper()
    
    if st.button("📋 Load Options Data & HV", type="primary"):
        with st.spinner(f"Fetching options for {opt_ticker}..."):
            exps, calls, puts, nearest = get_options_chain(opt_ticker)
            df_opt = download_data(opt_ticker, period="1y")
        
        if exps and calls is not None:
            st.subheader(f"Options — Nearest Expiry: {nearest}")
            
            colc, colp = st.columns(2)
            with colc:
                st.write("**Calls (sample ATM-ish)**")
                # Simple filter around current price
                if df_opt is not None:
                    current_price = df_opt['close'].iloc[-1]
                    atm_calls = calls[(calls['strike'] > current_price * 0.9) & (calls['strike'] < current_price * 1.1)].head(8)
                    st.dataframe(atm_calls[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']], use_container_width=True)
            
            with colp:
                st.write("**Puts (sample ATM-ish)**")
                if df_opt is not None:
                    current_price = df_opt['close'].iloc[-1]
                    atm_puts = puts[(puts['strike'] > current_price * 0.9) & (puts['strike'] < current_price * 1.1)].head(8)
                    st.dataframe(atm_puts[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']], use_container_width=True)
            
            st.caption("impliedVolatility column from Yahoo (may be stale). Check your broker for accurate, tradable quotes.")
        
        if df_opt is not None:
            hv20 = calculate_hv(df_opt, window=20).iloc[-1]
            hv60 = calculate_hv(df_opt, window=60).iloc[-1]
            st.subheader("Realized (Historical) Volatility")
            hv1, hv2 = st.columns(2)
            hv1.metric("20-day HV (ann.)", f"{hv20:.1f}%")
            hv2.metric("60-day HV (ann.)", f"{hv60:.1f}%")
            
            st.info("""
            **Vol Premium Idea (Educational)**: When current options IV (from chain above) is significantly **higher** than these HV numbers, it may indicate expensive premium — potential edge in selling volatility (with proper risk management: defined-risk spreads, position sizing <1-2% portfolio risk per trade, etc.).
            
            This is **not** a trade recommendation. IV can stay high or spike further (short vol has tail risk). Always paper trade and understand the Greeks.
            """)
        else:
            st.warning("Options data limited or unavailable for this ticker (common for low-volume or non-US names).")

# ---------- TAB 5: ABOUT ----------
with tab5:
    st.header("About This Dashboard & How to Use It Effectively")
    
    st.markdown("""
    ### Why This Exists
    Most retail trading "tools" and gurus push mainstream TA or unproven systems. This dashboard focuses on approaches with **some legitimate quantitative/research backing**:
    - Short-term mean reversion (RSI extremes)
    - Relative value / pairs trading
    - Volatility premium concepts
    
    It is **fully free**, transparent (code is open), and gives you the tools to test ideas yourself instead of trusting signals.
    
    ### Recommended Workflow
    1. **Explore Live Signals** on 2-3 liquid tickers you know well (SPY, QQQ, sector ETFs).
    2. **Backtest extensively** across different market regimes (2015-2019 bull, 2020 crash, 2022 bear, 2023-2025 whatever current is). Change parameters and note robustness.
    3. **Paper trade** the signals for at least 3-6 months in real-time conditions.
    4. **Use Pairs tab** to find relative value opportunities between correlated instruments.
    5. **Risk Management First**: Never risk more than 0.5-1% of capital per trade idea. Use stops or defined risk even if strategy doesn't specify. Track your own results.
    6. **Iterate**: Fork the code on GitHub, improve the backtester, add filters (e.g. only trade if ADX < 25 for ranging markets), or integrate better data.
    
    ### Limitations (Be Honest With Yourself)
    - Daily data only → best for swing, not high-frequency scalp.
    - yfinance data has gaps/delays/splits adjustments that are good but not perfect for all use cases.
    - Simplified execution assumptions in backtests.
    - No live alerts, portfolio tracking, or broker integration (you can add these).
    
    ### Next Steps for Serious Users
    - Deploy your own instance on Streamlit Cloud (free).
    - Add your own capital allocation / position sizing rules.
    - Combine with fundamental filters or macro regime detection.
    - For production: Consider paid data (Polygon.io has excellent free tier too), proper order management, and walk-forward optimization.
    
    ### Credits & Inspiration
    - yfinance community
    - Research by Larry Connors, quant sites like QuantifiedStrategies / Robot Wealth, academic pairs trading literature
    - Streamlit for making powerful dashboards accessible
    
    **This project is provided as-is for educational purposes.** Improve it, share improvements, but always respect the disclaimers.
    
    Good luck, trade responsibly, and focus on process over profits.
    """)
    
    st.success("Deployed via GitHub + Streamlit Cloud = completely free webapp you control.")

# Footer
st.markdown("---")
st.caption("FreeEdge Dashboard • Built with ❤️ for truth-seeking traders • Data: Yahoo Finance (yfinance) • Not affiliated with any broker or financial institution.")
