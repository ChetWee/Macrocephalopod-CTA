import investpy
import pandas as pd
import numpy as np
import streamlit as st
import time

# -------------------- Universe --------------------
UNIVERSE = {
    # Equities
    "SP500": "S&P 500 Futures",
    "NASDAQ": "NASDAQ 100 Futures",
    "EURO50": "EURO 50 Futures",
    "A50": "China A50 Futures",
    "HSI": "Hang Seng Futures",
    "NIKKEI": "Nikkei 225 Futures",
    "ASX200": "S&P/ASX 200 Futures",
    "KOSPI": "KOSPI 200 Futures",
    "NIFTY": "Nifty 50 Futures",
    "STI": "FTSE Straits Times Singapore",
    # Currencies
    "DXY": "US Dollar Index Futures",
    "EURUSD": "Euro US Dollar",
    "GBPUSD": "British Pound Futures",
    "AUDUSD": "Australian Dollar Futures",
    "NZDUSD": "New Zealand Dollar Futures",
    "CADUSD": "Canadian Dollar Futures",
    "USDJPY": "US Dollar Japanese Yen",
    "CHFUSD": "Swiss Franc Futures",
    # Asia FX
    "USDCNH": "US Dollar Chinese Yuan Offshore",
    "USDKRW": "US Dollar Korean Won",
    "USDTWD": "USDTWD",
    "USDSGD": "USDSGD",
    "USDMYR": "USDMYR",
    "USDTHB": "USDTHB",
    # Commodities
    "BTC": "Bitcoin Futures CME",
    "GOLD": "Gold Futures",
    "SILVER": "Silver Futures",
    "COPPER": "Copper Futures",
    "BRENT": "Brent Oil Futures",
    "NATGAS": "Natural Gas Futures",
    # Rates
    "US2Y": "US 2 Year T-Note Futures",
    "US5Y": "US 5 Year T-Note Futures",
    "US10Y": "US 10 Year T-Note Futures",
    "US30Y": "US 30 Year T-Bond Futures",
    "BUND": "Euro Bund Futures",
    "BTP": "Short-Term Euro-BTP Futures",
    "OAT": "Euro OAT Futures",
    "GILT": "UK Gilt Futures",
    "JGB": "Japan Government Bond Futures",
}

# Asset Classes
ASSET_CLASSES = {
    "Equities": ["SP500", "NASDAQ", "EURO50", "A50", "HSI", "NIKKEI", "ASX200", "KOSPI", "NIFTY", "STI"],
    "G7 Currencies": ["DXY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "CADUSD", "USDJPY", "CHFUSD"],
    "Asian Currencies": ["USDCNH", "USDKRW", "USDTWD", "USDSGD", "USDMYR", "USDTHB"],
    "Commodities": ["BTC", "GOLD", "SILVER", "COPPER", "BRENT", "NATGAS"],
    "Rates": ["US2Y", "US5Y", "US10Y", "US30Y", "BUND", "BTP", "OAT", "GILT", "JGB"]
}

FUTURES_ASSETS = list(UNIVERSE.keys())
START_DATE = "01/01/2023"
END_DATE = pd.Timestamp.today().strftime("%d/%m/%Y")

# -------------------- Streamlit Config (MUST BE FIRST) --------------------
st.set_page_config(page_title="CTA Trend Dashboard", layout="wide")

# -------------------- Cached Data Retrieval --------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_price_data():
    """Fetch all price data - cached to avoid re-fetching on every interaction"""
    price_data = {}
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    total_assets = len(UNIVERSE)
    for idx, (asset, search_text) in enumerate(UNIVERSE.items()):
        status_text.text(f"Fetching {asset}... ({idx+1}/{total_assets})")
        success = False
        retries = 0
        max_retries = 5
        while not success and retries < max_retries:
            try:
                search_result = investpy.search_quotes(text=search_text, n_results=1)
                df = search_result.retrieve_historical_data(from_date=START_DATE, to_date=END_DATE)
                price_data[asset] = df["Close"]
                success = True
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    st.sidebar.warning(f"Failed to retrieve {asset} after {max_retries} attempts")
                time.sleep(2)
        
        progress_bar.progress((idx + 1) / total_assets)
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(price_data).dropna(how="all")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_signals(price_df):
    """Calculate all signals - cached to avoid recalculating on every interaction"""
    # Invert USD-based currencies
    price_df = price_df.copy()
    
    for curr in usd_based_currencies:
        if curr in price_df.columns:
            price_df[curr] = 1 / price_df[curr]
    
    def map_signal(x: pd.Series, k: float = 1.0) -> pd.Series:
        return np.tanh(k * x)

    signal_cols = []
    for asset in price_df.columns:
        px = price_df[asset].dropna().astype(float).sort_index()
        log_px = np.log(px)
        ret = px.pct_change()

        # SHORT TERM: 5 / 21, vol=21
        vol_s = ret.rolling(21).std()
        raw_s = log_px.rolling(5).mean() - log_px.rolling(21).mean()
        sig_s = map_signal(raw_s / vol_s).rename(f"{asset}_short")

        # MEDIUM TERM: 21 / 63, vol=63
        vol_m = ret.rolling(63).std()
        raw_m = log_px.rolling(21).mean() - log_px.rolling(63).mean()
        sig_m = map_signal(raw_m / vol_m).rename(f"{asset}_medium")

        # LONG TERM: 50 / 252, vol=252
        vol_l = ret.rolling(252).std()
        raw_l = log_px.rolling(50).mean() - log_px.rolling(252).mean()
        sig_l = map_signal(raw_l / vol_l).rename(f"{asset}_long")

        signal_cols.extend([sig_s, sig_m, sig_l])

    return price_df, pd.concat(signal_cols, axis=1).dropna(how="all")

# Load data once
price_df, signal_df = calculate_signals(fetch_price_data())

# -------------------- Streamlit UI --------------------
st.title("CTA Trend Following Dashboard")

# Sidebar
asset_class = st.sidebar.selectbox("Asset Class", list(ASSET_CLASSES.keys()))

horizon = st.sidebar.radio("Signal Horizon", [
    "Short Term Model (5,21)",
    "Medium Term Model (21,63)", 
    "Long Term Model (50,252)"
])
# Extract the horizon key
horizon_map = {
    "Short Term Model (5,21)": "short",
    "Medium Term Model (21,63)": "medium",
    "Long Term Model (50,252)": "long"
}
horizon_key = horizon_map[horizon]

# Filter assets to only show those in selected asset class
assets_in_class = sorted(ASSET_CLASSES[asset_class])
asset = st.sidebar.selectbox("Chart Asset", assets_in_class)

def format_asset(a):
    # Asset display name mapping
    display_names = {
        # Equities
        "SP500": "S&P 500 Futures",
        "NASDAQ": "NASDAQ 100 Futures",
        "EURO50": "Euro Stoxx 50 Futures",
        "A50": "A50 Futures",
        "HSI": "HSI Futures",
        "NIKKEI": "Nikkei 225 Futures",
        "ASX200": "ASX 200 Futures",
        "KOSPI": "KOSPI 200 Futures",
        "NIFTY": "Nifty 50 Futures",
        "STI": "STI",
        # G7 Currencies
        "DXY": "DXY Futures",
        "EURUSD": "EUR",
        "GBPUSD": "GBP Futures",
        "AUDUSD": "AUD Futures",
        "NZDUSD": "NZD Futures",
        "CADUSD": "CAD Futures",
        "USDJPY": "JPY Futures",
        "CHFUSD": "CHF Futures",
        # Asian Currencies
        "USDCNH": "CNH",
        "USDKRW": "KRW",
        "USDTWD": "TWD",
        "USDSGD": "SGD",
        "USDMYR": "MYR",
        "USDTHB": "THB",
        # Commodities
        "BTC": "BTC Futures",
        "GOLD": "Gold Futures",
        "SILVER": "Silver Futures",
        "COPPER": "Copper Futures",
        "BRENT": "Brent Oil Futures",
        "NATGAS": "Natural Gas Futures",
        # Rates
        "US2Y": "US 2 Year T-Note Futures",
        "US5Y": "US 5 Year T-Note Futures",
        "US10Y": "US 10 Year T-Note Futures",
        "US30Y": "US 30 Year T-Bond Futures",
        "BUND": "German Bund Futures",
        "BTP": "Italian BTP Futures",
        "OAT": "French OAT Futures",
        "GILT": "UK Gilt Futures",
        "JGB": "JGBs Futures"
    }
    
    return display_names.get(a, a)

def gradient(val, vmin, vmax):
    if pd.isna(val):
        return ""
    x = (val - vmin) / (vmax - vmin + 1e-9)
    r = int(255 * (1 - x))
    g = int(255 * x)
    return f"background-color: rgb({r},{g},0)"

# ---------- Build Ranking Table ----------
lags = [5, 10, 21]  # 5,10,21 days ago
rows = []

# Filter assets by selected asset class
selected_assets = ASSET_CLASSES[asset_class]

for col in signal_df.filter(like=f"_{horizon_key}").columns:
    asset_name = col.split("_")[0]
    
    # Skip if not in selected asset class
    if asset_name not in selected_assets:
        continue
    
    sig_series = signal_df[col].dropna()
    if len(sig_series) < max(lags):
        continue

    # Get last updated date and price
    last_date = price_df[asset_name].dropna().index[-1].strftime('%Y-%m-%d')
    last_price = price_df[asset_name].dropna().iloc[-1]
    
    # Determine decimal places based on asset class
    if asset_class in ["G7 Currencies", "Asian Currencies"]:
        price_decimals = 5
    else:
        price_decimals = 2
    
    # Get current signal
    current_signal = round(sig_series.iloc[-1]*100,2)

    row = {
        "Asset": format_asset(asset_name), 
        "Last Updated": last_date,
        "Last Price": round(last_price, price_decimals),
        "Signal": current_signal
    }
    for lag in lags:
        row[f"Signal-{lag}d"] = round(sig_series.iloc[-lag]*100,2)
    rows.append(row)

ranked_df = pd.DataFrame(rows)
ranked_df = ranked_df.sort_values("Signal-21d", ascending=False).reset_index(drop=True)

# ---------- Styling ----------
sig_cols = ["Signal"] + [f"Signal-{lag}d" for lag in lags]
vmin = ranked_df[sig_cols].min().min()
vmax = ranked_df[sig_cols].max().max()

styled = ranked_df.style.map(
    lambda v: gradient(v, vmin, vmax),
    subset=sig_cols
)

st.subheader(f"{asset_class} - {horizon} CTA Signal Ranking")
st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------- Asset Chart ----------
col_name = f"{asset}_{horizon_key}"
st.subheader(f"{format_asset(asset)} — {horizon} Signal (Last 63 Days)")
# Multiply by 100 for chart display
chart_data = signal_df[[col_name]].dropna().tail(63) * 100

# Use Plotly for better date formatting control
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=chart_data.index,
    y=chart_data[col_name],
    mode='lines',
    name='Signal',
    line=dict(color='#1f77b4', width=2)
))

fig.update_xaxes(
    tickformat='%d %b',  # Format as "17 Nov"
    tickangle=0
)

fig.update_layout(
    showlegend=False,
    xaxis_title='',
    yaxis_title='Signal',
    height=400,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)