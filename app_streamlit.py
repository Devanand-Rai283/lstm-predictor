# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

from ml_bilstm import create_windows_multifeature, train_bilstm, load_trained_model

st.set_page_config(page_title="Bi-LSTM + Indicators Predictor", page_icon="📈", layout="wide")
st.title("📈 Bi-LSTM Stock Predictor with Technical Indicators")
st.markdown("Uses EMA(20,50), RSI(14), MACD as features and displays indicator charts.")

# --------------------
# Sidebar config
# --------------------
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
st.sidebar.markdown("### Date Range")
start_date = st.sidebar.date_input("Start Date:", value=datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date:", value=datetime.now())

st.sidebar.markdown("### Model Parameters")
window_size = st.sidebar.slider("Window Size (days):", 5, 60, 20)
train_split = st.sidebar.slider("Training Split (%):", 50, 90, 80)

st.sidebar.markdown("### Training Options")
train_epochs = st.sidebar.slider("Epochs:", 5, 200, 50)
batch_size = st.sidebar.selectbox("Batch Size:", [8, 16, 32, 64], index=2)

st.sidebar.markdown("### Persistence")
auto_save = st.sidebar.checkbox("Save trained model to disk", value=True)
load_existing = st.sidebar.checkbox("Try load existing model (if found)", value=True)
MODEL_DIR = "models"
model_path = f"{MODEL_DIR}/{stock_symbol}_bilstm.h5"

# --------------------
# Indicator functions
# --------------------
def compute_ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


# --------------------
# Data fetch & prepare
# --------------------
def fetch_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Flatten MultiIndex columns (e.g. ('Close','AAPL') → 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        st.write("🔍 DEBUG: yfinance raw output")
        st.write(df.head())
        st.write("Columns:", df.columns.tolist())
        st.write("Index:", df.index[:5])
        st.write("Types:", df.dtypes)

        if df is None or df.empty:
            return None

        # Convert index
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

        if "Close" not in df.columns:
            st.write("❌ 'Close' column missing")
            return None

        # Convert Close to numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        return df[["Close"]].dropna()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --------------------
# FIXED build_features()
# --------------------
def build_features(df: pd.DataFrame):
    df = df.copy()

    # Ensure Close is float
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Fix datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[df.index.notna()]

    df['Date'] = df.index

    # Compute indicators
    df['EMA20'] = compute_ema(df['Close'], 20)
    df['EMA50'] = compute_ema(df['Close'], 50)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    macd, signal, _ = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal

    # Drop rows where indicators could not compute
    df = df.dropna(subset=['EMA20','EMA50','RSI14','MACD','MACD_Signal'])

    df = df.reset_index(drop=True)
    return df

# --------------------
# Train/Test scaling & windows
# --------------------
def create_train_test_scaled_multifeature(df_features, train_size, window_size):
    values = df_features.iloc[:, 1:].values
    n_rows = values.shape[0]

    scaler = MinMaxScaler()
    scaler.fit(values[:train_size])
    scaled = scaler.transform(values)

    X_train, y_train = create_windows_multifeature(scaled[:train_size], window_size)

    if train_size < n_rows:
        combined = np.vstack([scaled[train_size - window_size:train_size], scaled[train_size:]])
    else:
        combined = scaled[train_size - window_size:train_size]

    X_test, y_test = create_windows_multifeature(combined, window_size)

    return scaler, X_train, y_train, X_test, y_test


# --------------------
# Metrics
# --------------------
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# --------------------
# Main action button
# --------------------
if st.sidebar.button("🔄 Load Data & Analyze", use_container_width=True):

    with st.spinner("📥 Fetching data..."):
        df_raw = fetch_stock_data(stock_symbol, start_date, end_date)
        if df_raw is None or len(df_raw) < window_size + 60:
            st.error("Not enough data.")
            st.stop()
        st.success(f"Loaded {len(df_raw)} rows")

    # Build indicators/features
    with st.spinner("⚙️ Computing indicators..."):
        df_feat = build_features(df_raw)
        dates = df_feat['Date']
        feature_cols = ['Close', 'EMA20', 'EMA50', 'RSI14', 'MACD', 'MACD_Signal']
        df_features = df_feat[['Date'] + feature_cols].copy()
        st.success("Indicators computed")

    # Split
    total_len = len(df_features)
    train_size = int(total_len * (train_split / 100))

    with st.spinner("🔧 Preparing training/testing data..."):
        scaler, X_train, y_train, X_test, y_test = create_train_test_scaled_multifeature(
            df_features, train_size, window_size
        )
        n_features = X_train.shape[2]
        st.success(f"Prepared windows: train {len(X_train)}, test {len(X_test)}")

    # Load model if exists
    model = None
    if load_existing:
        model = load_trained_model(model_path)

    # Train if not loaded
    if model is None:
        with st.spinner("🔨 Training model..."):
            model = train_bilstm(X_train, y_train, window_size, n_features,
                                 epochs=train_epochs, batch_size=batch_size,
                                 verbose=1, model_path=model_path if auto_save else None)
            st.success("Training complete")

    # Predict
    with st.spinner("🔮 Predicting..."):
        train_pred_norm = model.predict(X_train, verbose=0)
        test_pred_norm = model.predict(X_test, verbose=0)

        # inverse transform helper
        def inverse_close(arr):
            dummy = np.zeros((len(arr), n_features))
            dummy[:, 0] = arr.flatten()
            return scaler.inverse_transform(dummy)[:, 0].reshape(-1, 1)

        train_pred = inverse_close(train_pred_norm)
        y_train_actual = inverse_close(y_train)
        test_pred = inverse_close(test_pred_norm)
        y_test_actual = inverse_close(y_test)

        st.success("Predictions computed")

    # Save session
    st.session_state.data = {
        'df_features': df_features,
        'train_pred': train_pred,
        'y_train_actual': y_train_actual,
        'test_pred': test_pred,
        'y_test_actual': y_test_actual,
        'train_size': train_size,
        'window_size': window_size
    }
    st.session_state.stock_symbol = stock_symbol


# --------------------
# Display results
# --------------------
if hasattr(st.session_state, 'data'):

    df_features = st.session_state.data['df_features']
    df_dates = pd.to_datetime(df_features['Date'])

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🎯 Performance", "📊 Indicators"])

    # -------------------- PREDICTIONS --------------------
    with tab1:
        st.subheader("Actual vs Predicted")

        train_size = st.session_state.data['train_size']
        window_size = st.session_state.data['window_size']

        actual = df_features['Close'].values
        train_pred = st.session_state.data['train_pred'].flatten()
        test_pred = st.session_state.data['test_pred'].flatten()

        fig = go.Figure()

        # Actual
        fig.add_trace(go.Scatter(
            x=df_dates,
            y=actual,
            mode="lines",
            name="Actual Price",
            line=dict(color="blue", width=3)
        ))

        # Train
        train_dates = df_dates[window_size:train_size]
        fig.add_trace(go.Scatter(
            x=train_dates[:len(train_pred)],
            y=train_pred,
            mode="lines",
            name="Train Predictions",
            line=dict(color="green", width=2, dash="dash")
        ))

        # Test
        test_dates = df_dates[train_size:][:len(test_pred)]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_pred,
            mode="lines",
            name="Test Predictions",
            line=dict(color="red", width=2, dash="dash")
        ))

        fig.update_layout(
            height=600,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            title=f"{st.session_state.stock_symbol} - Actual vs Predicted"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------- PERFORMANCE --------------------
    with tab2:
        st.subheader("Performance Metrics")

        train_rmse = compute_rmse(st.session_state.data['y_train_actual'],
                                  st.session_state.data['train_pred'])
        train_mae = compute_mae(st.session_state.data['y_train_actual'],
                                 st.session_state.data['train_pred'])

        test_rmse = compute_rmse(st.session_state.data['y_test_actual'],
                                 st.session_state.data['test_pred'])
        test_mae = compute_mae(st.session_state.data['y_test_actual'],
                                st.session_state.data['test_pred'])

        st.metric("Train RMSE", f"{train_rmse:.2f}")
        st.metric("Test RMSE", f"{test_rmse:.2f}")
        st.metric("Train MAE", f"{train_mae:.2f}")
        st.metric("Test MAE", f"{test_mae:.2f}")

    # -------------------- INDICATORS --------------------
    with tab3:
        st.subheader("Technical Indicators")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_dates, y=df_features['Close'], name="Close"))
        fig1.add_trace(go.Scatter(x=df_dates, y=df_features['EMA20'], name="EMA20"))
        fig1.add_trace(go.Scatter(x=df_dates, y=df_features['EMA50'], name="EMA50"))
        fig1.update_layout(title="Price + EMA", height=350)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_dates, y=df_features['RSI14'], name="RSI14"))
        fig2.update_layout(title="RSI(14)", height=250, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig2, use_container_width=True)
                # MACD
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_dates, y=df_features['MACD'],
            name="MACD", line=dict(color="black")
        ))
        fig3.add_trace(go.Scatter(
            x=df_dates, y=df_features['MACD_Signal'],
            name="Signal", line=dict(color="red")
        ))
        fig3.update_layout(title="MACD & Signal", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("✅ Indicators shown above are used as input features for the Bi-LSTM model.")

else:
    st.info("👈 Configure parameters and click **Load Data & Analyze** to run the model.")
    with st.expander("ℹ️ Info"):
        st.write("""
        This app:
        - Downloads stock data from Yahoo Finance  
        - Computes EMA20, EMA50, RSI(14), MACD, Signal  
        - Trains a Bi-LSTM deep learning model  
        - Displays predictions & indicators  
        """)
