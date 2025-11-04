import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Plotly for charts
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 LSTM Stock Price Predictor")
st.markdown("*Simple Stock Price Prediction App*")
st.markdown("---")

# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Stock symbol input
stock_symbol = st.sidebar.text_input(
    "Enter Stock Ticker:",
    value="AAPL",
    help="e.g., AAPL, MSFT, GOOGL, TSLA"
).upper()

# Date range
st.sidebar.markdown("### Date Range")
start_date = st.sidebar.date_input(
    "Start Date:",
    value=datetime(2020, 1, 1)  # Changed to more recent data
)
end_date = st.sidebar.date_input(
    "End Date:",
    value=datetime.now()
)

# Model parameters
st.sidebar.markdown("### Model Parameters")
window_size = st.sidebar.slider("Window Size (days):", 1, 10, 3)
train_split = st.sidebar.slider("Training Split (%):", 50, 90, 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = df[['Close']].copy()
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_dataset(df, window_size):
    """Create windowed dataset"""
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[i:(i + window_size), 0])
        y.append(df[i + window_size, 0])
    return np.array(X), np.array(y)

def simple_moving_average_predictor(X_train, y_train, X_test):
    """Simple moving average prediction"""
    avg_price = np.mean(y_train)
    predictions = []
    for x in X_test:
        trend = x[-1] - x[0] if len(x) > 0 else 0
        pred = avg_price + (trend * 0.5)
        predictions.append(pred)
    return np.array(predictions)

def compute_rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Calculate MAE"""
    return mean_absolute_error(y_true, y_pred)

# ============================================================================
# MAIN APP
# ============================================================================

# Train button
if st.sidebar.button("🔄 Load Data & Analyze", use_container_width=True):
    
    # =====================================================================
    # STEP 1: FETCH DATA
    # =====================================================================
    
    with st.spinner("📥 Fetching stock data..."):
        df = fetch_stock_data(stock_symbol, start_date, end_date)
        
        if df is None or len(df) < window_size + 50:
            st.error(f"❌ Not enough data for {stock_symbol}")
            st.stop()
        
        st.success(f"✓ Loaded {len(df)} records")
    
    # =====================================================================
    # STEP 2: PREPARE DATA
    # =====================================================================
    
    with st.spinner("📊 Preparing data..."):
        # Get close prices
        data = df.values
        
        # Split train/test
        train_size = int(len(data) * (train_split / 100))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        scaled_test = scaler.transform(test_data)
        
        # Create windows
        X_train, y_train = create_dataset(scaled_train, window_size)
        X_test, y_test = create_dataset(scaled_test, window_size)
        
        st.success(f"✓ Data prepared: Train={len(X_train)}, Test={len(X_test)}")
    
    # =====================================================================
    # STEP 3: MAKE PREDICTIONS
    # =====================================================================
    
    with st.spinner("🔮 Making predictions..."):
        # Get predictions
        train_predict_norm = simple_moving_average_predictor(X_train, y_train, X_train)
        test_predict_norm = simple_moving_average_predictor(X_train, y_train, X_test)
        
        # Inverse transform to real prices
        train_predict = scaler.inverse_transform(train_predict_norm.reshape(-1, 1))
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        
        test_predict = scaler.inverse_transform(test_predict_norm.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        st.success("✓ Analysis complete!")
    
    # =====================================================================
    # STORE IN SESSION
    # =====================================================================
    
    st.session_state.data = {
        'df': df,
        'train_predict': train_predict,
        'y_train_actual': y_train_actual,
        'test_predict': test_predict,
        'y_test_actual': y_test_actual,
        'train_size': train_size,
        'window_size': window_size,
        'X_train': X_train,
        'X_test': X_test
    }
    st.session_state.stock_symbol = stock_symbol

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if hasattr(st.session_state, 'data'):
    
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stock", st.session_state.stock_symbol)
    with col2:
        st.metric("Data Points", len(st.session_state.data['df']))
    with col3:
        price_range = st.session_state.data['df']['Close'].values
        min_price = float(price_range.min())
        max_price = float(price_range.max())
        st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
    with col4:
        current_price = st.session_state.data['df']['Close'].values[-1]
        st.metric("Current Price", f"${current_price.item():.2f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🎯 Performance", "📋 Info"])
    
    # =====================================================================
    # TAB 1: PREDICTIONS (FIXED!)
    # =====================================================================
    
    with tab1:
        st.subheader("Price Predictions: Actual vs Predicted")
        
        # Get data
        df_dates = st.session_state.data['df'].index
        actual_prices = st.session_state.data['df']['Close'].values
        
        train_size = st.session_state.data['train_size']
        window_size = st.session_state.data['window_size']
        
        train_predict = st.session_state.data['train_predict'].flatten()
        test_predict = st.session_state.data['test_predict'].flatten()
        
        # Create figure - SIMPLE AND CLEAN
        fig = go.Figure()
        
        # 1. Add ACTUAL PRICE line (BLUE)
        fig.add_trace(go.Scatter(
            x=df_dates,
            y=actual_prices,
            name='Actual Price',
            mode='lines',
            line=dict(
                color='blue',
                width=3
            ),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # 2. Add TRAIN PREDICTIONS (GREEN DASHED)
        train_start_idx = window_size
        train_end_idx = train_size
        train_dates = df_dates[train_start_idx:train_end_idx]
        
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_predict,
            name='Train Predictions',
            mode='lines',
            line=dict(
                color='green',
                width=2,
                dash='dash'
            ),
            hovertemplate='<b>Train Pred</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # 3. Add TEST PREDICTIONS (RED DASHED)
        test_start_idx = train_size + window_size
        test_dates = df_dates[test_start_idx:]
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict,
            name='Test Predictions',
            mode='lines',
            line=dict(
                color='red',
                width=2,
                dash='dash'
            ),
            hovertemplate='<b>Test Pred</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'<b>{st.session_state.stock_symbol} - Actual vs Predicted Prices</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'blue'}
            },
            xaxis_title='<b>Date</b>',
            yaxis_title='<b>Price ($)</b>',
            hovermode='x unified',
            height=600,
            template='plotly_white',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=12)
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show explanation
        st.success("""
        ✅ **Chart Explanation:**
        - 🔵 **BLUE LINE (Solid)** = Actual historical prices (what really happened)
        - 🟢 **GREEN LINE (Dashed)** = Training predictions (model learning phase)
        - 🔴 **RED LINE (Dashed)** = Test predictions (model evaluation phase)
        
        **How to read:**
        - Blue line = Ground truth
        - Green and red lines should follow blue closely = Good model
        - If they diverge = Model needs improvement
        """)
    
    # =====================================================================
    # TAB 2: PERFORMANCE
    # =====================================================================
    
    with tab2:
        st.subheader("Performance Metrics")
        
        # Calculate metrics
        train_rmse = compute_rmse(st.session_state.data['y_train_actual'], st.session_state.data['train_predict'])
        train_mae = compute_mae(st.session_state.data['y_train_actual'], st.session_state.data['train_predict'])
        
        test_rmse = compute_rmse(st.session_state.data['y_test_actual'], st.session_state.data['test_predict'])
        test_mae = compute_mae(st.session_state.data['y_test_actual'], st.session_state.data['test_predict'])
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🟢 Training Set (Model Learning)**")
            st.metric("RMSE", f"${train_rmse:.2f}")
            st.metric("MAE", f"${train_mae:.2f}")
        
        with col2:
            st.write("**🔴 Test Set (Model Evaluation)**")
            st.metric("RMSE", f"${test_rmse:.2f}")
            st.metric("MAE", f"${test_mae:.2f}")
        
        st.markdown("---")
        
        # Baseline comparison
        st.write("**Baseline Comparison**")
        
        # Naive baseline
        naive_pred = np.concatenate([[st.session_state.data['y_test_actual'][0]], st.session_state.data['y_test_actual'][:-1]])
        naive_mae = compute_mae(st.session_state.data['y_test_actual'], naive_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Naive Baseline MAE", f"${naive_mae:.2f}")
        with col2:
            st.metric("Model MAE", f"${test_mae:.2f}")
        with col3:
            improvement = ((naive_mae - test_mae) / naive_mae) * 100
            if improvement > 0:
                st.metric("Improvement", f"✅ {improvement:.1f}%")
            else:
                st.metric("Improvement", f"❌ {improvement:.1f}%")
    
    # =====================================================================
    # TAB 3: INFO
    # =====================================================================
    
    with tab3:
        st.subheader("About This App")
        
        with st.expander("📖 How it works", expanded=True):
            st.write("""
            **Process:**
            1. Fetches historical stock data from Yahoo Finance
            2. Normalizes prices to 0-1 range
            3. Creates 3-day rolling windows
            4. Splits into 80% training, 20% testing
            5. Makes predictions using simple moving average
            6. Evaluates accuracy on unseen test data
            
            **Why this matters:**
            - Training data: Shows if model can learn patterns
            - Test data: Shows if model can predict NEW data
            - Both should be close to actual prices = Good model
            """)
        
        with st.expander("📊 Metrics Explained"):
            st.write("""
            **RMSE (Root Mean Squared Error)**
            - Average prediction error in dollars
            - Penalizes large errors more
            - Lower is better
            
            **MAE (Mean Absolute Error)**
            - Average absolute error in dollars
            - More interpretable than RMSE
            - $5 MAE = predictions off by $5 on average
            
            **Example:**
            - MAE = $2 means predictions are ~$2 off
            - RMSE = $3 means some predictions are worse
            """)
        
        with st.expander("⚠️ Important Disclaimer"):
            st.write("""
            ❌ **DO NOT use this for real trading!**
            
            Reasons:
            - Past performance ≠ future results
            - Markets are affected by unpredictable events
            - This simple model can't capture everything
            - Use only as ONE signal in a broader strategy
            
            ✅ **Better approach:**
            - Combine with other technical indicators
            - Include fundamental analysis
            - Use proper risk management
            - Consult financial advisors
            """)

else:
    st.info("👈 **Configure parameters and click 'Load Data & Analyze'** to get started!")
    
    with st.expander("ℹ️ How to use this app", expanded=True):
        st.write("""
        **Steps:**
        1. Enter stock ticker (AAPL, MSFT, GOOGL, etc.)
        2. Set date range (default: last 5 years)
        3. Set window size (default: 3 days)
        4. Click "Load Data & Analyze"
        5. Wait for analysis
        6. View chart and metrics
        
        **Tips:**
        - Use 2-5 years of data for best results
        - Window size 3-10 days works well
        - More data = potentially better predictions
        """)

st.markdown("---")
st.markdown("""
**Stock Price Prediction | Moving Average Model**
- 🔵 Blue = Actual | 🟢 Green = Train | 🔴 Red = Test
- Shows: Actual prices vs Model predictions
""")