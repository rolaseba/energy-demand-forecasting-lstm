import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Energy Demand Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stSlider > div > div > div > div {
        color: #4F46E5;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load data from standard paths."""
    # Path relative to this script
    script_dir = Path(__file__).parent
    DATA_DIR = script_dir / 'data'
    
    try:
        original = pd.read_parquet(DATA_DIR / 'original_data.parquet')
        return original
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load LSTM model and scaler."""
    script_dir = Path(__file__).parent
    MODEL_DIR = script_dir / 'model'
    
    try:
        model = load_model(MODEL_DIR / 'best_lstm_model.keras')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

original_data = load_data()
model, scaler = load_model_artifacts()

if original_data is None or model is None:
    st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_sequence(raw_values, scaler, sequence_length=168):
    """Preprocess data for LSTM input."""
    raw_2d = np.array(raw_values).reshape(-1, 1)
    scaled = scaler.transform(raw_2d).flatten()
    return scaled[-sequence_length:].reshape(1, sequence_length, 1)

def predict_future(model, scaler, history_window, horizon=168):
    """Generate recursive forecast."""
    predictions = []
    current_batch = preprocess_sequence(history_window, scaler)
    
    # We work in scaled space for recursion
    # But for display we need to inverse transform each step or end result
    # It's better to stay in scaled space for the loop
    
    current_input_scaled = current_batch[0, :, 0] # (168,)
    
    for _ in range(horizon):
        # Reshape for model: (1, 168, 1)
        X_in = current_input_scaled[-168:].reshape(1, 168, 1)
        
        # Predict next step
        y_pred_scaled = model.predict(X_in, verbose=0)[0, 0]
        predictions.append(y_pred_scaled)
        
        # Append to input for next step
        current_input_scaled = np.append(current_input_scaled, y_pred_scaled)
    
    # Inverse transform all predictions at once
    pred_array = np.array(predictions).reshape(-1, 1)
    pred_mw = scaler.inverse_transform(pred_array).flatten()
    
    return pred_mw

# ============================================================================
# APP LAYOUT & LOGIC
# ============================================================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Strategic Energy Demand Forecasting")
    st.markdown("##### Business intelligence for national grid optimization powered by **LSTM** (a Recurrent Neural Network model).")

# Sidebar Controls
with st.sidebar:
    st.header("Pattern Controls")
    
    # Navigation
    max_idx = len(original_data) - 168 - 1
    default_idx = max_idx - 168
    
    start_index = st.slider(
        "Observation Point (Time Step)",
        min_value=168,
        max_value=max_idx,
        value=41880, # Custom default for demo
        step=24,
        help="Select the point in time to stand and look forward."
    )
    
    st.divider()
    
    # Overlays
    st.subheader("Base Patterns")
    pattern_mode = st.radio(
        "Show Cycle Overlay",
        ["None", "Daily (24h)", "Weekly (168h)"],
        index=2, # Default to Weekly
        help="Visual guides to see repeating patterns. 'Daily' marks every 24 steps (virtual day). 'Weekly' marks every 168 steps (virtual week)."
    )
    
    st.divider()
    
    # Forecast
    horizon = st.slider(
        "Forecast Horizon (Steps)", 
        24, 336,
        value=96, # Custom default for demo
        step=24,
        help="How far ahead to predict. 24 steps ≈ 1 day, 168 steps ≈ 1 week."
    )
    
    st.divider()
    
    # Technical Details Expander
    with st.expander("Model Architecture & Specs"):
        st.markdown("### 🧠 LSTM Configuration")
        st.code("""
Input: (168, 1)
  ↓
LSTM (128 units) + BN
  ↓
LSTM (64 units) + BN
  ↓
LSTM (32 units) + BN
  ↓
Dense (1) -> Output
        """, language="text")
        
        st.markdown("### ⚙️ Hyperparameters")
        st.markdown("""
        - **Optimizer**: Adam (lr=0.001)
        - **Loss**: MSE
        - **Dropout**: 0.2
        - **Batch Size**: 64
        - **Params**: ~129,313
        """)
        
        st.markdown("### 📊 Performance (Test)")
        st.markdown("""
        - **MAPE**: 3.95% (Excellent)
        - **MAE**: 320.21 MW
        - **RMSE**: 1456.82 MW
        """)
        
        st.markdown("### 📂 Dataset")
        st.caption("PGCB Hourly Generation Dataset (Bangladesh)")
    
    st.divider()
    
    # Portfolio Footer
    st.markdown("### 👨‍💻 Project & Author")
    st.info(
        "**Sebastián Rolando**\n\n"
        "This project is part of my Data Science portfolio.\n\n"
        "[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/rolaseba/energy-demand-forecasting-lstm)"
    )

# Main Visualization Logic
# We need:
# 1. Historical Context (e.g., last 1680 indices = 10 weeks)
# 2. Future Ground Truth (if available, for comparison)
# 3. Forecast Line

history_len = 168 * 2 # Show 2 weeks of history context
context_start = max(0, start_index - history_len)

# Extract Data
history_slice = original_data.iloc[context_start:start_index]
future_slice = original_data.iloc[start_index:min(start_index + horizon, len(original_data))]

# Generate Forecast
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.info("Generating Neural Forecast...")
    
# Get the strict 168 context needed for prediction
prediction_context = original_data['demand_mw'].values[start_index-168:start_index]
forecast_values = predict_future(model, scaler, prediction_context, horizon=horizon)

loading_placeholder.empty()

# Create Pattern Continuum Plot
fig = go.Figure()

# 1. Historical Data
fig.add_trace(go.Scatter(
    x=history_slice.index,
    y=history_slice['demand_mw'],
    mode='lines',
    name='History',
    line=dict(color='#94A3B8', width=2),
    hovertemplate='Time Step: %{x}<br>Demand: %{y:.0f} MW<extra></extra>'
))

# 2. Ground Truth (Future)
if not future_slice.empty:
    fig.add_trace(go.Scatter(
        x=future_slice.index,
        y=future_slice['demand_mw'],
        mode='lines',
        name='Actual Future',
        line=dict(color='#475569', width=2, dash='dot'),
        opacity=0.5,
        hovertemplate='Time Step: %{x}<br>Actual: %{y:.0f} MW<extra></extra>'
    ))

# 3. Model Forecast
forecast_indices = np.arange(start_index, start_index + len(forecast_values))
fig.add_trace(go.Scatter(
    x=forecast_indices,
    y=forecast_values,
    mode='lines',
    name='LSTM Forecast',
    line=dict(color='#4F46E5', width=3),
    hovertemplate='Time Step: %{x}<br>Pred: %{y:.0f} MW<extra></extra>'
))

# 4. Pattern Overlays
# Calculate view range for overlaps
view_min = context_start
view_max = start_index + horizon

if pattern_mode == "Daily (24h)":
    # Find first index multiple of 24 in view
    first_daily = (view_min // 24) * 24
    if first_daily < view_min:
        first_daily += 24
        
    daily_indices = range(first_daily, view_max, 24)
    
    for idx in daily_indices:
        # Increased visibility: darker color, higher opacity
        fig.add_vline(x=idx, line_width=1, line_color="#475569", opacity=0.5)

elif pattern_mode == "Weekly (168h)":
    # Find first index multiple of 168 in view
    first_weekly = (view_min // 168) * 168
    if first_weekly < view_min:
        first_weekly += 168
    
    weekly_indices = range(first_weekly, view_max, 168)
    for idx in weekly_indices:
        # Increased visibility: much darker, full opacity, thicker line
        fig.add_vline(x=idx, line_width=2, line_color="#1E293B", opacity=0.8, annotation_text="Cycle Start", annotation_position="top left")

# Layout Polish
fig.update_layout(
    title=dict(text=f"Timeline at Step {start_index}", font=dict(size=18)),
    xaxis=dict(
        title="Time Step",
        showgrid=False,
        zeroline=False,
        range=[context_start, start_index + horizon + 24]
    ),
    yaxis=dict(
        title="Demand (MW)",
        showgrid=True,
        gridcolor='#1E293B'
    ),
    height=500,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template="plotly_white"
)

st.plotly_chart(fig, width="stretch")

# Insights Section
st.subheader("Pattern Insights")
c1, c2, c3 = st.columns(3)

with c1:
    idx_mod_24 = start_index % 24
    st.metric("Cycle Position (0-23)", f"{idx_mod_24}", help="Position within the daily cycle (0-23)")
    st.caption(f"Comparing step {start_index} to similar hourly positions.")

with c2:
    current_val = original_data.iloc[start_index]['demand_mw']
    week_ago_val = original_data.iloc[start_index - 168]['demand_mw']
    delta = current_val - week_ago_val
    st.metric(
        "Week-over-Week Delta", 
        f"{delta:+.0f} MW", 
        delta_color="inverse",
        help="Difference vs. exactly 168 steps ago. Positive means demand is higher than last 'week'."
    )
    
    with c3:
        # Simple "confidence" proxy based on horizon
        confidence = max(100 - (horizon / 336 * 50), 50)
        st.metric(
            "Forecast Confidence", 
            f"{confidence:.0f}%", 
            help="Estimated reliability. Predicting further into the future increases uncertainty."
        )

# Gamification / Exploration
with st.expander("🕵️ Pattern Detective: Find Similar Days"):
    st.write("The model learns by seeing similar patterns. Below are indices with similar recent history to your current position.")
    
    # Naive similarity search (euclidean distance of last 24h)
    current_pattern = prediction_context[-24:] # Last 24 pts
    
    # We'll just check a few random past weeks for demo speed
    candidates = [start_index - 168*i for i in range(1, 11)]
    scores = []
    
    for cand in candidates:
        if cand < 24: continue
        past_pattern = original_data.iloc[cand-24:cand]['demand_mw'].values
        if len(past_pattern) == 24:
            dist = np.linalg.norm(current_pattern - past_pattern)
            scores.append((cand, dist))
    
    scores.sort(key=lambda x: x[1])
    
    cols = st.columns(3)
    for i, (idx, dist) in enumerate(scores[:3]):
        with cols[i]:
            st.info(f"Index {idx}")
            st.caption(f"Difference: {dist:.1f}")

