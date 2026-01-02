# Data Files Usage Guide for Streamlit App
## Energy Demand Forecasting LSTM Model

---

## Table of Contents

1. [Overview](#overview)
2. [Data Files Summary](#data-files-summary)
3. [Loading Data Files](#loading-data-files)
4. [Data Structure & Content](#data-structure--content)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Making Predictions](#making-predictions)
7. [Complete Streamlit App Example](#complete-streamlit-app-example)
8. [Common Use Cases](#common-use-cases)
9. [Error Handling & Validation](#error-handling--validation)

---

## Overview

The `prepare_streamlit_data.py` script generates **4 Parquet files** stored in `/datasets/processed/` that contain all necessary data for building a Streamlit prediction app. These files are optimized for:

- **Fast Loading**: Parquet format is columnar and compressed
- **Type Safety**: Schema preserved across formats
- **Production Ready**: Minimal preprocessing required
- **Easy Integration**: Direct compatibility with Pandas and Streamlit

### Key Advantages of This Approach

| Aspect | Benefit |
|--------|---------|
| **Format** | Parquet is 10-100x faster than CSV for large datasets |
| **Schema** | Data types are preserved (no type inference errors) |
| **Storage** | ~80% smaller than CSV due to compression |
| **Scalability** | Handles 100k+ rows efficiently |
| **Streamlit Ready** | Native Pandas integration, instant caching |

---

## Data Files Summary

### File 1: `original_data.parquet`
**Purpose**: Complete cleaned time series data  
**Size**: ~2-3 MB  
**Records**: ~92,000 rows  
**Primary Use**: Historical data display, trend analysis, data visualization

```
Columns:
├── datetime          (datetime64) → Timestamp of each observation
├── demand_mw         (float64)    → Electricity demand in megawatts
└── generation_mw     (float64)    → Electricity generation in megawatts
```

### File 2: `train_test_data.parquet`
**Purpose**: Train/Test split information and indices  
**Size**: ~1-2 MB  
**Records**: ~92,000 rows  
**Primary Use**: Understanding data split, filtering training vs test data

```
Columns:
├── index             (int64)      → Row index in original data
├── demand_mw         (float64)    → Demand value (copy from original)
├── data_type         (string)     → 'train' or 'test' label
├── is_in_training_set (int64)    → 1=train, 0=test (binary format)
└── datetime          (datetime64) → [Optional] Timestamp if available
```

### File 3: `scaler_stats.parquet`
**Purpose**: Scaler parameters and statistics  
**Size**: <1 MB  
**Records**: 1 row (metadata)  
**Primary Use**: Understanding normalization applied to training data

```
Columns (varies by scaler type):
├── scaler_type           (string) → 'StandardScaler' or 'MinMaxScaler'
├── scale_data            (bool)   → Whether scaling was applied
├── sequence_length       (int)    → LSTM input window (168 hours)
├── data_min              (float)  → [MinMaxScaler] Minimum training value
├── data_max              (float)  → [MinMaxScaler] Maximum training value
├── feature_range         (tuple)  → [MinMaxScaler] Output range (0, 1)
├── mean                  (float)  → [StandardScaler] Training mean
└── scale                 (float)  → [StandardScaler] Training std deviation
```

### File 4: `metadata.parquet`
**Purpose**: Configuration, timestamps, and summary statistics  
**Size**: <1 MB  
**Records**: 1 row (metadata)  
**Primary Use**: Validation, logging, and configuration reference

```
Columns:
├── creation_date              (string)   → ISO format timestamp
├── model_path                 (string)   → Path to LSTM model
├── scaler_path                (string)   → Path to scaler object
├── sequence_length            (int)      → Input window size (168)
├── original_data_shape        (tuple)    → (rows, columns)
├── scaler_statistics          (dict)     → Statistics dictionary
└── files_created              (list)     → List of output files
```

---

## Loading Data Files

### Basic Loading with Pandas

```python
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# Define paths
DATA_DIR = Path('datasets/processed')
MODEL_DIR = Path('model')

# ============================================================================
# STEP 1: LOAD ALL DATA FILES
# ============================================================================

def load_all_data_files():
    """
    Load all 4 data files from processed directory.
    
    Returns:
    --------
    dict : Dictionary containing all loaded dataframes
    """
    
    print("Loading data files...")
    
    # Load original cleaned time series
    original_data = pd.read_parquet(DATA_DIR / 'original_data.parquet')
    print(f"✓ Original data: {original_data.shape}")
    
    # Load train/test split information
    train_test_info = pd.read_parquet(DATA_DIR / 'train_test_data.parquet')
    print(f"✓ Train/Test info: {train_test_info.shape}")
    
    # Load scaler statistics
    scaler_stats = pd.read_parquet(DATA_DIR / 'scaler_stats.parquet')
    print(f"✓ Scaler stats: {scaler_stats.shape}")
    
    # Load metadata
    metadata = pd.read_parquet(DATA_DIR / 'metadata.parquet')
    print(f"✓ Metadata: {metadata.shape}")
    
    return {
        'original_data': original_data,
        'train_test_info': train_test_info,
        'scaler_stats': scaler_stats,
        'metadata': metadata
    }

# Load all files
data_files = load_all_data_files()
```

### Streamlit-Optimized Loading with Caching

```python
import streamlit as st
from pathlib import Path

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load data files with Streamlit caching for performance."""
    
    DATA_DIR = Path('datasets/processed')
    
    original_data = pd.read_parquet(DATA_DIR / 'original_data.parquet')
    train_test_info = pd.read_parquet(DATA_DIR / 'train_test_data.parquet')
    scaler_stats = pd.read_parquet(DATA_DIR / 'scaler_stats.parquet')
    metadata = pd.read_parquet(DATA_DIR / 'metadata.parquet')
    
    return {
        'original': original_data,
        'train_test': train_test_info,
        'scaler': scaler_stats,
        'metadata': metadata
    }

# Load in Streamlit app
data = load_data()
original_data = data['original']
train_test_info = data['train_test']
scaler_stats = data['scaler']
metadata = data['metadata']
```

---

## Data Structure & Content

### Understanding Original Data

```python
# Load and inspect
original_data = pd.read_parquet('datasets/processed/original_data.parquet')

print(original_data.head(10))
print("\nData Info:")
print(original_data.info())
print("\nStatistics:")
print(original_data.describe())

# Output Example:
#              datetime  demand_mw  generation_mw
# 0 2016-01-01 00:00:00     5241.0       4845.000
# 1 2016-01-01 01:00:00     5089.0       4721.000
# 2 2016-01-01 02:00:00     4867.0       4532.000
# ...

# Key Statistics:
# - Min demand: 2,000 MW
# - Max demand: 16,500 MW
# - Mean demand: 8,500 MW
# - Std dev: 2,200 MW
```

### Understanding Train/Test Split

```python
train_test_info = pd.read_parquet('datasets/processed/train_test_data.parquet')

# Count samples by type
train_count = (train_test_info['data_type'] == 'train').sum()
test_count = (train_test_info['data_type'] == 'test').sum()

print(f"Training samples: {train_count:,} ({train_count/len(train_test_info)*100:.1f}%)")
print(f"Test samples: {test_count:,} ({test_count/len(train_test_info)*100:.1f}%)")

# Get date ranges
if 'datetime' in train_test_info.columns:
    train_dates = train_test_info[train_test_info['data_type'] == 'train']['datetime']
    test_dates = train_test_info[train_test_info['data_type'] == 'test']['datetime']
    
    print(f"\nTraining period: {train_dates.min()} to {train_dates.max()}")
    print(f"Test period: {test_dates.min()} to {test_dates.max()}")
```

### Understanding Scaler Statistics

```python
scaler_stats = pd.read_parquet('datasets/processed/scaler_stats.parquet')

# Convert to dictionary for easier access
scaler_info = scaler_stats.iloc[0].to_dict()

print(f"Scaler Type: {scaler_info['scaler_type']}")
print(f"Sequence Length: {scaler_info['sequence_length']} hours")

# For StandardScaler
if scaler_info['scaler_type'] == 'StandardScaler':
    mean = scaler_info['mean']
    std = scaler_info['scale']
    
    print(f"Training Data Mean: {mean:,.2f} MW")
    print(f"Training Data Std Dev: {std:,.2f} MW")
    
    # Interpretation: Data was normalized to (value - mean) / std

# For MinMaxScaler
elif scaler_info['scaler_type'] == 'MinMaxScaler':
    min_val = scaler_info['data_min']
    max_val = scaler_info['data_max']
    
    print(f"Training Data Min: {min_val:,.2f} MW")
    print(f"Training Data Max: {max_val:,.2f} MW")
    
    # Interpretation: Data was scaled to (value - min) / (max - min)
```

---

## Preprocessing Pipeline

### Step 1: Load Raw Input Data

```python
import pandas as pd
import numpy as np

# Load the original data
original_data = pd.read_parquet('datasets/processed/original_data.parquet')

# Extract demand values (the target variable)
demand_series = original_data['demand_mw'].values  # Shape: (92000,)

print(f"Loaded demand series shape: {demand_series.shape}")
print(f"Data range: [{demand_series.min():.2f}, {demand_series.max():.2f}] MW")
```

### Step 2: Load and Apply Scaler

```python
import joblib

# Load the fitted scaler
scaler = joblib.load('model/scaler.pkl')

# IMPORTANT: The scaler was fitted on TRAINING DATA ONLY
# This prevents data leakage and ensures fair evaluation

# Normalize the data
# For new/future data (e.g., last 168 hours for prediction)
raw_demand = np.array([5241.0, 5089.0, 4867.0, ...])  # Your input data

# Reshape to 2D for scaler (required format)
raw_demand_2d = raw_demand.reshape(-1, 1)  # Shape: (n, 1)

# Scale the data
demand_scaled = scaler.transform(raw_demand_2d)  # Shape: (n, 1)
demand_scaled_flat = demand_scaled.flatten()  # Shape: (n,)

print(f"Original range: [{raw_demand.min():.2f}, {raw_demand.max():.2f}]")
print(f"Scaled range: [{demand_scaled.min():.4f}, {demand_scaled.max():.4f}]")
```

### Step 3: Create Sequences for LSTM

```python
def create_sequences(data, sequence_length=168):
    """
    Create sliding window sequences for LSTM input.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of scaled demand values
    sequence_length : int
        Number of timesteps to look back (default: 168 hours = 1 week)
    
    Returns:
    --------
    np.ndarray
        3D array of shape (num_sequences, sequence_length, 1)
    
    Example:
    --------
    Input data: [1, 2, 3, 4, 5, 6, 7, 8]
    sequence_length: 3
    
    Output sequences:
    - [1, 2, 3] → predict 4
    - [2, 3, 4] → predict 5
    - [3, 4, 5] → predict 6
    - ...
    """
    
    X = []
    
    for i in range(len(data) - sequence_length):
        # Extract window of size sequence_length
        window = data[i:i + sequence_length]
        X.append(window)
    
    X = np.array(X)  # Shape: (num_sequences, sequence_length)
    
    # Reshape to 3D for LSTM: (num_sequences, sequence_length, 1 feature)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X

# Apply to your data
SEQUENCE_LENGTH = 168  # Must match training configuration

# Get last 168 hours of demand data
last_168_hours = demand_scaled_flat[-168:]  # Last 168 values

# Create sequence for prediction
X_input = create_sequences(last_168_hours, sequence_length=SEQUENCE_LENGTH)
# Output shape: (1, 168, 1) - Ready for model.predict()

print(f"Input sequence shape: {X_input.shape}")
print(f"Expected by LSTM: (batch_size, 168, 1)")
```

### Complete Preprocessing Function

```python
def preprocess_for_prediction(raw_demand_values, scaler, sequence_length=168):
    """
    Complete preprocessing pipeline for making predictions.
    
    Parameters:
    -----------
    raw_demand_values : np.ndarray or list
        Raw demand values in MW (not scaled)
    scaler : sklearn.preprocessing scaler
        Fitted scaler object (loaded from model/scaler.pkl)
    sequence_length : int
        LSTM input window size
    
    Returns:
    --------
    np.ndarray
        3D array ready for model.predict(), shape (1, sequence_length, 1)
    
    Raises:
    -------
    ValueError : If input has fewer than sequence_length values
    """
    
    # Convert to numpy array if needed
    if isinstance(raw_demand_values, list):
        raw_demand_values = np.array(raw_demand_values)
    
    # Validate input
    if len(raw_demand_values) < sequence_length:
        raise ValueError(
            f"Input has {len(raw_demand_values)} values, "
            f"but {sequence_length} are required"
        )
    
    # Step 1: Scale the data
    raw_2d = raw_demand_values.reshape(-1, 1)
    scaled = scaler.transform(raw_2d).flatten()
    
    # Step 2: Take the last 'sequence_length' values
    # (most recent data point is at the end)
    recent_data = scaled[-sequence_length:]
    
    # Step 3: Reshape for LSTM
    # (batch_size=1, timesteps=168, features=1)
    X = recent_data.reshape(1, sequence_length, 1)
    
    return X

# Usage example:
raw_demand = np.array([5241, 5089, 4867, 4932, 5123, ...])  # 168+ values

X_prepared = preprocess_for_prediction(
    raw_demand,
    scaler=scaler,
    sequence_length=168
)

print(f"Prepared input shape: {X_prepared.shape}")  # (1, 168, 1)
```

---

## Making Predictions

### Step 1: Load the Trained Model

```python
from tensorflow.keras.models import load_model
import os

# Load model with error handling
model_path = 'model/best_lstm_model.keras'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = load_model(model_path)

print(f"✓ Model loaded successfully")
print(f"  Total parameters: {model.count_params():,}")
print(f"\nModel Architecture:")
model.summary()

# Model Summary Output:
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 168, 128)         66,560
# batch_normalization          (None, 168, 128)         512
# dropout (Dropout)            (None, 168, 128)         0
# lstm_1 (LSTM)                (None, 168, 64)          49,408
# batch_normalization_1        (None, 168, 64)          256
# dropout_1 (Dropout)          (None, 168, 64)          0
# lstm_2 (LSTM)                (None, 32)               12,416
# batch_normalization_2        (None, 32)               128
# dropout_2 (Dropout)          (None, 32)               0
# dense (Dense)                (None, 1)                33
# =================================================================
# Total params: 129,313
```

### Step 2: Make Scaled Prediction

```python
# Using preprocessed input from previous section
X_prepared = preprocess_for_prediction(raw_demand, scaler, sequence_length=168)

# Make prediction in SCALED space
y_pred_scaled = model.predict(X_prepared, verbose=0)
# Output shape: (1, 1) - Single value for next hour

# Flatten to get single value
y_pred_scaled_value = y_pred_scaled.flatten()[0]

print(f"Predicted scaled value: {y_pred_scaled_value:.4f}")
print(f"(This value is in normalized space, not MW)")
```

### Step 3: Inverse Transform to Original Scale

```python
# Convert prediction back to MW (original scale)
y_pred_original = scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()[0]

print(f"Predicted demand for next hour: {y_pred_original:,.2f} MW")
```

### Complete Prediction Pipeline

```python
def predict_next_hour(
    raw_demand_history: np.ndarray,
    model,
    scaler,
    sequence_length: int = 168,
    return_confidence: bool = False
) -> dict:
    """
    Complete pipeline: preprocess → predict → postprocess.
    
    Parameters:
    -----------
    raw_demand_history : np.ndarray
        Historical demand values in MW (must include last 168 hours minimum)
    model : keras.Model
        Trained LSTM model
    scaler : sklearn scaler
        Fitted scaler
    sequence_length : int
        LSTM input window (must match training)
    return_confidence : bool
        If True, include prediction uncertainty (not available from single prediction)
    
    Returns:
    --------
    dict : Dictionary with prediction details
        - 'predicted_demand_mw': Single-hour prediction in MW
        - 'timestamp': Datetime of prediction (if available)
        - 'confidence_level': 'High', 'Medium', 'Low' (approximate)
    """
    
    try:
        # Step 1: Validate input
        if len(raw_demand_history) < sequence_length:
            raise ValueError(
                f"Need {sequence_length} historical values, "
                f"got {len(raw_demand_history)}"
            )
        
        # Step 2: Preprocess
        X_prepared = preprocess_for_prediction(
            raw_demand_history,
            scaler,
            sequence_length
        )
        
        # Step 3: Predict (in scaled space)
        y_pred_scaled = model.predict(X_prepared, verbose=0)
        
        # Step 4: Inverse transform (back to MW)
        y_pred_original = scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]
        
        # Step 5: Validate prediction (sanity checks)
        # Check if prediction is within reasonable bounds
        data_min = raw_demand_history.min()
        data_max = raw_demand_history.max()
        data_mean = raw_demand_history.mean()
        data_std = raw_demand_history.std()
        
        # Predictions usually within ±3 std from mean
        lower_bound = max(0, data_mean - 3*data_std)
        upper_bound = data_mean + 3*data_std
        
        is_reasonable = lower_bound <= y_pred_original <= upper_bound
        confidence = "High" if is_reasonable else "Low"
        
        return {
            'predicted_demand_mw': round(y_pred_original, 2),
            'confidence': confidence,
            'reasonable': is_reasonable,
            'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)},
            'historical_mean': round(data_mean, 2),
            'historical_std': round(data_std, 2)
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'predicted_demand_mw': None,
            'confidence': 'Failed'
        }

# Usage example:
history = original_data['demand_mw'].tail(168).values  # Last 168 hours

prediction = predict_next_hour(
    raw_demand_history=history,
    model=model,
    scaler=scaler,
    sequence_length=168
)

print(f"Predicted Demand: {prediction['predicted_demand_mw']} MW")
print(f"Confidence: {prediction['confidence']}")
print(f"Reasonable: {prediction['reasonable']}")
```

### Batch Predictions (Multiple Hours)

```python
def predict_multiple_hours(
    raw_demand_history: np.ndarray,
    model,
    scaler,
    hours_ahead: int = 24,
    sequence_length: int = 168
) -> np.ndarray:
    """
    Predict multiple hours ahead iteratively.
    
    NOTE: This is a simple approach. For better results, consider:
    - Training a multi-step model
    - Using ensemble methods
    - Including external features
    
    Parameters:
    -----------
    raw_demand_history : np.ndarray
        Historical values
    hours_ahead : int
        How many hours in the future to predict
    
    Returns:
    --------
    np.ndarray
        Array of predictions for the next 'hours_ahead' hours
    """
    
    predictions = []
    
    # Start with the historical data
    current_history = raw_demand_history.copy()
    
    for i in range(hours_ahead):
        # Predict next hour
        pred = predict_next_hour(
            current_history[-sequence_length:],
            model,
            scaler,
            sequence_length
        )
        
        if 'error' not in pred:
            next_pred = pred['predicted_demand_mw']
            predictions.append(next_pred)
            
            # Add prediction to history for next iteration
            current_history = np.append(current_history, next_pred)
        else:
            break
    
    return np.array(predictions)

# Usage:
forecast_24h = predict_multiple_hours(
    history,
    model,
    scaler,
    hours_ahead=24,
    sequence_length=168
)

print(f"24-hour forecast: {forecast_24h}")
```

---

## Complete Streamlit App Example

### Minimal Working App

```python
# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Energy Demand Forecasting App")
st.markdown("""
This app predicts electricity demand for Bangladesh's national power grid
using a trained LSTM neural network.
""")

# ============================================================================
# LOAD DATA AND MODELS (WITH CACHING)
# ============================================================================

@st.cache_data(ttl=3600)
def load_all_files():
    """Load data files with caching."""
    data_dir = Path('datasets/processed')
    
    original = pd.read_parquet(data_dir / 'original_data.parquet')
    train_test = pd.read_parquet(data_dir / 'train_test_data.parquet')
    scaler_stats = pd.read_parquet(data_dir / 'scaler_stats.parquet')
    metadata = pd.read_parquet(data_dir / 'metadata.parquet')
    
    return original, train_test, scaler_stats, metadata

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler (resource-level caching)."""
    model = load_model('model/best_lstm_model.keras')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# Load everything
original_data, train_test, scaler_stats, metadata = load_all_files()
model, scaler = load_model_and_scaler()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_for_prediction(raw_values, sequence_length=168):
    """Preprocess data for LSTM prediction."""
    if len(raw_values) < sequence_length:
        st.error(f"Need {sequence_length} values, got {len(raw_values)}")
        return None
    
    raw_2d = np.array(raw_values).reshape(-1, 1)
    scaled = scaler.transform(raw_2d).flatten()
    recent = scaled[-sequence_length:]
    
    return recent.reshape(1, sequence_length, 1)

def make_prediction(history):
    """Make single-hour prediction."""
    X = preprocess_for_prediction(history, sequence_length=168)
    if X is None:
        return None
    
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_original = scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    return y_pred_original

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Sidebar
with st.sidebar:
    st.header("📊 Settings")
    
    # Select date range
    date_range = st.date_input(
        "Select date range for analysis",
        value=(
            original_data['datetime'].min().date(),
            original_data['datetime'].max().date()
        ),
        key="date_range"
    )
    
    # Historical data selection
    st.subheader("Historical Data")
    show_stats = st.checkbox("Show data statistics", value=True)
    
    # Prediction settings
    st.subheader("Prediction Settings")
    hours_to_predict = st.slider(
        "Hours to forecast",
        min_value=1,
        max_value=168,
        value=24,
        step=1
    )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Historical Data", "🔮 Make Prediction", "📊 Data Stats", "ℹ️ Info"]
)

# ============================================================================
# TAB 1: HISTORICAL DATA
# ============================================================================

with tab1:
    st.header("Historical Electricity Demand")
    
    # Filter data by date
    filtered_data = original_data[
        (pd.to_datetime(original_data['datetime']).dt.date >= date_range[0]) &
        (pd.to_datetime(original_data['datetime']).dt.date <= date_range[1])
    ]
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_data['datetime'],
        y=filtered_data['demand_mw'],
        mode='lines',
        name='Actual Demand',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Electricity Demand Over Time",
        xaxis_title="Date",
        yaxis_title="Demand (MW)",
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    if show_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean Demand",
                f"{filtered_data['demand_mw'].mean():,.0f} MW"
            )
        
        with col2:
            st.metric(
                "Min Demand",
                f"{filtered_data['demand_mw'].min():,.0f} MW"
            )
        
        with col3:
            st.metric(
                "Max Demand",
                f"{filtered_data['demand_mw'].max():,.0f} MW"
            )
        
        with col4:
            st.metric(
                "Std Dev",
                f"{filtered_data['demand_mw'].std():,.0f} MW"
            )

# ============================================================================
# TAB 2: MAKE PREDICTION
# ============================================================================

with tab2:
    st.header("🔮 Predict Next Hours")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        This model predicts electricity demand based on the last 168 hours
        (1 week) of historical data.
        
        **Model Type**: LSTM Neural Network  
        **Input Window**: 168 hours  
        **Output**: Next hour demand prediction
        """)
    
    with col2:
        if st.button("Generate Forecast", key="predict_btn", use_container_width=True):
            st.write("### Prediction Results")
            
            # Get last 168 hours
            latest_history = original_data['demand_mw'].tail(168).values
            
            # Make prediction
            with st.spinner("Making prediction..."):
                pred = make_prediction(latest_history)
            
            if pred is not None:
                st.success(f"✅ Prediction Complete")
                
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                
                with col_pred1:
                    st.metric(
                        "Next Hour Prediction",
                        f"{pred:,.2f} MW",
                        delta=f"{pred - latest_history[-1]:,.2f} MW"
                    )
                
                with col_pred2:
                    mean_val = latest_history.mean()
                    st.metric(
                        "vs Historical Mean",
                        f"{(pred - mean_val)/mean_val*100:+.1f}%"
                    )
                
                with col_pred3:
                    st.metric(
                        "Last Hour",
                        f"{latest_history[-1]:,.2f} MW"
                    )
            else:
                st.error("❌ Prediction failed")

# ============================================================================
# TAB 3: DATA STATISTICS
# ============================================================================

with tab3:
    st.header("📊 Data Statistics & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Records**: {len(original_data):,}")
        st.write(f"**Date Range**: {original_data['datetime'].min()} to {original_data['datetime'].max()}")
        st.write(f"**Duration**: ~{len(original_data)/24/365:.1f} years")
    
    with col2:
        st.subheader("Train/Test Split")
        train_count = (train_test['data_type'] == 'train').sum()
        test_count = (train_test['data_type'] == 'test').sum()
        st.write(f"**Training Set**: {train_count:,} samples ({train_count/len(train_test)*100:.1f}%)")
        st.write(f"**Test Set**: {test_count:,} samples ({test_count/len(train_test)*100:.1f}%)")
    
    st.subheader("Demand Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{original_data['demand_mw'].mean():,.0f} MW")
    with col2:
        st.metric("Median", f"{original_data['demand_mw'].median():,.0f} MW")
    with col3:
        st.metric("Std Dev", f"{original_data['demand_mw'].std():,.0f} MW")
    with col4:
        st.metric("Range", f"{original_data['demand_mw'].max() - original_data['demand_mw'].min():,.0f} MW")

# ============================================================================
# TAB 4: APP INFO
# ============================================================================

with tab4:
    st.header("ℹ️ App Information")
    
    st.subheader("Model Details")
    metadata_dict = metadata.iloc[0].to_dict()
    st.write(f"**Creation Date**: {metadata_dict['creation_date']}")
    st.write(f"**Sequence Length**: {metadata_dict['sequence_length']} hours")
    st.write(f"**Model Path**: {metadata_dict['model_path']}")
    
    st.subheader("Data Files Used")
    st.write("""
    - `original_data.parquet`: Historical electricity demand
    - `train_test_data.parquet`: Train/test split information
    - `scaler_stats.parquet`: Normalization parameters
    - `metadata.parquet`: Configuration and metadata
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. **Data Loading**: Load last 168 hours of demand data
    2. **Normalization**: Scale using fitted scaler from training
    3. **Prediction**: Feed to LSTM model for next hour prediction
    4. **Inverse Transform**: Convert prediction back to MW
    5. **Display**: Show result with confidence metrics
    """)
```

### Save and Run

```bash
# Save the file
# streamlit_app.py

# Run the app
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

---

## Common Use Cases

### Use Case 1: Real-time Demand Monitoring Dashboard

```python
# Show current demand with next-hour forecast

def dashboard_view():
    """Create real-time monitoring dashboard."""
    
    col1, col2, col3 = st.columns(3)
    
    # Current demand
    current_demand = original_data['demand_mw'].iloc[-1]
    
    # 24-hour average
    demand_24h_avg = original_data['demand_mw'].tail(24).mean()
    
    # Prediction
    history = original_data['demand_mw'].tail(168).values
    next_pred = make_prediction(history)
    
    with col1:
        st.metric(
            "Current Demand",
            f"{current_demand:,.0f} MW",
            delta=f"{current_demand - demand_24h_avg:,.0f} MW"
        )
    
    with col2:
        st.metric("24h Average", f"{demand_24h_avg:,.0f} MW")
    
    with col3:
        st.metric("Next Hour Forecast", f"{next_pred:,.0f} MW")
```

### Use Case 2: Multi-Step Forecasting with Visualization

```python
def multi_step_forecast_view():
    """Generate and visualize multi-step forecast."""
    
    st.header("📊 Multi-Step Forecast")
    
    hours = st.slider("Forecast hours", 1, 72, 24)
    
    # Get historical data
    history = original_data['demand_mw'].values
    
    # Generate predictions (simple iterative approach)
    predictions = []
    current_history = history[-168:].copy()
    
    for _ in range(hours):
        pred = make_prediction(current_history)
        predictions.append(pred)
        current_history = np.append(current_history[1:], pred)
    
    # Create visualization
    forecast_df = pd.DataFrame({
        'hour': range(1, hours + 1),
        'predicted_demand': predictions
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_df['hour'],
        y=forecast_df['predicted_demand'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"{hours}-Hour Energy Demand Forecast",
        xaxis_title="Hour Ahead",
        yaxis_title="Predicted Demand (MW)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Forecast", f"{np.mean(predictions):,.0f} MW")
    with col2:
        st.metric("Min Forecast", f"{np.min(predictions):,.0f} MW")
    with col3:
        st.metric("Max Forecast", f"{np.max(predictions):,.0f} MW")
```

### Use Case 3: Comparison with Training Data

```python
def model_comparison_view():
    """Compare predictions with historical training data patterns."""
    
    st.header("📈 Model vs Historical Patterns")
    
    # Get hour of day
    current_hour = pd.Timestamp.now().hour
    
    # Historical demand at same hour
    historical_same_hour = original_data[
        pd.to_datetime(original_data['datetime']).dt.hour == current_hour
    ]['demand_mw']
    
    # Current prediction
    history = original_data['demand_mw'].tail(168).values
    pred = make_prediction(history)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Average at {current_hour}:00",
            f"{historical_same_hour.mean():,.0f} MW"
        )
    
    with col2:
        st.metric(
            "Std Dev at this hour",
            f"{historical_same_hour.std():,.0f} MW"
        )
    
    with col3:
        st.metric(
            "Model Prediction",
            f"{pred:,.0f} MW"
        )
```

---

## Error Handling & Validation

### Input Validation

```python
def validate_and_preprocess(raw_data, sequence_length=168):
    """
    Validate input data before prediction.
    
    Checks:
    - Data type and shape
    - Value ranges
    - Missing values
    - Sufficient length
    """
    
    # Type checking
    if not isinstance(raw_data, (list, np.ndarray, pd.Series)):
        raise TypeError(f"Expected array-like, got {type(raw_data)}")
    
    # Convert to numpy
    data = np.array(raw_data)
    
    # Check length
    if len(data) < sequence_length:
        raise ValueError(
            f"Need {sequence_length} values, got {len(data)}"
        )
    
    # Check for NaNs
    if np.isnan(data).any():
        raise ValueError("Input contains NaN values")
    
    # Check for infinite values
    if np.isinf(data).any():
        raise ValueError("Input contains infinite values")
    
    # Check for reasonable range
    if (data < 0).any():
        st.warning("⚠️ Negative values detected in input data")
    
    return data

# Usage
try:
    validated_data = validate_and_preprocess(input_data)
    prediction = make_prediction(validated_data)
except (ValueError, TypeError) as e:
    st.error(f"❌ Data validation error: {e}")
```

### Model Validation

```python
def validate_model_output(prediction, historical_data):
    """
    Validate prediction against historical statistics.
    """
    
    mean = historical_data.mean()
    std = historical_data.std()
    min_val = historical_data.min()
    max_val = historical_data.max()
    
    # Check if within reasonable bounds
    lower_bound = max(0, mean - 3*std)
    upper_bound = mean + 3*std
    
    is_valid = lower_bound <= prediction <= upper_bound
    
    return {
        'is_valid': is_valid,
        'bounds': (lower_bound, upper_bound),
        'deviation_from_mean': (prediction - mean) / std,  # In std units
        'warning': None if is_valid else "Prediction outside typical range"
    }

# Usage
validation = validate_model_output(pred, history)

if not validation['is_valid']:
    st.warning(f"⚠️ {validation['warning']}")
```

### Exception Handling in Streamlit

```python
def safe_prediction_with_error_handling():
    """Make prediction with comprehensive error handling."""
    
    try:
        # Load data
        with st.spinner("Loading data..."):
            data = load_all_files()
        
        # Validate
        history = original_data['demand_mw'].tail(168).values
        validate_and_preprocess(history)
        
        # Predict
        with st.spinner("Making prediction..."):
            pred = make_prediction(history)
        
        # Validate output
        validation = validate_model_output(pred, history)
        
        if validation['is_valid']:
            st.success(f"Prediction: {pred:,.0f} MW")
        else:
            st.warning(validation['warning'])
            st.info(f"Prediction: {pred:,.0f} MW (outside typical range)")
    
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
    
    except ValueError as e:
        st.error(f"❌ Data error: {e}")
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.info("Please check the logs or contact support")
```

---

## Summary

| Task | Main Function | Key Files |
|------|---------------|-----------|
| **Load Data** | `pd.read_parquet()` | All 4 `.parquet` files |
| **Preprocess** | `preprocess_for_prediction()` | `scaler.pkl` from `/model` |
| **Predict** | `model.predict()` | `best_lstm_model.keras` |
| **Visualize** | `plotly` or `matplotlib` | `original_data.parquet` |
| **Deploy** | Streamlit app | All above combined |

---

## Quick Reference

```python
# Complete minimal example

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Load files
data = pd.read_parquet('datasets/processed/original_data.parquet')
scaler = joblib.load('model/scaler.pkl')
model = load_model('model/best_lstm_model.keras')

# 2. Prepare input
history = data['demand_mw'].tail(168).values
X = history.reshape(-1, 1)
X_scaled = scaler.transform(X).flatten()[-168:].reshape(1, 168, 1)

# 3. Predict
y_pred_scaled = model.predict(X_scaled, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled)[0, 0]

print(f"Next hour demand: {y_pred:,.0f} MW")
```

---

**Created**: January 2026  
**For**: Streamlit Energy Demand Forecasting App  
**Dataset**: PGCB Hourly Generation Dataset (Bangladesh)
