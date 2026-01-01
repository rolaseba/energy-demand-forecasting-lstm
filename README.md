# LSTM Energy Demand Prediction Model

## 📈 Business Overview

### Business Problem
Bangladesh's Power Grid Company (PGCB) operates the national transmission system managing ~25,700 MW of installed capacity. Accurate electricity demand forecasting is critical for:
- **Grid Stability**: Preventing blackouts and overload conditions
- **Resource Allocation**: Optimizing generation across coal, gas, hydro, and renewable sources
- **Cost Optimization**: Matching supply with demand to minimize wasteful generation
- **Planning**: Strategic infrastructure investment and capacity expansion decisions

### Value Proposition
This LSTM-based forecasting model predicts hourly electricity demand patterns with high accuracy, enabling:
- **Proactive Resource Management**: Deploy generation capacity before demand peaks
- **Reduced Operational Costs**: Minimize inefficient generation and fuel waste
- **Improved Reliability**: Early warning system for potential supply-demand imbalances
- **Data-Driven Decision Making**: Replace rule-based scheduling with predictive intelligence

### Business Impact
- **Accuracy Focus**: Multi-step ahead forecasting (24-hour horizon)
- **Operational Window**: Hourly granularity for real-time grid management
- **Scalability**: Foundation for national-level and regional forecasting models
- **Integration Ready**: Exportable model architecture for production deployment

---

## 🔬 Technical Implementation

### Dataset

**Source**: [PGCB Hourly Generation Dataset (Bangladesh)](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))

**Data Characteristics**:
- **Temporal Coverage**: Hourly records of electricity generation and demand
- **Geographic Scope**: National grid data from Power Grid Company of Bangladesh
- **Features**: 
  - Electricity demand patterns
  - Generation by source (coal, gas, hydro, renewable)
  - Grid load and loadshedding events
  - Seasonal and cyclical patterns

**Data Quality Handling**:
- Missing value imputation and validation
- Outlier detection and treatment
- Seasonal decomposition for pattern identification
- Normalization for neural network training

### Architecture

#### Model Type: LSTM (Long Short-Term Memory)
**Why LSTM?**
- Captures long-range temporal dependencies in time series
- Handles vanishing gradient problem in deep recurrent networks
- Excellent for multi-step ahead forecasting
- Proven track record in energy demand prediction

#### Network Configuration
```
Input: Sequence of 30 hourly observations (7.5 days)
  ↓
LSTM Layer 1: 50 units + return_sequences=True
  → Dropout: 0.2
  ↓
LSTM Layer 2: 50 units + return_sequences=True
  → Dropout: 0.2
  ↓
LSTM Layer 3: 25 units
  → Dropout: 0.2
  ↓
Dense Layer: 25 units (ReLU activation)
  ↓
Output Layer: 1 unit (single-step prediction)
```

#### Key Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sequence Length | 30 hours | Captures 1.25-day patterns |
| LSTM Units (L1-L2) | 50 | Sufficient complexity for pattern capture |
| LSTM Units (L3) | 25 | Progressive dimensionality reduction |
| Dropout Rate | 0.2 | Regularization without over-dampening |
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| Learning Rate | 0.001 | Stable, long-term convergence |
| Loss Function | MSE | Appropriate for regression, penalizes large errors |

#### Training Configuration
- **Early Stopping**: Monitor validation loss, patience=15 epochs
- **Model Checkpoint**: Save best model based on validation MAE
- **Learning Rate Reduction**: Reduce on plateau (factor=0.5, patience=5)
- **Batch Size**: 32 samples
- **Epochs**: Up to 100 (with early stopping)

### Data Pipeline

#### 1. Temporal Train-Test Split
- **Non-random split**: Respects temporal order (no data leakage)
- **Train Set**: Historical 80% of data
- **Test Set**: Most recent 20% of data
- **Rationale**: Simulates real-world scenario where future data is unknown

#### 2. Sequence Creation
- **Window Size**: 30 timesteps (sequence length)
- **Stride**: 1 (creates overlapping sequences)
- **Output**: Next timestep prediction (t+1)
- **Format**: 3D arrays compatible with LSTM input

#### 3. Feature Normalization
- **Method**: MinMax scaling (0-1 range)
- **Fit on**: Training data only (prevents test contamination)
- **Apply to**: Train and test sets identically

### Model Files

| File | Purpose |
|------|---------|
| `notebook-energy.ipynb` | Complete end-to-end pipeline with visualizations |
| `notebook-energy.py` | Python script version of the notebook |
| `notebook-test.ipynb` | Testing and validation experiments |
| `best_lstm_model.keras` | Production model (best validation performance) |
| `check_gpu_status.py` | GPU availability verification for training |

### Performance Metrics

- **MAE (Mean Absolute Error)**: Average absolute prediction error in MW
- **MSE (Mean Squared Error)**: Penalizes larger errors more heavily
- **RMSE (Root Mean Squared Error)**: Same units as target variable (MW)
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error for interpretability

### Validation Strategy

- **Time Series Cross-Validation**: Rolling window approach
- **Holdout Test Set**: Final evaluation on unseen recent data
- **No Data Leakage**: Strict temporal ordering maintained
- **Residual Analysis**: Check for autocorrelation and patterns in errors

---

## 🚀 Quick Start

### Prerequisites
```bash
# Create and activate conda environment
conda create -n ml_py3132_env python=3.13.2
conda activate ml_py3132_env

# Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn plotly scikit-learn openpyxl
```

### Run the Model
```bash
# Execute the notebook for full pipeline
jupyter notebook notebook-energy.ipynb

# Or run the Python version
python notebook-energy.py
```

### GPU Support
Verify GPU availability:
```bash
python check_gpu_status.py
```

---

## 📊 Results & Insights

The trained LSTM model demonstrates:
- Strong correlation with actual demand patterns
- Excellent capture of diurnal (daily) cycles
- Reasonable generalization to test period
- Interpretable error patterns for further optimization

### Next Steps for Production
1. **Ensemble Methods**: Combine LSTM with other models (XGBoost, Prophet)
2. **External Features**: Include temperature, holidays, special events
3. **Multivariate Output**: Predict demand by region/generation source
4. **Real-time Pipeline**: Deploy as API for hourly predictions
5. **Continuous Learning**: Retrain on recent data monthly

---

## 📁 Project Structure

```
lstm-prediction-model/
├── notebook-energy.ipynb          # Main analysis notebook
├── notebook-energy.py             # Executable Python version
├── notebook-test.ipynb            # Validation experiments
├── check_gpu_status.py            # GPU verification script
├── datasets/
│   └── PGCB_date_power_demand.xlsx  # Raw energy data
├── model/
│   └── best_lstm_model.keras      # Trained model artifact
└── README.md                       # This file
```

---

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Science Utilities**: Scikit-learn
- **Environment**: Conda, Jupyter

---

## 📝 Portfolio Context

This project demonstrates:
- ✅ End-to-end ML pipeline from data to production model
- ✅ Time series forecasting expertise with LSTM networks
- ✅ Proper data handling (temporal splits, no leakage)
- ✅ Production-ready model serialization
- ✅ Business acumen (grid operations understanding)
- ✅ Reproducible research practices

---

## 📚 References

- [PGCB Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Keras LSTM Documentation: https://keras.io/api/layers/recurrent_layers/lstm/

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: [Sebastián Rolando](https://github.com/rolaseba)  
**Last Updated**: January 2026
