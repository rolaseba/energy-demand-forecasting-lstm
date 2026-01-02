"""
================================================================================
PREPARE DATA FOR STREAMLIT APP
================================================================================

This script prepares all necessary data files for the Streamlit forecasting app.
It exports processed data in Parquet format for efficient loading and inference.

Files created:
- datasets/processed/original_data.parquet     → Full cleaned time series data
- datasets/processed/train_test_data.parquet   → Train/Test split information
- datasets/processed/test_predictions.parquet  → Test set with predictions
- datasets/processed/metadata.parquet          → Configuration & metadata
- datasets/processed/scaler_stats.parquet      → Scaler statistics for reference

Dependencies:
- pandas, numpy, scikit-learn, tensorflow, joblib
- model/best_lstm_model.keras (trained model)
- model/scaler.pkl (fitted scaler)

================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model

# ============================================================================
# CONFIGURATION - DYNAMIC PATH RESOLUTION
# ============================================================================

# Get the root project directory (parent of deployment folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Model and scaler paths (relative to project root)
MODEL_PATH = PROJECT_ROOT / 'model' / 'best_lstm_model.keras'
SCALER_PATH = PROJECT_ROOT / 'model' / 'scaler.pkl'
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'processed'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
EXCEL_FILE = DATASETS_DIR / 'PGCB_date_power_demand.xlsx'

# Model parameters (should match notebook)
SEQUENCE_LENGTH = 168  # 1 week
SCALER_TYPE = 'StandardScaler'
SCALE_DATA = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir(output_dir) -> None:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir.resolve()}")


def load_model_and_scaler(model_path, scaler_path) -> tuple:
    """Load the trained LSTM model and scaler."""
    print("\n" + "="*70)
    print("LOADING MODEL AND SCALER")
    print("="*70)
    
    # Convert to Path objects
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path.resolve()}")
    
    model = load_model(str(model_path))
    print(f"✓ Model loaded from: {model_path.resolve()}")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Load scaler
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path.resolve()}")
    
    scaler = joblib.load(str(scaler_path))
    print(f"✓ Scaler loaded from: {scaler_path.resolve()}")
    print(f"  Scaler type: {type(scaler).__name__}")
    
    return model, scaler


def prepare_test_predictions(
    model,
    scaler,
    X_test_seq_scaled: np.ndarray,
    y_test_seq_scaled: np.ndarray,
    sequence_length: int
) -> pd.DataFrame:
    """
    Generate predictions for the test set and prepare results DataFrame.
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    scaler : sklearn scaler
        Fitted scaler for inverse transformation
    X_test_seq_scaled : np.ndarray
        Scaled test input sequences
    y_test_seq_scaled : np.ndarray
        Scaled test target values
    sequence_length : int
        Length of input sequences
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions and original values
    """
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    # Generate predictions in scaled space
    y_pred_scaled = model.predict(X_test_seq_scaled, verbose=0).flatten()
    y_test_scaled_flat = y_test_seq_scaled.flatten()
    
    print(f"✓ Predictions generated")
    print(f"  Test samples: {len(y_pred_scaled):,}")
    print(f"  Sequence length: {sequence_length}")
    
    # Inverse transform to original scale
    y_pred_original = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_test_original = scaler.inverse_transform(
        y_test_scaled_flat.reshape(-1, 1)
    ).flatten()
    
    print(f"\nOriginal scale ranges:")
    print(f"  Predictions: [{y_pred_original.min():,.2f}, {y_pred_original.max():,.2f}] MW")
    print(f"  Actuals:     [{y_test_original.min():,.2f}, {y_test_original.max():,.2f}] MW")
    
    # Calculate errors
    absolute_error = np.abs(y_test_original - y_pred_original)
    percentage_error = (absolute_error / (np.abs(y_test_original) + 1e-10)) * 100
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'actual_demand_mw': y_test_original,
        'predicted_demand_mw': y_pred_original,
        'absolute_error_mw': absolute_error,
        'percentage_error': percentage_error,
        'index_in_test_set': np.arange(len(y_pred_original))
    })
    
    return results_df


def get_scaler_statistics(scaler) -> dict:
    """Extract statistics from the fitted scaler."""
    stats = {
        'scaler_type': type(scaler).__name__,
        'scale_data': SCALE_DATA,
        'sequence_length': SEQUENCE_LENGTH
    }
    
    # MinMaxScaler stats
    if hasattr(scaler, 'data_min_'):
        stats['data_min'] = float(scaler.data_min_[0])
        stats['data_max'] = float(scaler.data_max_[0])
        stats['feature_range'] = scaler.feature_range
    
    # StandardScaler stats
    if hasattr(scaler, 'mean_'):
        stats['mean'] = float(scaler.mean_[0])
        stats['scale'] = float(scaler.scale_[0])  # This is std deviation
    
    return stats


def save_metadata(
    output_dir: str,
    model_path: str,
    scaler_path: str,
    test_predictions: pd.DataFrame,
    scaler_stats: dict,
    sequence_length: int
) -> None:
    """Save metadata and configuration information."""
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'model_path': model_path,
        'scaler_path': scaler_path,
        'sequence_length': sequence_length,
        'scaler_statistics': scaler_stats,
        'test_predictions_stats': {
            'num_samples': len(test_predictions),
            'mean_actual_demand_mw': float(test_predictions['actual_demand_mw'].mean()),
            'mean_predicted_demand_mw': float(test_predictions['predicted_demand_mw'].mean()),
            'mean_absolute_error_mw': float(test_predictions['absolute_error_mw'].mean()),
            'mean_percentage_error': float(test_predictions['percentage_error'].mean()),
            'min_actual_mw': float(test_predictions['actual_demand_mw'].min()),
            'max_actual_mw': float(test_predictions['actual_demand_mw'].max()),
            'min_predicted_mw': float(test_predictions['predicted_demand_mw'].min()),
            'max_predicted_mw': float(test_predictions['predicted_demand_mw'].max()),
        },
        'description': {
            'original_data': 'Full cleaned time series (demand_mw)',
            'train_test_data': 'Train/Test split information with indices',
            'test_predictions': 'Test set predictions with error metrics',
            'metadata': 'Configuration and statistics (JSON format)',
            'scaler_stats': 'Scaler parameters and statistics'
        }
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(output_dir, 'metadata.parquet')
    metadata_df.to_parquet(metadata_path, index=False)
    
    print(f"✓ Metadata saved to: {metadata_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("STREAMLIT DATA PREPARATION SCRIPT")
    print("="*70)
    
    try:
        # Step 1: Create output directory
        ensure_output_dir(OUTPUT_DIR)
        
        # Step 2: Load model and scaler
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        # Step 3: Load original data from notebook's variables
        # This should be imported from the notebook or loaded from raw Excel file
        print("\n" + "="*70)
        print("PREPARING ORIGINAL DATA")
        print("="*70)
        
        # Load from Excel (same as notebook)
        if not EXCEL_FILE.exists():
            raise FileNotFoundError(f"Excel file not found at: {EXCEL_FILE.resolve()}")
        
        df = pd.read_excel(str(EXCEL_FILE))
        
        # Clean outliers (same preprocessing as notebook)
        df = df[df["generation_mw"] < 60_000_000]
        
        # Get the demand column
        original_data = pd.DataFrame({
            'datetime': df['datetime'] if 'datetime' in df.columns else pd.date_range(start='2020-01-01', periods=len(df), freq='H'),
            'demand_mw': df['demand_mw'],
            'generation_mw': df['generation_mw']
        })
        
        print(f"✓ Original data loaded")
        print(f"  Shape: {original_data.shape}")
        print(f"  Date range: {original_data['datetime'].min()} to {original_data['datetime'].max()}")
        print(f"  Demand range: [{original_data['demand_mw'].min():,.2f}, {original_data['demand_mw'].max():,.2f}] MW")
        
        # Save original data
        original_data_path = OUTPUT_DIR / 'original_data.parquet'
        original_data.to_parquet(str(original_data_path), index=False)
        print(f"✓ Original data saved to: {original_data_path.resolve()}")
        
        # Step 4: Prepare training data (for reference)
        print("\n" + "="*70)
        print("PREPARING TRAINING/TEST SPLIT INFORMATION")
        print("="*70)
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        # Extract target
        y_series = original_data['demand_mw'].copy()
        
        # Split (same logic as notebook)
        test_size = 0.20
        split_idx = int(len(y_series) * (1 - test_size))
        
        train_test_info = pd.DataFrame({
            'index': np.arange(len(y_series)),
            'demand_mw': y_series.values,
            'data_type': ['train'] * split_idx + ['test'] * (len(y_series) - split_idx),
            'is_in_training_set': [1] * split_idx + [0] * (len(y_series) - split_idx)
        })
        
        if 'datetime' in original_data.columns:
            train_test_info['datetime'] = original_data['datetime'].values
        
        train_test_path = OUTPUT_DIR / 'train_test_data.parquet'
        train_test_info.to_parquet(str(train_test_path), index=False)
        print(f"✓ Train/Test split info saved to: {train_test_path.resolve()}")
        print(f"  Training samples: {(train_test_info['data_type'] == 'train').sum():,}")
        print(f"  Test samples: {(train_test_info['data_type'] == 'test').sum():,}")
        
        # Step 5: Prepare test predictions (this requires the actual test data from the notebook)
        # For complete setup, you need to also save X_test_seq_scaled and y_test_seq_scaled from notebook
        
        print("\n" + "="*70)
        print("SCALER STATISTICS")
        print("="*70)
        scaler_stats = get_scaler_statistics(scaler)
        print(f"✓ Scaler type: {scaler_stats['scaler_type']}")
        for key, value in scaler_stats.items():
            if key != 'scaler_type':
                print(f"  {key}: {value}")
        
        # Save scaler stats
        scaler_stats_df = pd.DataFrame([scaler_stats])
        scaler_stats_path = OUTPUT_DIR / 'scaler_stats.parquet'
        scaler_stats_df.to_parquet(str(scaler_stats_path), index=False)
        print(f"✓ Scaler stats saved to: {scaler_stats_path.resolve()}")
        
        # Step 6: Save metadata
        metadata_obj = {
            'creation_date': datetime.now().isoformat(),
            'model_path': str(MODEL_PATH),
            'scaler_path': str(SCALER_PATH),
            'sequence_length': SEQUENCE_LENGTH,
            'original_data_shape': original_data.shape,
            'scaler_statistics': scaler_stats,
            'files_created': [
                'original_data.parquet',
                'train_test_data.parquet',
                'scaler_stats.parquet'
            ]
        }
        
        metadata_df = pd.DataFrame([metadata_obj])
        metadata_path = OUTPUT_DIR / 'metadata.parquet'
        metadata_df.to_parquet(str(metadata_path), index=False)
        print(f"✓ Metadata saved to: {metadata_path.resolve()}")
        
        # Step 7: Summary
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*70)
        print("\nFiles created in datasets/processed/:")
        print("  1. original_data.parquet        → Full cleaned time series")
        print("  2. train_test_data.parquet      → Train/test split info")
        print("  3. scaler_stats.parquet         → Scaler statistics")
        print("  4. metadata.parquet             → Configuration & metadata")
        print("\n⚠️  NOTE: For complete setup with predictions:")
        print("  - Add test predictions from notebook (X_test_seq_scaled, y_test_seq_scaled)")
        print("  - Run prepare_test_predictions() with those data")
        print("  - Save to: test_predictions.parquet")
        print("\n" + "="*70)
        
        # Step 8: Example usage information
        print("\nUSAGE IN STREAMLIT APP:")
        print("-" * 70)
        print("""
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load data
original_data = pd.read_parquet('datasets/processed/original_data.parquet')
train_test_info = pd.read_parquet('datasets/processed/train_test_data.parquet')
scaler_stats = pd.read_parquet('datasets/processed/scaler_stats.parquet')

# Load model and scaler
model = load_model('model/best_lstm_model.keras')
scaler = joblib.load('model/scaler.pkl')

# Make predictions
# Use original_data for inference
# Scale input: scaled = scaler.transform(data)
# Predict: y_pred_scaled = model.predict(sequences)
# Inverse: y_pred = scaler.inverse_transform(y_pred_scaled)
        """)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
