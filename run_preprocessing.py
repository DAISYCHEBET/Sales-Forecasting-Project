"""
Runner script for data preprocessing.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import DataPreprocessor

if __name__ == "__main__":
    print("=" * 60)
    print("SALES FORECASTING - DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    df_processed, df_original = preprocessor.preprocess()
    preprocessor.save_processed_data(df_processed)
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"\nFirst few rows:")
    print(df_processed.head())
    print(f"\nColumns: {list(df_processed.columns)}")