"""
Runner script for feature engineering.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import FeatureEngineer

if __name__ == "__main__":
    print("=" * 60)
    print("SALES FORECASTING - FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    df = pd.read_csv(
        "data/processed/processed_data.csv",
        parse_dates=['Order Date', 'Ship Date']
    )
    print(f"Loaded {len(df)} rows")
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features, df_monthly = engineer.engineer_features(df)
    
    # Save
    engineer.save_features(df_features, "data/processed/features_daily.csv")
    if df_monthly is not None:
        engineer.save_features(df_monthly, "data/processed/features_monthly.csv")
    
    print("\n" + "=" * 60)
    print(" FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nDaily features shape: {df_features.shape}")
    if df_monthly is not None:
        print(f"Monthly features shape: {df_monthly.shape}")
        print(f"\nMonthly data preview:")
        print(df_monthly.head())
        print(f"\nMonthly columns:")
        for i, col in enumerate(df_monthly.columns, 1):
            print(f"  {i}. {col}")