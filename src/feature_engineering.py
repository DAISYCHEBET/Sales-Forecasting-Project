"""
Feature engineering module for time series forecasting.
Creates time-based features, lag features, and rolling statistics.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.utils import logger, load_config


class FeatureEngineer:
    """
    Handles feature engineering for time series data including:
    - Time-based features (month, day, week, etc.)
    - Lag features
    - Rolling statistics
    - Business-specific features
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the FeatureEngineer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
        self.training_config = self.config['training']
        logger.info("FeatureEngineer initialized")
    
    def create_time_features(self, df: pd.DataFrame, date_column: str = 'Order Date') -> pd.DataFrame:
        """
        Extract time-based features from date column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
        
        Returns:
            DataFrame with time features added
        """
        logger.info(f"Creating time features from '{date_column}'")
        
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Week'] = df[date_column].dt.isocalendar().week.astype(int)
        df['Quarter'] = df[date_column].dt.quarter
        df['DayOfYear'] = df[date_column].dt.dayofyear
        
        # Is weekend?
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Season (Northern Hemisphere)
        df['Season'] = df['Month'].apply(self._get_season)
        
        logger.info(f"Created {9} time-based features")
        return df
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delivery-related features.
        
        Args:
            df: Input DataFrame with Order Date and Ship Date
        
        Returns:
            DataFrame with delivery features
        """
        logger.info("Creating delivery features")
        
        # Calculate delivery time in days
        df['DeliveryTime'] = (df['Ship Date'] - df['Order Date']).dt.days
        
        # Categorize delivery speed
        def categorize_delivery(days):
            if days == 0:
                return 'Same Day'
            elif days <= 2:
                return 'Fast'
            elif days <= 5:
                return 'Normal'
            else:
                return 'Slow'
        
        df['DeliverySpeed'] = df['DeliveryTime'].apply(categorize_delivery)
        
        logger.info("Created 2 delivery features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Sales', 
                           lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series.
        
        Args:
            df: Input DataFrame (must be sorted by date)
            target_col: Column to create lags for
            lags: List of lag periods. If None, uses config values
        
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = self.feature_config['lag_features']
        
        logger.info(f"Creating lag features for '{target_col}' with lags: {lags}")
        
        for lag in lags:
            df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'Sales',
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame (must be sorted by date)
            target_col: Column to calculate rolling statistics for
            windows: List of window sizes. If None, uses config values
        
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = self.feature_config['rolling_windows']
        
        logger.info(f"Creating rolling features for '{target_col}' with windows: {windows}")
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_RollingMean_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{target_col}_RollingStd_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{target_col}_RollingMin_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).min()
            )
            df[f'{target_col}_RollingMax_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).max()
            )
        
        features_created = len(windows) * 4  # mean, std, min, max per window
        logger.info(f"Created {features_created} rolling features")
        return df
    
    def aggregate_to_monthly(self, df: pd.DataFrame, date_column: str = 'Order Date',
                            target_col: str = 'Sales') -> pd.DataFrame:
        """
        Aggregate data to monthly level for time series forecasting.
        
        Args:
            df: Input DataFrame
            date_column: Date column to use for aggregation
            target_col: Target column to aggregate
        
        Returns:
            Monthly aggregated DataFrame
        """
        logger.info(f"Aggregating data to monthly level")
        
        # Set date as index
        df_monthly = df.set_index(date_column)
        
        # Resample to monthly frequency and sum sales
        monthly_sales = df_monthly[target_col].resample('ME').sum().reset_index()
        monthly_sales.columns = ['Date', 'Sales']
        
        # Add month and year
        monthly_sales['Year'] = monthly_sales['Date'].dt.year
        monthly_sales['Month'] = monthly_sales['Date'].dt.month
        
        logger.info(f"Monthly data shape: {monthly_sales.shape}")
        return monthly_sales
    
    def engineer_features(self, df: pd.DataFrame, for_monthly_forecast: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame (preprocessed)
            for_monthly_forecast: If True, also create monthly aggregated data
        
        Returns:
            Tuple of (df_with_features, monthly_df) or (df_with_features, None)
        """
        logger.info("=" * 50)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 50)
        
        original_shape = df.shape
        
        # Create time-based features
        df = self.create_time_features(df)
        
        # Create delivery features
        if 'Ship Date' in df.columns:
            df = self.create_delivery_features(df)
        
        # For daily-level features (optional - can be memory intensive)
        # Uncomment if you want lag and rolling features at daily level
        # df = self.create_lag_features(df)
        # df = self.create_rolling_features(df)
        
        logger.info(f"Features added. Shape: {original_shape} → {df.shape}")
        
        # Create monthly aggregated data for forecasting
        monthly_df = None
        if for_monthly_forecast:
            monthly_df = self.aggregate_to_monthly(df)
            
            # Add lag and rolling features to monthly data
            monthly_df = self.create_lag_features(
                monthly_df, 
                target_col='Sales',
                lags=[1, 3, 6, 12]
            )
            monthly_df = self.create_rolling_features(
                monthly_df,
                target_col='Sales',
                windows=[3, 6, 12]
            )
        
        logger.info("=" * 50)
        logger.info("Feature engineering completed successfully")
        if monthly_df is not None:
            logger.info(f"Monthly data shape: {monthly_df.shape}")
        logger.info("=" * 50)
        
        return df, monthly_df
    
    def save_features(self, df: pd.DataFrame, output_path: str = None) -> None:
        """
        Save engineered features to CSV.
        
        Args:
            df: DataFrame with features
            output_path: Path to save the file
        """
        if output_path is None:
            output_path = "data/processed/features.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Test the feature engineering
    from src.data_preprocessing import DataPreprocessor
    
    # Load preprocessed data
    preprocessor = DataPreprocessor()
    df = pd.read_csv("data/processed/processed_data.csv", parse_dates=['Order Date', 'Ship Date'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features, df_monthly = engineer.engineer_features(df)
    
    # Save
    engineer.save_features(df_features, "data/processed/features_daily.csv")
    if df_monthly is not None:
        engineer.save_features(df_monthly, "data/processed/features_monthly.csv")
    
    print("\n Feature engineering completed successfully!")
    print(f"\nDaily features shape: {df_features.shape}")
    if df_monthly is not None:
        print(f"Monthly features shape: {df_monthly.shape}")
        print(f"\nMonthly data preview:\n{df_monthly.head()}")
        print(f"\nMonthly columns: {list(df_monthly.columns)}")