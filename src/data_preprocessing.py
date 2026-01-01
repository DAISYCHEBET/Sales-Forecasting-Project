"""
Data preprocessing module for sales time series forecasting.
Handles data loading, cleaning, and initial transformations.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils import logger, load_config


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Loading raw data
    - Handling missing values
    - Removing duplicates
    - Data type conversions
    - Initial data validation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file. If None, uses config path.
        
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = self.data_config['raw_path']
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # Log missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Drop Postal Code column (has missing values and not useful)
        if 'Postal Code' in df.columns:
            df = df.drop(columns=['Postal Code'], errors='ignore')
            logger.info("Dropped 'Postal Code' column")
        
        return df
    
    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove identifier columns that won't help the model.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with unnecessary columns removed
        """
        columns_to_drop = [
            'Order ID',
            'Customer ID',
            'Customer Name',
            'Product ID',
            'Product Name'
        ]
        
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop, errors='ignore')
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
        
        return df
    
    def convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to datetime format.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with converted date columns
        """
        logger.info("Converting date columns")
        
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True)
                logger.info(f"Converted '{col}' to datetime")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_shape = df.shape
        df = df.drop_duplicates()
        final_shape = df.shape
        
        duplicates_removed = initial_shape[0] - final_shape[0]
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def apply_log_transformation(self, df: pd.DataFrame, column: str = 'Sales') -> pd.DataFrame:
        """
        Apply log transformation to handle skewed data.
        
        Args:
            df: Input DataFrame
            column: Column name to transform
        
        Returns:
            DataFrame with log-transformed column
        """
        skewness = df[column].skew()
        logger.info(f"Original skewness of '{column}': {skewness:.2f}")
        
        if skewness > 1:  # Positively skewed
            df[f'Log_{column}'] = np.log1p(df[column])
            new_skewness = df[f'Log_{column}'].skew()
            logger.info(f"Applied log transformation. New skewness: {new_skewness:.2f}")
        
        return df
    
    def sort_by_date(self, df: pd.DataFrame, date_column: str = 'Order Date') -> pd.DataFrame:
        """
        Sort DataFrame by date (critical for time series).
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column to sort by
        
        Returns:
            Sorted DataFrame
        """
        df = df.sort_values(by=date_column)
        df = df.reset_index(drop=True)
        logger.info(f"Data sorted by '{date_column}'")
        
        return df
    
    def preprocess(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame. If None, loads from config path.
        
        Returns:
            Tuple of (processed_df, original_df)
        """
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 50)
        
        # Load data if not provided
        if df is None:
            df = self.load_data()
        
        # Keep original copy
        df_original = df.copy()
        
        # Execute preprocessing steps
        df = self.handle_missing_values(df)
        df = self.remove_unnecessary_columns(df)
        df = self.convert_date_columns(df)
        df = self.remove_duplicates(df)
        df = self.apply_log_transformation(df)
        df = self.sort_by_date(df)
        
        logger.info("=" * 50)
        logger.info("Preprocessing completed successfully")
        logger.info(f"Final shape: {df.shape}")
        logger.info("=" * 50)
        
        return df, df_original
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = None) -> None:
        """
        Save processed data to CSV.
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the file. If None, uses config path.
        """
        if output_path is None:
            output_path = self.data_config['processed_path']
        
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = DataPreprocessor()
    df_processed, df_original = preprocessor.preprocess()
    preprocessor.save_processed_data(df_processed)
    print("\n Preprocessing completed successfully!")
    print(f"Processed data shape: {df_processed.shape}")
    print(f"\nFirst few rows:\n{df_processed.head()}")