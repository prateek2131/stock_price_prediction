"""
Data Preprocessor Module
Handles data normalization, feature engineering, and windowing for LSTM models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from typing import Tuple, List, Optional
import pickle
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses financial data for LSTM models
    """
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize DataPreprocessor
        
        Args:
            scaler_type: 'minmax' or 'standard'
        """
        self.scaler_type = scaler_type
        self.scalers = {}  # Store scalers for different columns
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(close, period=14)
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        df['BB_Middle'] = sma
        
        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(high, low, close, period=14)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Handle NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Technical indicators calculated. Shape: {df.shape}")
        return df
    
    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss!=0, 
                      out=np.full_like(avg_gain, np.nan))
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN to match original length
        rsi = np.concatenate([np.full(period, np.nan), rsi])
        return rsi
    
    @staticmethod
    def _calculate_atr(high: np.ndarray, low: np.ndarray, 
                       close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.max([tr1, tr2, tr3], axis=0)
        
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        atr = np.concatenate([np.full(period-1, np.nan), atr])
        return atr
    
    def normalize_data(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize data using specified scaler
        
        Args:
            data: DataFrame to normalize
            columns: List of columns to normalize (default: all except Date)
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized DataFrame
        """
        df = data.copy()
        
        if columns is None:
            columns = [col for col in df.columns if col != 'Date']
        
        for col in columns:
            if fit:
                if self.scaler_type == "minmax":
                    scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    scaler = StandardScaler()
                
                self.scalers[col] = scaler
                df[col] = scaler.fit_transform(df[[col]])
            else:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
                else:
                    logger.warning(f"Scaler for {col} not found. Skipping normalization.")
        
        logger.info(f"Data normalized using {self.scaler_type} scaler")
        return df
    
    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int = 60,
        target_col_idx: int = 3  # Close price index
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data array
            seq_length: Length of each sequence
            target_col_idx: Index of target column (Close price)
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :])
            y.append(data[i+seq_length, target_col_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def inverse_transform(
        self,
        data: np.ndarray,
        column: str = 'Close'
    ) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data to transform back
            column: Column name to inverse transform
            
        Returns:
            Original scale data
        """
        if column not in self.scalers:
            logger.warning(f"Scaler for {column} not found")
            return data
        
        return self.scalers[column].inverse_transform(data.reshape(-1, 1))
    
    def save_scalers(self, filepath: str) -> None:
        """Save scalers to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str) -> None:
        """Load scalers from disk"""
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"Scalers loaded from {filepath}")


class DataProcessor:
    """
    Orchestrates the full data processing pipeline
    """
    
    def __init__(self, scaler_type: str = "minmax"):
        self.preprocessor = DataPreprocessor(scaler_type)
    
    def process_data(
        self,
        data: pd.DataFrame,
        seq_length: int = 60,
        normalize_columns: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete data processing pipeline
        
        Args:
            data: Raw OHLCV data
            seq_length: Sequence length for LSTM
            normalize_columns: Columns to normalize
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Calculate technical indicators
        data = self.preprocessor.calculate_technical_indicators(data)
        
        # Normalize data
        data = self.preprocessor.normalize_data(
            data,
            columns=normalize_columns,
            fit=fit_scaler
        )
        
        # Create sequences
        data_values = data.drop('Date', axis=1).values
        X, y = self.preprocessor.create_sequences(data_values, seq_length)
        
        return X, y


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_collector import DataCollector
    from datetime import datetime, timedelta
    
    # Fetch sample data
    collector = DataCollector()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    
    data = collector.fetch_stock_data("AAPL", start_date, end_date)
    
    # Process data
    processor = DataProcessor()
    X, y = processor.process_data(data, seq_length=60)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X sample:\n{X[0]}")
