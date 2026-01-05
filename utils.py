"""
Utility Functions
Helper functions for common tasks
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DataUtils:
    """Data utility functions"""
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str) -> None:
        """Save object as pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load object from pickle file"""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> None:
        """Save dictionary as JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load dictionary from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    
    @staticmethod
    def data_to_sequences(
        data: np.ndarray,
        seq_length: int = 60,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time series data to sequences
        
        Args:
            data: Input data (samples, features)
            seq_length: Length of sequences
            stride: Step between sequences
            
        Returns:
            Tuple of (X, y)
        """
        X, y = [], []
        
        for i in range(0, len(data) - seq_length, stride):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  # Assume first column is target
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def normalize_data(
        data: np.ndarray,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize data
        
        Args:
            data: Data to normalize
            method: 'minmax' or 'standard'
            feature_range: Range for minmax scaling
            
        Returns:
            Normalized data and scaling parameters
        """
        params = {}
        
        if method == 'minmax':
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            scaled = (data - min_val) / (max_val - min_val)
            scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
            params = {'min': min_val, 'max': max_val}
        
        else:  # standard
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            scaled = (data - mean) / std
            params = {'mean': mean, 'std': std}
        
        return scaled, params
    
    @staticmethod
    def denormalize_data(
        data: np.ndarray,
        params: Dict,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Denormalize data back to original scale
        
        Args:
            data: Normalized data
            params: Scaling parameters from normalization
            method: 'minmax' or 'standard'
            feature_range: Range used in minmax scaling
            
        Returns:
            Denormalized data
        """
        if method == 'minmax':
            data_unscaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
            return data_unscaled * (params['max'] - params['min']) + params['min']
        else:
            return data * params['std'] + params['mean']


class MetricsUtils:
    """Metrics utility functions"""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate simple returns"""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns"""
        return np.diff(np.log(prices))
    
    @staticmethod
    def calculate_cumulative_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns"""
        returns = MetricsUtils.calculate_returns(prices)
        return np.cumprod(1 + returns)
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, periods: int = 252) -> float:
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(periods)
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - (risk_free_rate / 252)
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_drawdown(prices: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate maximum drawdown and drawdown series
        
        Returns:
            Tuple of (max_drawdown, drawdown_series)
        """
        cumulative = np.cumprod(1 + MetricsUtils.calculate_returns(prices))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown), drawdown


class ModelUtils:
    """Model utility functions"""
    
    @staticmethod
    def create_directory(path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory ensured: {path}")
    
    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> str:
        """Get latest model checkpoint"""
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
        if not files:
            return None
        return os.path.join(
            checkpoint_dir,
            max(files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
        )
    
    @staticmethod
    def format_model_summary(model) -> str:
        """Get model summary as string"""
        import io
        import sys
        
        buffer = io.StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()


class FileUtils:
    """File utility functions"""
    
    @staticmethod
    def ensure_directories(directories: List[str]) -> None:
        """Ensure multiple directories exist"""
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    @staticmethod
    def get_file_size(filepath: str) -> str:
        """Get human-readable file size"""
        size_bytes = os.path.getsize(filepath)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def create_backup(filepath: str) -> str:
        """Create backup of a file"""
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        
        import shutil
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
        return backup_path


class LoggingUtils:
    """Logging utility functions"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: str = None,
        level: str = 'INFO'
    ) -> logging.Logger:
        """
        Setup logger with file and console handlers
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(ch)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(fh)
        
        return logger
    
    @staticmethod
    def log_execution_time(func):
        """Decorator to log function execution time"""
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} executed in {duration:.2f} seconds")
            return result
        
        return wrapper


class ValidationUtils:
    """Data validation utility functions"""
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality and return report
        
        Args:
            data: DataFrame to check
            
        Returns:
            Quality report dictionary
        """
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'numeric_stats': data.describe().to_dict(),
        }
        
        return report
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> bool:
        """Validate price data integrity"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check columns
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return False
        
        # Check OHLC relationships
        if (data['High'] < data['Low']).any():
            logger.error("High < Low detected")
            return False
        
        if (data['High'] < data['Close']).any():
            logger.error("High < Close detected")
            return False
        
        if (data['Low'] > data['Close']).any():
            logger.error("Low > Close detected")
            return False
        
        # Check for negative values
        if (data[required_columns] < 0).any().any():
            logger.error("Negative values detected in price data")
            return False
        
        logger.info("Price data validation passed")
        return True


if __name__ == "__main__":
    # Example usage
    logger.info("Utility functions loaded")
