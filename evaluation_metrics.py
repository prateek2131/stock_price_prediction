"""
Evaluation Metrics Module
Calculates various metrics for model evaluation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Calculate and track evaluation metrics for time series models
    """
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def adjusted_r_squared(
        y_true: np.ndarray, y_pred: np.ndarray, n_features: int
    ) -> float:
        """Calculate adjusted R-squared"""
        r2 = EvaluationMetrics.r_squared(y_true, y_pred)
        n = len(y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions)
        """
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy * 100
    
    @staticmethod
    def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Directional Accuracy using Theil's U statistic
        """
        true_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)
        
        correct = np.sum((true_direction > 0) == (pred_direction > 0))
        return (correct / len(true_direction)) * 100
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'MAE': EvaluationMetrics.mean_absolute_error(y_true, y_pred),
            'MSE': EvaluationMetrics.mean_squared_error(y_true, y_pred),
            'RMSE': EvaluationMetrics.root_mean_squared_error(y_true, y_pred),
            'MAPE': EvaluationMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'SMAPE': EvaluationMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'R2': EvaluationMetrics.r_squared(y_true, y_pred),
            'Directional_Accuracy': EvaluationMetrics.directional_accuracy(y_true, y_pred),
            'MDA': EvaluationMetrics.mean_directional_accuracy(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Model Evaluation") -> None:
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print(f"{title:^50}")
        print("="*50)
        for metric_name, value in metrics.items():
            if 'Accuracy' in metric_name:
                print(f"{metric_name:.<35} {value:>10.2f}%")
            else:
                print(f"{metric_name:.<35} {value:>10.6f}")
        print("="*50 + "\n")


class PortfolioMetrics:
    """
    Calculate portfolio performance metrics
    """
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate daily returns"""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_cumulative_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns"""
        returns = PortfolioMetrics.calculate_returns(prices)
        return np.cumprod(1 + returns)
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (considers only downside volatility)
        
        Args:
            returns: Array of returns
            target_return: Target return threshold
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - target_return
        downside_returns = np.where(excess_returns < 0, excess_returns, 0)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            prices: Array of prices
            
        Returns:
            Maximum drawdown as percentage
        """
        cumulative = PortfolioMetrics.calculate_cumulative_returns(prices)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100
    
    @staticmethod
    def calmar_ratio(prices: np.ndarray) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Maximum Drawdown)
        
        Args:
            prices: Array of prices
            
        Returns:
            Calmar ratio
        """
        returns = PortfolioMetrics.calculate_returns(prices)
        annual_return = np.mean(returns) * 252
        max_dd = abs(PortfolioMetrics.maximum_drawdown(prices)) / 100
        
        if max_dd == 0:
            return 0
        
        return annual_return / max_dd
    
    @staticmethod
    def win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate win rate of predictions
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            
        Returns:
            Win rate as percentage
        """
        correct = np.sum((y_true[1:] > y_true[:-1]) == (y_pred[1:] > y_pred[:-1]))
        return (correct / (len(y_true) - 1)) * 100
    
    @staticmethod
    def calculate_portfolio_metrics(prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate all portfolio metrics
        
        Args:
            prices: Array of portfolio values over time
            
        Returns:
            Dictionary of portfolio metrics
        """
        returns = PortfolioMetrics.calculate_returns(prices)
        
        metrics = {
            'Total_Return': (prices[-1] / prices[0] - 1) * 100,
            'Annual_Return': np.mean(returns) * 252 * 100,
            'Volatility': np.std(returns) * np.sqrt(252) * 100,
            'Sharpe_Ratio': PortfolioMetrics.sharpe_ratio(returns),
            'Sortino_Ratio': PortfolioMetrics.sortino_ratio(returns),
            'Max_Drawdown': PortfolioMetrics.maximum_drawdown(prices),
            'Calmar_Ratio': PortfolioMetrics.calmar_ratio(prices),
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate actual and predicted prices
    y_true = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    y_pred = np.array([100.5, 101.8, 101.5, 103.2, 104.5, 104.2, 105.8, 107.9, 107.5, 108.8])
    
    # Calculate metrics
    metrics = EvaluationMetrics.calculate_all_metrics(y_true, y_pred)
    EvaluationMetrics.print_metrics(metrics, "Stock Price Prediction")
    
    # Portfolio metrics
    portfolio_metrics = PortfolioMetrics.calculate_portfolio_metrics(y_true)
    print("\nPortfolio Metrics:")
    for key, value in portfolio_metrics.items():
        print(f"{key}: {value:.4f}")
