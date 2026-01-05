"""
Visualizer Module
Provides visualization utilities for financial data and model results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialVisualizer:
    """
    Visualize financial data and model predictions
    """
    
    @staticmethod
    def plot_price_and_volume(
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 8),
        title: str = "Stock Price and Volume"
    ) -> None:
        """
        Plot stock price and trading volume
        
        Args:
            data: DataFrame with OHLC and Volume data
            figsize: Figure size
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Price
        ax1.plot(data['Close'], linewidth=2, color='blue', label='Close Price')
        ax1.fill_between(
            range(len(data)),
            data['Low'],
            data['High'],
            alpha=0.2,
            color='gray',
            label='High-Low Range'
        )
        ax1.set_ylabel('Price ($)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2.bar(range(len(data)), data['Volume'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.set_title('Trading Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('price_and_volume.png', dpi=100)
        logger.info("Price and volume plot saved")
        plt.close()
    
    @staticmethod
    def plot_technical_indicators(
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot technical indicators
        
        Args:
            data: DataFrame with technical indicators
            figsize: Figure size
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Price with Moving Averages
        axes[0].plot(data['Close'], label='Close', linewidth=2)
        if 'SMA_5' in data.columns:
            axes[0].plot(data['SMA_5'], label='SMA 5', alpha=0.7)
        if 'SMA_20' in data.columns:
            axes[0].plot(data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in data.columns:
            axes[0].plot(data['SMA_50'], label='SMA 50', alpha=0.7)
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title('Close Price with Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in data.columns:
            axes[1].plot(data['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim([0, 100])
            axes[1].set_title('Relative Strength Index (RSI)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in data.columns:
            axes[2].plot(data['MACD'], label='MACD', color='blue')
            axes[2].plot(data['Signal_Line'], label='Signal Line', color='red')
            axes[2].bar(
                range(len(data)),
                data['MACD_Histogram'],
                label='MACD Histogram',
                alpha=0.3,
                color='gray'
            )
            axes[2].set_ylabel('MACD')
            axes[2].set_title('MACD (Moving Average Convergence Divergence)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            axes[3].plot(data['Close'], label='Close', linewidth=2, color='blue')
            axes[3].plot(data['BB_Upper'], label='Upper Band', alpha=0.7, linestyle='--')
            axes[3].plot(data['BB_Middle'], label='Middle Band', alpha=0.7)
            axes[3].plot(data['BB_Lower'], label='Lower Band', alpha=0.7, linestyle='--')
            axes[3].fill_between(
                range(len(data)),
                data['BB_Upper'],
                data['BB_Lower'],
                alpha=0.1,
                color='blue'
            )
            axes[3].set_xlabel('Date')
            axes[3].set_ylabel('Price ($)')
            axes[3].set_title('Bollinger Bands')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('technical_indicators.png', dpi=100)
        logger.info("Technical indicators plot saved")
        plt.close()
    
    @staticmethod
    def plot_prediction_results(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (15, 6),
        title: str = "Prediction Results"
    ) -> None:
        """
        Plot actual vs predicted values
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            figsize: Figure size
            title: Plot title
        """
        plt.figure(figsize=figsize)
        
        plt.plot(y_test, label='Actual', linewidth=2, color='black')
        plt.plot(y_pred, label='Predicted', linewidth=2, color='red', alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=100)
        logger.info("Prediction results plot saved")
        plt.close()
    
    @staticmethod
    def plot_error_analysis(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot prediction errors
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            figsize: Figure size
        """
        errors = y_test - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Error distribution
        axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Errors over time
        axes[1].plot(errors, linewidth=1)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].fill_between(range(len(errors)), errors, alpha=0.3)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Prediction Errors Over Time')
        axes[1].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[2].scatter(y_pred, errors, alpha=0.5)
        axes[2].axhline(y=0, color='r', linestyle='--')
        axes[2].set_xlabel('Predicted Values')
        axes[2].set_ylabel('Residuals')
        axes[2].set_title('Residuals vs Predicted')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=100)
        logger.info("Error analysis plot saved")
        plt.close()
    
    @staticmethod
    def plot_interactive_price(
        data: pd.DataFrame,
        title: str = "Stock Price Interactive Chart"
    ) -> None:
        """
        Create interactive price chart using Plotly
        
        Args:
            data: DataFrame with OHLC data
            title: Chart title
        """
        # Handle date parsing if needed
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        fig = go.Figure(data=[go.Candlestick(
            x=data['Date'] if 'Date' in data.columns else range(len(data)),
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Stock Price (USD)',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        fig.write_html('interactive_chart.html')
        logger.info("Interactive chart saved as 'interactive_chart.html'")
    
    @staticmethod
    def plot_returns_distribution(
        data: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot returns distribution
        
        Args:
            data: DataFrame with return data
            figsize: Figure size
        """
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Histogram
        axes[0].hist(data['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Daily Returns Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(data['Daily_Return'].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + data['Daily_Return']).cumprod()
        axes[2].plot(cumulative_returns, linewidth=2)
        axes[2].fill_between(range(len(cumulative_returns)), cumulative_returns, alpha=0.3)
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].set_title('Cumulative Returns')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('returns_distribution.png', dpi=100)
        logger.info("Returns distribution plot saved")
        plt.close()


class ModelVisualizer:
    """
    Visualize model-specific information
    """
    
    @staticmethod
    def plot_loss_curves(
        history: dict,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot training and validation loss curves
        
        Args:
            history: Training history dictionary
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            axes[0].plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Model Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
            axes[1].plot(history.get('val_mae', []), label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MAPE
        if 'mape' in history:
            axes[2].plot(history['mape'], label='Training MAPE', linewidth=2)
            axes[2].plot(history.get('val_mape', []), label='Validation MAPE', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('MAPE (%)')
            axes[2].set_title('Mean Absolute Percentage Error')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_curves.png', dpi=100)
        logger.info("Loss curves plot saved")
        plt.close()


class ComprehensiveVisualizer:
    """
    Generate all visualizations comprehensively
    """
    
    @staticmethod
    def generate_all_plots(
        data: pd.DataFrame,
        stock_name: str = "Stock",
        save_dir: str = "results"
    ) -> None:
        """
        Generate all plots for a stock
        
        Args:
            data: DataFrame with OHLCV data and returns
            stock_name: Name of stock for titles
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Generating comprehensive visualizations for {stock_name}...")
        
        # 1. Price and Volume (like your top image)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Price plot
        ax1.plot(data['Close'], linewidth=2, color='blue', label='Close Price')
        ax1.fill_between(
            range(len(data)),
            data['Low'],
            data['High'],
            alpha=0.2,
            color='gray',
            label='High-Low Range'
        )
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.set_title(f'{stock_name} Price and Volume', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
                  for i in range(len(data))]
        ax2.bar(range(len(data)), data['Volume'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Trading Volume', fontsize=12)
        ax2.set_title('Trading Volume', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{stock_name}_price_and_volume.png', dpi=300, bbox_inches='tight')
        logger.info(f" Saved: {stock_name}_price_and_volume.png")
        plt.close()
        
        # 2. Returns Analysis (like your bottom left image)
        if 'Daily_Return' not in data.columns:
            data['Daily_Return'] = data['Close'].pct_change()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Histogram - Daily Returns Distribution
        axes[0].hist(data['Daily_Return'].dropna() * 100, bins=50, 
                     edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Daily Return (%)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Q-Q Plot - like your bottom middle image
        from scipy import stats
        stats.probplot(data['Daily_Return'].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Cumulative Returns - like your bottom right image
        cumulative_returns = (1 + data['Daily_Return']).cumprod()
        axes[2].plot(cumulative_returns, linewidth=2, color='steelblue')
        axes[2].fill_between(range(len(cumulative_returns)), cumulative_returns, 
                            alpha=0.3, color='steelblue')
        axes[2].set_xlabel('Time', fontsize=11)
        axes[2].set_ylabel('Cumulative Return', fontsize=11)
        axes[2].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{stock_name}_returns_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f" Saved: {stock_name}_returns_analysis.png")
        plt.close()
        
        # 3. Technical Indicators
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # Price with Moving Averages
        axes[0].plot(data['Close'], label='Close Price', linewidth=2, color='blue')
        if 'SMA_5' in data.columns:
            axes[0].plot(data['SMA_5'], label='SMA 5', alpha=0.7, linewidth=1)
        if 'SMA_20' in data.columns:
            axes[0].plot(data['SMA_20'], label='SMA 20', alpha=0.7, linewidth=1)
        if 'SMA_50' in data.columns:
            axes[0].plot(data['SMA_50'], label='SMA 50', alpha=0.7, linewidth=1)
        axes[0].set_ylabel('Price (₹)', fontsize=11)
        axes[0].set_title('Close Price with Moving Averages', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in data.columns:
            axes[1].plot(data['RSI'], label='RSI(14)', color='purple', linewidth=2)
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].fill_between(range(len(data)), 30, 70, alpha=0.1, color='gray')
            axes[1].set_ylabel('RSI', fontsize=11)
            axes[1].set_ylim([0, 100])
            axes[1].set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in data.columns:
            axes[2].plot(data['MACD'], label='MACD', color='blue', linewidth=2)
            if 'Signal_Line' in data.columns:
                axes[2].plot(data['Signal_Line'], label='Signal Line', color='red', linewidth=2)
            if 'MACD_Histogram' in data.columns:
                colors_macd = ['green' if x > 0 else 'red' for x in data['MACD_Histogram']]
                axes[2].bar(range(len(data)), data['MACD_Histogram'], 
                           label='MACD Histogram', alpha=0.3, color=colors_macd)
            axes[2].set_ylabel('MACD', fontsize=11)
            axes[2].set_title('MACD (Moving Average Convergence Divergence)', 
                            fontsize=12, fontweight='bold')
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            axes[3].plot(data['Close'], label='Close Price', linewidth=2, color='blue')
            axes[3].plot(data['BB_Upper'], label='Upper Band', alpha=0.7, 
                        linestyle='--', color='red', linewidth=1)
            if 'BB_Middle' in data.columns:
                axes[3].plot(data['BB_Middle'], label='Middle Band (SMA 20)', 
                            alpha=0.7, color='black', linewidth=1)
            axes[3].plot(data['BB_Lower'], label='Lower Band', alpha=0.7, 
                        linestyle='--', color='green', linewidth=1)
            axes[3].fill_between(range(len(data)), data['BB_Upper'], data['BB_Lower'],
                                alpha=0.1, color='blue')
            axes[3].set_xlabel('Date', fontsize=11)
            axes[3].set_ylabel('Price (₹)', fontsize=11)
            axes[3].set_title('Bollinger Bands', fontsize=12, fontweight='bold')
            axes[3].legend(loc='best')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{stock_name}_technical_indicators.png', dpi=300, bbox_inches='tight')
        logger.info(f" Saved: {stock_name}_technical_indicators.png")
        plt.close()
        
        # 4. Volume Analysis
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Volume with moving average
        if 'Volume_SMA' not in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        axes[0].bar(range(len(data)), data['Volume'], color='steelblue', alpha=0.6, label='Volume')
        axes[0].plot(data['Volume_SMA'], color='red', linewidth=2, label='Volume SMA(20)')
        axes[0].set_ylabel('Volume', fontsize=11)
        axes[0].set_title('Trading Volume Analysis', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Volume vs Price Change
        price_change_pct = data['Close'].pct_change() * 100
        axes[1].scatter(data['Volume'], price_change_pct, alpha=0.5, s=30, color='steelblue')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Volume', fontsize=11)
        axes[1].set_ylabel('Price Change (%)', fontsize=11)
        axes[1].set_title('Volume vs Price Change', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{stock_name}_volume_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f" Saved: {stock_name}_volume_analysis.png")
        plt.close()
        
        logger.info(f" All visualizations generated for {stock_name}!")
