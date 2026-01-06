"""
Configuration Module
Centralized configuration for MSLSTM Stock Prediction System
"""

import os
from datetime import datetime, timedelta

# Project Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# PATHS dictionary for backward compatibility
PATHS = {
    'project': PROJECT_DIR,
    'data': DATA_DIR,
    'models': MODELS_DIR,
    'results': RESULTS_DIR,
}

# Sector-wise Indian Stocks (NSE) - 8 Sectors, 40 Stocks
INDIAN_STOCKS_BY_SECTOR = {
    'Technology': [
        'TCS.NS',       # Tata Consultancy Services
        'INFY.NS',      # Infosys
        'WIPRO.NS',     # Wipro
        'HCLTECH.NS',   # HCL Technologies
        'TECHM.NS',     # Tech Mahindra
    ],
    'Banking': [
        'HDFCBANK.NS',  # HDFC Bank
        'ICICIBANK.NS', # ICICI Bank
        'SBIN.NS',      # State Bank of India
        'KOTAKBANK.NS', # Kotak Mahindra Bank
        'AXISBANK.NS',  # Axis Bank
    ],
    'Pharmaceuticals': [
        'SUNPHARMA.NS', # Sun Pharmaceutical
        'DRREDDY.NS',   # Dr. Reddy's Laboratories
        'CIPLA.NS',     # Cipla
        'DIVISLAB.NS',  # Divi's Laboratories
        'APOLLOHOSP.NS', # Apollo Hospitals
    ],
    'Automobile': [
        'MARUTI.NS',     # Maruti Suzuki
        'M&M.NS',        # Mahindra & Mahindra
        'BAJAJ-AUTO.NS', # Bajaj Auto
        'HEROMOTOCO.NS', # Hero MotoCorp
        'TATAMOTORS.NS', # Tata Motors
    ],
    'Energy': [
        'RELIANCE.NS',  # Reliance Industries
        'ONGC.NS',      # Oil and Natural Gas Corporation
        'NTPC.NS',      # NTPC Limited
        'POWERGRID.NS', # Power Grid Corporation
        'ADANIGREEN.NS', # Adani Green Energy
    ],
    'FMCG': [
        'HINDUNILVR.NS', # Hindustan Unilever
        'ITC.NS',        # ITC Limited
        'NESTLEIND.NS',  # Nestle India
        'BRITANNIA.NS',  # Britannia Industries
        'DABUR.NS',      # Dabur India
    ],
    'Metals': [
        'TATASTEEL.NS', # Tata Steel
        'HINDALCO.NS',  # Hindalco Industries
        'JSWSTEEL.NS',  # JSW Steel
        'VEDL.NS',      # Vedanta Limited
        'COALINDIA.NS', # Coal India
    ],
    'Infrastructure': [
        'LT.NS',        # Larsen & Toubro
        'ADANIENT.NS',  # Adani Enterprises
        'ADANIPORTS.NS', # Adani Ports
        'ULTRACEMCO.NS', # UltraTech Cement
        'GRASIM.NS',    # Grasim Industries
    ],
}

# Get all tickers as flat list
ALL_INDIAN_TICKERS = [ticker for sector_tickers in INDIAN_STOCKS_BY_SECTOR.values() 
                      for ticker in sector_tickers]

DATA_CONFIG = {
    'start_date': (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
    'end_date': datetime.now().strftime("%Y-%m-%d"),
    'tickers': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],  # Default tickers (Technology)
    'all_tickers': ALL_INDIAN_TICKERS,
    'sectors': INDIAN_STOCKS_BY_SECTOR,
    'index_ticker': '^NSEI',  # Nifty 50
    'currency': 'INR',
}

# Data Preprocessing Configuration
PREPROCESS_CONFIG = {
    'scaler_type': 'minmax',  
    'sequence_length': 60,  # Reduced from 90 to 60 for faster training and better pattern capture
    'target_column': 'Close',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'add_lagged_features': True,  
    'num_lags': 15,  # Increased from 10 to 15 for richer feature set
}

# LSTM Model Configuration 
LSTM_CONFIG = {
    # Architecture: Increased depth and width to capture complex patterns
    'lstm_units': 128,  
    'num_lstm_layers': 2,
    'dense_units': 128,  
    
    # Regularization: Prevent overfitting while maintaining capacity
    'dropout_rate': 0.3,  
    'recurrent_dropout': 0.2,  
    'l2_regularization': 0.0005,
    
    # Training optimization
    'learning_rate': 0.001,  
    'optimizer': 'adam',  
    'loss_function': 'mse',  
    'epochs': 100,  # Reduced from 200 for faster training on M1/M2 Macs
    'batch_size': 16,  # Increased from 8 for better M1/M2 performance
    'validation_split': 0.15,
    
    # Early stopping: Prevent overfitting
    'early_stopping_patience': 25,  
    'early_stopping_metric': 'val_loss',  
    'early_stopping_min_delta': 0.00001,  
    
    # Attention mechanism: Capture important historical periods
    'use_attention': True,  
    'attention_type': 'multi_head',  # Changed from single_head to multi_head
    'num_attention_heads': 4,  # Number of attention heads
    'attention_key_dim': 32,  # Key dimension for each head (lstm_units // 4)
    'attention_dropout': 0.1,  
    # Advanced features for Phase 1
    'use_bidirectional': False,  
    'use_layer_normalization': True,  
    'use_batch_normalization': False,  
}

# ===== MSLSTM/MSLSTMA CONFIGURATION (Novel Hybrid Architectures) =====

# MSLSTM (Multivariate Sequential Long Short-Term Memory)
MSLSTM_CONFIG = {
    # Architecture Configuration
    'num_branches': 3,  # Number of parallel LSTM branches
    'branch_units': [64, 128, 64],  # Units per branch (different scales)
    'fusion_units': 128,  # Units in fusion layer
    'dense_units': [64, 32],  # Dense layer units after fusion
    
    # Sequence Configuration
    'sequence_length': 60,
    'prediction_horizon': 1,  # Predict next day
    
    # Regularization
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.2,
    'l2_reg': 0.0001,
    
    # Training Configuration
    'learning_rate': 0.001,
    'epochs': 150,
    'batch_size': 32,
    'early_stopping_patience': 25,
    'reduce_lr_patience': 10,
    
    # Feature Groups for Multi-Branch Processing
    'feature_groups': {
        'price_features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'trend_features': ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26'],
        'momentum_features': ['RSI', 'MACD', 'Signal_Line', 'MACD_Histogram'],
        'volatility_features': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR'],
        'volume_features': ['Volume_SMA', 'Volume_Ratio'],
        'return_features': ['Daily_Return', 'Log_Return'],
    },
}

# MSLSTMA (Multivariate Sequential LSTM Autoencoder)
MSLSTMA_CONFIG = {
    **MSLSTM_CONFIG,  # Inherit base config
    
    # Autoencoder Configuration
    'encoding_dim': 32,  # Dimension of encoded representation
    'encoder_units': [128, 64],  # Encoder LSTM units
    'decoder_units': [64, 128],  # Decoder LSTM units
    'reconstruction_weight': 0.3,  # Weight for reconstruction loss
    'prediction_weight': 0.7,  # Weight for prediction loss
    
    # Denoising Configuration
    'noise_factor': 0.1,  # Noise for denoising autoencoder
    'use_denoising': True,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy', 'MDA'],
    'risk_free_rate': 0.02,
}

# Visualization Configuration (For Phase 1 Research & Analysis)
VISUALIZATION_CONFIG = {
    'figsize': (15, 6),
    'style': 'seaborn',
    'save_path': RESULTS_DIR,
    'dpi': 300,  # High resolution for papers
    'save_formats': ['png', 'pdf', 'svg'],  # Multiple formats for publication
    'plots_to_generate': [
        'price_predictions_vs_actual',
        'model_comparison_metrics',
        'directional_accuracy_comparison',
        'error_distribution',
        'training_history',
        'fund_projections',
    ],
}



# Logging Configuration (For Research Tracking)
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(PROJECT_DIR, 'logs', 'app.log'),
            'formatter': 'detailed',
        },
        'research_file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(PROJECT_DIR, 'logs', 'research.log'),
            'formatter': 'detailed',
        },
        'training_file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(PROJECT_DIR, 'logs', 'training.log'),
            'formatter': 'detailed',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
        'research': {
            'handlers': ['research_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'training': {
            'handlers': ['training_file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


# Target Assets for Phase 1
PHASE1_ASSETS = {
    'stocks': {
        'tickers': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
        'description': 'Indian IT/Tech Blue-chip Stocks',
        'currency': 'INR',
        'market': 'NSE (National Stock Exchange)',
        'focus': 'Daily price prediction and tracking',
    },
    'market_index': '^NSEI',  # Nifty 50
}


# Automation Configuration (Phase 1)
AUTOMATION_CONFIG = {
    'daily_execution': True,
    'execution_time': '16:30',  # 4:30 PM IST (after market close)
    'trading_days_only': True,
    'excel_export': True,
    'email_notifications': False,
    'log_predictions': True,
}

# Data Fetching Configuration (Holiday/Market Closure Handling)
DATA_FETCH_CONFIG = {
    'max_lookback_days': 5,  # Check up to 5 prior trading days if market is closed
    'timeout_seconds': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 2,
    'handle_holidays': True,  # Automatically fall back to prior trading day if holiday
    'supported_stocks': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
    'description': 'Configuration for robust data fetching with holiday handling for NSE stocks',
}

# Feature names for preprocessing
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20',
    'EMA_12', 'EMA_26',
    'MACD', 'Signal_Line', 'MACD_Histogram',
    'RSI',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'ATR',
    'Volume_SMA', 'Volume_Ratio',
    'Daily_Return', 'Log_Return'
]

# Technical Indicator Periods
INDICATOR_PERIODS = {
    'sma': [5, 10, 20, 50],
    'ema': [12, 26],
    'rsi': 14,
    'macd': (12, 26, 9),
    'bollinger': 20,
    'atr': 14,
}

# ===== RESEARCH OUTPUTS & REPORTING =====

# Research Results Directory Structure
RESEARCH_OUTPUT_CONFIG = {
    'results_dir': RESULTS_DIR,
    'reports_dir': os.path.join(RESULTS_DIR, 'reports'),
    'visualizations_dir': os.path.join(RESULTS_DIR, 'visualizations'),
    'models_dir': MODELS_DIR,
    'logs_dir': os.path.join(PROJECT_DIR, 'logs'),
    'data_dir': DATA_DIR,
}

# Create research output directories
for dir_path in RESEARCH_OUTPUT_CONFIG.values():
    if isinstance(dir_path, str):
        os.makedirs(dir_path, exist_ok=True)



if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1 RESEARCH-FOCUSED CONFIGURATION LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nProject Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\n Phase 1 Target Stocks: {', '.join(PHASE1_ASSETS['stocks']['tickers'])}")
    print(f" Market Focus: {PHASE1_ASSETS['stocks']['focus']}")
    print(f"\nResearch Components:")
    print("\n" + "=" * 70)
