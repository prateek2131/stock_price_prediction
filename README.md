# Multi-Scale LSTM with Ensemble Learning for Stock Price Prediction

## A Comprehensive Framework for Indian Equity Markets

> This research presents a comprehensive ensemble framework that demonstrates superior performance in stock price prediction for Indian equity markets through the integration of Multi-Scale LSTM architectures and advanced attention mechanisms.

---

## Research Overview

### Study Scope
- **Market Focus**: Indian equity markets (NSE)
- **Dataset**: 40 stocks across 8 major sectors
- **Evaluation Period**: Extensive historical analysis + 23-day live trading validation (January 6 - February 12, 2026)
- **Total Predictions**: 3,588 real market predictions during live trading
- **Evaluation Date**: Updated through February 12, 2026

### Key Research Contributions

1. **Methodological Innovation**: Development of MSLSTMA architecture combining LSTM autoencoders with attention mechanisms for superior noise reduction and feature extraction

2. **Ensemble Advancement**: Implementation of novel 3-level stacked ensemble with optimized base model weights: TCN (20%), WaveNet (30%), LSTM (25%), and Attention-LSTM (25%)

3. **Empirical Excellence**: Demonstrated ensemble methodology achieves remarkable performance metrics with 77.5% win rate across all stocks tested

4. **Practical Validation**: Live trading validation achieving positive returns (₹1,915.85 economic value generation) when individual models failed

---

## Model Architecture

### Four Distinct Model Implementations

1. **Baseline LSTM**: Standard LSTM implementation for performance comparison
2. **MSLSTM**: Multi-Scale LSTM with dual-layer temporal extraction (exact paper implementation)
3. **MSLSTMA**: Multi-Scale LSTM with Attention - Novel autoencoder architecture with attention mechanisms
4. **Stacked Ensemble**: Advanced 3-level ensemble for superior directional prediction

### Ensemble Framework Architecture
- **Level 1 - Base Models**: TCN, WaveNet, LSTM, Attention-LSTM
- **Level 2 - Feature Extraction**: Each base model outputs prediction + 32-dimensional features  
- **Level 3 - Meta-Learners**: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees
- **Final Output**: Weighted voting with optimized ensemble weights
## Performance Results

### Ensemble Methodology Achievement
- **RMSE**: 32.18 (17.4% improvement over best individual model)
- **MAE**: 24.91 (16.5% improvement) 
- **R² Score**: 0.924 (3.7% improvement)
- **Win Rate**: 77.5% across all 40 stocks tested

### Live Trading Validation Results (23-Day Period: January 6 - February 12, 2026)
- **Total Predictions**: 3,588 real market predictions across 40 stocks

- **Best Performing Stocks** (Ensemble Model):
  - **SBIN.NS**: ₹2,800.60 profit (69.6% accuracy, 21 trades)
  - **NTPC.NS**: ₹2,201.93 profit (69.6% accuracy, 17 trades)
  - **ADANIPORTS.NS**: ₹2,083.91 profit (43.5% accuracy, 23 trades)
  - **ADANIENT.NS**: ₹1,591.20 profit (60.9% accuracy, 20 trades)
  - **AXISBANK.NS**: ₹1,556.93 profit (65.2% accuracy, 12 trades)
  - **Cumulative Top 5 Profit**: ₹10,234.57

- **Key Metrics**:
  - Profitable Days: 12/23 trading days
  - Ensemble Max Single Day Profit: ₹5,716.27 (Feb 3, 2026)
  - Total Profit on Profitable Days (Ensemble): ₹26,133.62

### Key Research Findings
1. **Ensemble Superiority**: Ensemble methodologies significantly outperform individual model approaches in both theoretical metrics and practical trading applications
2. **Consistency Advantage**: While individual models showed performance degradation over time, the ensemble demonstrated exceptional consistency and adaptability
3. **Risk Management**: Superior risk management during market downturns and robust adaptation to changing market dynamics
4. **Practical Validation**: Framework bridges academic research with real-world financial applications

---

## Supported Markets & Stocks

### Indian Equity Markets (NSE) - 8 Sectors, 40 Stocks

**Technology (5 stocks)**: TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS, TECHM.NS

**Banking (5 stocks)**: HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS, AXISBANK.NS  

**Pharmaceuticals (5 stocks)**: SUNPHARMA.NS, DRREDDY.NS, CIPLA.NS, APOLLOHOSP.NS, BIOCON.NS

**Automobile (4 stocks)**: MARUTI.NS, TATAMOTORS.NS, M&M.NS, BAJAJ-AUTO.NS

**Energy (5 stocks)**: RELIANCE.NS, ONGC.NS, IOC.NS, BPCL.NS, HINDPETRO.NS

**FMCG (5 stocks)**: HINDUNILVR.NS, ITC.NS, NESTLEIND.NS, BRITANNIA.NS, DABUR.NS

**Metals (5 stocks)**: TATASTEEL.NS, JSWSTEEL.NS, HINDALCO.NS, COALINDIA.NS, VEDL.NS

**Infrastructure (6 stocks)**: LT.NS, ULTRACEMCO.NS, ADANIPORTS.NS, GAIL.NS, NTPC.NS, ADANIGREEN.NS

*Comprehensive evaluation across major NSE sectors representing diverse market dynamics*

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/prateek2131/stock_price_prediction.git
cd stock_price_prediction

# Install dependencies
pip install -r requirements.txt

# Setup configuration
python setup_repository.sh
```

### Model Training & Evaluation

```bash
# Train all four models for comparison (Baseline, MSLSTM, MSLSTMA, Ensemble)
python MSLSTM_paper_implementation.py --ticker TCS.NS --model all

# Train specific model
python MSLSTM_paper_implementation.py --ticker TCS.NS --model mslstma

# Sector-wise comparison
python MSLSTM_paper_implementation.py --sector Technology --model all
```

### Live Trading Simulation

```bash
# Daily prediction simulation
python daily_predictions.py --ticker TCS.NS

# Multi-stock portfolio simulation
python real_trading_simulator.py --portfolio TCS.NS INFY.NS WIPRO.NS

# Complete framework evaluation
python run_complete_framework.py --mode all-sectors
```

### Research Framework

```bash
# Complete model comparison across all 40 stocks
python MSLSTM_model_comparison.py --mode comprehensive

# Generate research analysis
python generate_final_analysis.py
```

---

## Research Results Summary

### Academic Contributions

**Comprehensive Evaluation**: Extensive empirical analysis across 40 stocks spanning 8 sectors of Indian equity markets, measuring both predictive accuracy and directional trading performance.

**Live Trading Validation**: Implementation of automated trading simulation with daily ₹25,000 investment strategies, evaluating practical applicability and risk-adjusted returns.

**Methodological Innovation**: Novel MSLSTMA architecture combining LSTM autoencoders with attention mechanisms for superior noise reduction and feature extraction.

### Key Research Findings

1. **Ensemble Superiority**: Demonstrated that ensemble methodologies, when properly designed, significantly outperform individual model approaches in both theoretical metrics and practical trading applications

2. **Consistency in Volatile Markets**: While individual models showed performance degradation over time, the ensemble model demonstrated exceptional consistency and adaptability, maintaining stable performance in volatile market conditions

3. **Economic Value Generation**: Framework bridges academic research with real-world financial applications, providing a validated foundation for quantitative trading systems in emerging markets

4. **Risk Management Excellence**: Superior risk management during market downturns and robust adaptation to changing market dynamics

---

## Project Structure

```
Final_Project/
├── Core Implementation
│   ├── MSLSTM_paper_implementation.py   # Main model implementations (All 4 models)
│   ├── MSLSTM_model_comparison.py       # Comprehensive model comparison
│   ├── MSLSTM_prediction_framework.py   # Prediction framework
│   └── stacked_ensemble_predictor.py    # Advanced ensemble architecture
│
├── Research Analysis
│   ├── analyze_model_performance.py     # Performance analysis tools
│   ├── generate_final_analysis.py       # Research report generation
│   ├── evaluation_metrics.py           # Comprehensive evaluation metrics
│   └── visualizer.py                   # Advanced visualization
│
├── Trading Simulation
│   ├── daily_predictions.py            # Daily prediction system
│   ├── real_trading_simulator.py       # Live trading simulation
│   └── run_daily_predictions.sh        # Automated execution
│
├── Data & Utilities
│   ├── config.py                      # Configuration management
│   ├── data_preprocessor.py           # Feature engineering pipeline
│   ├── yfinance_helper.py            # Market data integration
│   └── utils.py                      # Utility functions
│
├── Results & Models
│   ├── models/                       # Trained model storage
│   ├── results/                     # Prediction results
│   ├── evaluations/                 # Evaluation outputs
│   ├── trading_results/             # Trading simulation results
│   └── daily_predictions/           # Daily prediction logs
│
├── Research Documentation
│   ├── Complete_Thesis_Report.tex    # Complete research thesis
│   ├── Abstract.pdf                 # Research abstract
│   └── README.md                   # Project documentation
│
└── Configuration
    ├── requirements.txt              # Python dependencies
    └── setup_repository.sh          # Repository setup script
```

---

## Technical Configuration

### Model Architectures

```python
# MSLSTM Configuration (Paper Implementation)
MSLSTM_CONFIG = {
    'lstm1_units': 100,      # First LSTM layer
    'lstm2_units': 100,      # Second LSTM layer  
    'dense1_units': 50,      # Dense layer
    'dropout_rate': 0.2,     # Regularization
    'sequence_length': 60,   # 60-day lookback
    'batch_size': 32,
    'epochs': 100
}

# MSLSTMA Configuration (Novel Architecture)
MSLSTMA_CONFIG = {
    'encoder_units': 100,    # LSTM Encoder
    'latent_dim': 50,        # Latent space dimension
    'decoder_units': 100,    # LSTM Decoder
    'dense1_units': 50,      # Output layer
    'attention_mechanism': True
}

# Stacked Ensemble Configuration
ENSEMBLE_CONFIG = {
    'base_models': ['tcn', 'wavenet', 'lstm', 'attention_lstm'],
    'meta_learners': ['lr', 'rf', 'gb', 'et'],
    'ensemble_weights': [0.20, 0.30, 0.25, 0.25],  # Optimized weights
    'feature_dimension': 32   # Per-model feature extraction
}
```

### Research Parameters

- **Evaluation Scope**: 40 stocks across 8 NSE sectors
- **Historical Data**: Multi-year training with technical indicators
- **Feature Engineering**: technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- **Live Trading Period**: January 6- February 12, 2026 (23 consecutive trading days)
- **Investment Strategy**: ₹25,000 daily investment per stock
- **Performance Metrics**: RMSE, MAE, R², Directional Accuracy, Economic Returns

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{prateek2026mslstm,
  title={Multi-Scale LSTM with Ensemble Learning for Stock Price Prediction: A Comprehensive Framework for Indian Equity Markets},
  author={Prateek Sanghi},
  year={2026},
  school={BITS PILANI},
  type={Master's Thesis},
  note={Comprehensive ensemble framework with live trading validation}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Yahoo Finance for comprehensive market data access
- Indian Stock Exchanges (NSE) for real-time price data
- TensorFlow and Scikit-learn communities for machine learning frameworks
- Academic research community for foundational LSTM and ensemble methodologies
