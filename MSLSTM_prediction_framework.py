"""
MSLSTM Model Prediction and Visualization Framework
==================================================

Advanced prediction and visualization system for all paper implementation models:
- Real-time prediction generation
- Interactive plots and charts
- Performance dashboards
- Future prediction capabilities
- Model confidence analysis

Features:
- Live predictions for next N days
- Model confidence intervals
- Ensemble prediction weights
- Technical analysis overlay
- Export predictions for trading

Author: Research Implementation
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Interactive plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MSLSTM_paper_implementation import (
    PaperModelTrainer, MSLSTMPaper, MSLSTMAPaper, 
    BaselineLSTM, StackedEnsemble, StockFeatureEngineer
)
from config import DATA_CONFIG, PATHS
import yfinance as yf

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pio.templates.default = "plotly_white"


class ModelPredictor:
    """
    Advanced prediction system for all MSLSTM paper models
    """
    
    def __init__(self, ticker='TCS.NS', prediction_days=1):
        self.ticker = ticker
        self.prediction_days = prediction_days
        
        # Trained models storage
        self.models = {}
        self.predictions = {}
        self.confidence_intervals = {}
        
        # Data
        self.data = None
        self.feature_engineer = StockFeatureEngineer()
        
        print(f"\n{'='*80}")
        print(f"MSLSTM NEXT-DAY PREDICTION FRAMEWORK")
        print(f"Ticker: {ticker} | Prediction Days: {prediction_days}")
        print('='*80)
    
    def load_trained_models(self, model_dir=None):
        """Load pre-trained models"""
        model_dir = model_dir or PATHS['models']
        
        print(f"\n📂 Loading trained models from {model_dir}...")
        
        model_types = ['baseline', 'mslstm', 'mslstma', 'ensemble']
        
        for model_type in model_types:
            try:
                print(f"Loading {model_type}...")
                
                # Create trainer and load model
                trainer = PaperModelTrainer(self.ticker, model_type)
                trainer.load_and_prepare_data()  # Need this for feature engineering
                
                # Try to load saved model
                if model_type in ['ensemble', 'stacked']:
                    model_path = os.path.join(model_dir, f"{self.ticker}_{model_type}_paper.pkl")
                else:
                    model_path = os.path.join(model_dir, f"{self.ticker}_{model_type}_paper.h5")
                
                if os.path.exists(model_path):
                    trainer.model.load(model_path)
                    self.models[model_type] = trainer
                    print(f"✅ {model_type} loaded")
                else:
                    print(f"⚠️  {model_type} model file not found, training new model...")
                    trainer.train(epochs=50)
                    trainer.save_model()
                    self.models[model_type] = trainer
                    print(f"✅ {model_type} trained and saved")
                    
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} models")
    
    def download_latest_data(self, days=365):
        """Download latest data for predictions"""
        print(f"\n📈 Downloading latest {days} days of data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        self.data = yf.download(
            self.ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        ).dropna()
        
        print(f"✅ Downloaded {len(self.data)} trading days")
        return self.data
    
    def generate_predictions(self, confidence_level=0.95):
        """Generate next-day predictions from all models"""
        print(f"\n🔮 Generating NEXT-DAY predictions...")
        
        if not self.models:
            print("❌ No models loaded. Please load models first.")
            return None
        
        # Get latest features
        latest_features = self.feature_engineer.calculate_features(self.data)
        
        # Generate predictions for each model
        for model_type, trainer in self.models.items():
            print(f"Predicting with {model_type}...")
            
            try:
                # Prepare input sequence
                X_input = self._prepare_prediction_input(latest_features, trainer)
                
                if model_type == 'ensemble':
                    # Ensemble returns probabilities, convert to price using paper approach
                    pred_probs = trainer.model.predict_proba(X_input)
                    next_day_price = self._convert_probs_to_next_day_price(pred_probs, latest_features)
                    confidence = self._calculate_ensemble_confidence(pred_probs)
                else:
                    # Regression models - direct next day prediction
                    predictions = trainer.model.predict(X_input)
                    next_day_price = trainer.feature_engineer.inverse_transform_predictions(
                        predictions, trainer.target_idx)[0]  # Get single prediction
                    
                    # Ensure it's a scalar
                    if hasattr(next_day_price, 'iloc'):
                        next_day_price = next_day_price.iloc[0] if len(next_day_price) > 0 else float(next_day_price)
                    elif hasattr(next_day_price, 'item'):
                        next_day_price = next_day_price.item()
                    else:
                        next_day_price = float(next_day_price)
                    
                    confidence = self._calculate_regression_confidence(predictions)
                    
                    # Ensure confidence is scalar
                    if hasattr(confidence, 'iloc'):
                        confidence = confidence.iloc[0] if len(confidence) > 0 else float(confidence)
                    elif hasattr(confidence, 'item'):
                        confidence = confidence.item()
                    else:
                        confidence = float(confidence)
                
                # Ensure scalar values
                current_price = self.data['Close'].iloc[-1]
                if hasattr(current_price, 'item'):
                    current_price = current_price.item()
                elif hasattr(current_price, 'iloc'):
                    current_price = current_price.iloc[0] if len(current_price) > 0 else float(current_price)
                
                price_change = next_day_price - current_price
                if hasattr(price_change, 'item'):
                    price_change = price_change.item()
                elif hasattr(price_change, 'iloc'):
                    price_change = price_change.iloc[0] if len(price_change) > 0 else float(price_change)
                
                price_change_pct = ((next_day_price / current_price) - 1) * 100
                if hasattr(price_change_pct, 'item'):
                    price_change_pct = price_change_pct.item()
                elif hasattr(price_change_pct, 'iloc'):
                    price_change_pct = price_change_pct.iloc[0] if len(price_change_pct) > 0 else float(price_change_pct)
                
                self.predictions[model_type] = {
                    'next_day_price': float(next_day_price),
                    'current_price': float(current_price),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'confidence': float(confidence),
                    'prediction_date': self._get_next_trading_day()
                }
                
                # Convert to scalar for printing
                price_change_pct = self.predictions[model_type]['price_change_pct']
                if hasattr(price_change_pct, 'iloc'):
                    price_change_pct = price_change_pct.iloc[0] if len(price_change_pct) > 0 else price_change_pct
                elif hasattr(price_change_pct, 'item'):
                    price_change_pct = price_change_pct.item()
                else:
                    price_change_pct = float(price_change_pct)
                
                next_day_price_scalar = next_day_price
                if hasattr(next_day_price, 'iloc'):
                    next_day_price_scalar = next_day_price.iloc[0] if len(next_day_price) > 0 else next_day_price
                elif hasattr(next_day_price, 'item'):
                    next_day_price_scalar = next_day_price.item()
                else:
                    next_day_price_scalar = float(next_day_price)
                
                print(f"✅ {model_type}: ₹{float(next_day_price_scalar):.2f} ({float(price_change_pct):+.2f}%)")
                
            except Exception as e:
                print(f"❌ Error generating {model_type} predictions: {e}")
        
        # Create ensemble of all predictions
        # self._create_next_day_meta_ensemble()  # Removed: Not needed
        
        print(f"\n✅ All next-day predictions generated!")
        
        # Print summary
        self._print_prediction_summary()
    
    def _prepare_prediction_input(self, features, trainer):
        """Prepare input sequence for prediction"""
        # Use the same sequence length as training
        seq_length = trainer.config['sequence_length']
        
        # Get recent data
        recent_data = features.iloc[-seq_length:].values
        
        # Apply same scaling as training
        if hasattr(trainer.feature_engineer, 'scaler'):
            recent_data = trainer.feature_engineer.scaler.transform(recent_data)
        
        return recent_data.reshape(1, seq_length, -1)
    
    def _convert_direction_to_next_day_price(self, direction_pred, features):
        """Convert ensemble direction prediction (0/1) to next-day price prediction"""
        current_price = self.data['Close'].iloc[-1]
        
        # Average daily change from recent history
        daily_changes = np.abs(np.diff(self.data['Close'].values))
        avg_daily_change = np.mean(daily_changes[-30:])  # Last 30 days
        
        # Convert direction to price change
        # direction_pred: 1 = UP, 0 = DOWN
        direction_value = direction_pred[0] if hasattr(direction_pred, '__getitem__') and len(direction_pred) > 0 else direction_pred
        if hasattr(direction_value, 'iloc'):
            direction_value = direction_value.iloc[0] if len(direction_value) > 0 else float(direction_value)
        elif hasattr(direction_value, 'item'):
            direction_value = direction_value.item()
        
        # Use recent volatility to estimate price change magnitude
        recent_returns = np.diff(self.data['Close'].values[-30:]) / self.data['Close'].values[-31:-1]
        avg_return_magnitude = np.mean(np.abs(recent_returns))
        
        # Apply direction to price change
        price_change_pct = avg_return_magnitude if float(direction_value) == 1 else -avg_return_magnitude
        next_day_price = current_price * (1 + price_change_pct)
        
        return float(next_day_price)
    
    def _convert_probs_to_next_day_price(self, probs, features):
        """Convert ensemble probabilities to next-day price prediction using paper approach"""
        current_price = self.data['Close'].iloc[-1]
        
        # Extract probability value
        prob_value = probs[0] if hasattr(probs, '__getitem__') and len(probs) > 0 else probs
        if hasattr(prob_value, 'iloc'):
            prob_value = prob_value.iloc[0] if len(prob_value) > 0 else float(prob_value)
        elif hasattr(prob_value, 'item'):
            prob_value = prob_value.item()
        
        # Calculate historical volatility (same as paper implementation)
        prices = self.data['Close'].values
        price_volatility = np.std(prices[-30:]) if len(prices) > 30 else np.std(prices)  # Last 30 days
        mean_price = np.mean(prices[-30:]) if len(prices) > 30 else np.mean(prices)
        max_change_pct = price_volatility / mean_price if mean_price > 0 else 0.02
        
        # Convert probability to percentage change (paper implementation approach)
        prob_to_change = (float(prob_value) - 0.5) * 2 * max_change_pct
        next_day_price = current_price * (1 + prob_to_change)
        
        return float(next_day_price)
    
    def _get_next_trading_day(self):
        """Get next trading day (skip weekends)"""
        last_date = self.data.index[-1]
        next_date = last_date + timedelta(days=1)
        
        # Skip weekends
        while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_date += timedelta(days=1)
            
        return next_date
    
    def _create_next_day_meta_ensemble(self):
        """Create ensemble of all next-day predictions"""
        if len(self.predictions) < 2:
            return
        
        print("🔮 Creating next-day meta-ensemble...")
        
        # Filter out predictions with NaN values
        valid_predictions = {}
        for model_type, pred in self.predictions.items():
            next_day_price = pred['next_day_price']
            if hasattr(next_day_price, 'item'):
                next_day_price = next_day_price.item()
            elif hasattr(next_day_price, 'iloc'):
                next_day_price = next_day_price.iloc[0] if len(next_day_price) > 0 else float(next_day_price)
            
            # Check if prediction is valid (not NaN or inf)
            if not (np.isnan(next_day_price) or np.isinf(next_day_price)):
                valid_predictions[model_type] = pred
                print(f"  ✓ {model_type}: ₹{next_day_price:.2f}")
            else:
                print(f"  ✗ {model_type}: Invalid prediction (NaN/Inf)")
        
        if len(valid_predictions) == 0:
            print("❌ No valid predictions available for ensemble")
            return
        
        # Weight models based on their type (ensemble gets higher weight)
        weights = {
            'ensemble': 0.4,
            'mslstma': 0.25,
            'mslstm': 0.20,
            'baseline': 0.15
        }
        
        # Normalize weights for available valid models
        available_models = list(valid_predictions.keys())
        total_weight = sum(weights.get(model, 0.1) for model in available_models)
        normalized_weights = {model: weights.get(model, 0.1) / total_weight 
                            for model in available_models}
        
        # Create weighted ensemble
        ensemble_price = 0
        ensemble_confidence = 0
        current_price = self.data['Close'].iloc[-1]
        
        # Ensure current_price is valid
        if hasattr(current_price, 'item'):
            current_price = current_price.item()
        elif hasattr(current_price, 'iloc'):
            current_price = current_price.iloc[0] if len(current_price) > 0 else float(current_price)
        
        if np.isnan(current_price) or np.isinf(current_price):
            print(f"❌ Invalid current price: {current_price}")
            return
        
        for model_type, weight in normalized_weights.items():
            if model_type in valid_predictions:
                pred = valid_predictions[model_type]
                
                # Ensure next_day_price is scalar
                next_day_price = pred['next_day_price']
                if hasattr(next_day_price, 'iloc'):
                    next_day_price = next_day_price.iloc[0] if len(next_day_price) > 0 else float(next_day_price)
                elif hasattr(next_day_price, 'item'):
                    next_day_price = next_day_price.item()
                
                # Ensure confidence is scalar
                confidence = pred['confidence']
                if hasattr(confidence, 'iloc'):
                    confidence = confidence.iloc[0] if len(confidence) > 0 else float(confidence)
                elif hasattr(confidence, 'item'):
                    confidence = confidence.item()
                
                # Ensure confidence is valid
                if np.isnan(confidence) or np.isinf(confidence):
                    confidence = 0.5  # Default confidence
                
                ensemble_price += weight * float(next_day_price)
                ensemble_confidence += weight * float(confidence)
        
        # Ensure ensemble values are valid
        if np.isnan(ensemble_price) or np.isinf(ensemble_price):
            print("❌ Ensemble price calculation resulted in NaN/Inf")
            return
        
        price_change = ensemble_price - current_price
        price_change_pct = ((ensemble_price / current_price) - 1) * 100
        
        self.predictions['meta_ensemble'] = {
            'next_day_price': float(ensemble_price),
            'current_price': float(current_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'confidence': float(ensemble_confidence),
            'prediction_date': self._get_next_trading_day(),
            'weights': normalized_weights
        }
        
        print(f"✅ Meta-ensemble: ₹{ensemble_price:.2f} ({price_change_pct:+.2f}%)")
    
    def _print_prediction_summary(self):
        """Print comprehensive prediction summary"""
        print(f"\n{'='*80}")
        print(f"NEXT-DAY PREDICTION SUMMARY - {self.ticker}")
        print('='*80)
        
        current_price = self.data['Close'].iloc[-1]
        # Ensure current_price is scalar
        if hasattr(current_price, 'item'):
            current_price = current_price.item()
        elif hasattr(current_price, 'iloc'):
            current_price = current_price.iloc[0] if len(current_price) > 0 else float(current_price)
        else:
            current_price = float(current_price)
            
        next_trading_day = self._get_next_trading_day()
        
        print(f"📅 Current Date: {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📅 Next Trading Day: {next_trading_day.strftime('%Y-%m-%d')}")
        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"\n{'─'*80}")
        
        # Sort predictions by confidence  
        sorted_predictions = sorted(
            [(model, pred) for model, pred in self.predictions.items()],  # Removed meta_ensemble exclusion
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )
        
        print(f"{'Model':<15} {'Price':<10} {'Change':<12} {'Change%':<10}")
        print('─'*65)
        
        for model_type, pred in sorted_predictions:
            # Ensure all values are scalars before formatting
            next_day_price = float(pred['next_day_price'])
            price_change = float(pred['price_change'])
            price_change_pct = float(pred['price_change_pct'])
            
            print(f"{model_type.title():<15} "
                  f"₹{next_day_price:<9.2f} "
                  f"₹{price_change:<11.2f} "
                  f"{price_change_pct:<9.2f}%")
        
        # Meta ensemble section removed - not needed
        
        print('='*65)
    
    def _calculate_ensemble_confidence(self, probs):
        """Calculate confidence for ensemble predictions"""
        # Confidence based on how close to 0.5 (uncertainty) the prediction is
        prob_value = probs[0] if hasattr(probs, '__getitem__') and len(probs) > 0 else probs
        if hasattr(prob_value, 'iloc'):
            prob_value = prob_value.iloc[0] if len(prob_value) > 0 else float(prob_value)
        elif hasattr(prob_value, 'item'):
            prob_value = prob_value.item()
        
        confidence = abs(float(prob_value) - 0.5) * 2  # Scale to 0-1
        return float(confidence)
    
    def _calculate_regression_confidence(self, predictions):
        """Calculate confidence for regression models"""
        # For regression models, use volatility-based confidence
        recent_volatility = np.std(np.diff(self.data['Close'].values[-30:]))
        confidence = 1.0 / (1.0 + recent_volatility)  # Higher volatility = lower confidence
        return float(confidence)
    
    def _generate_future_sequence(self, initial_pred, last_actual, days):
        """Generate sequence of future prices"""
        prices = [last_actual]
        
        # Simple trend continuation based on initial prediction
        daily_trend = (initial_pred[0] - last_actual) / days
        
        for day in range(1, days + 1):
            # Add some realistic volatility
            volatility = np.std(np.diff(self.data['Close'].values[-30:]))
            noise = np.random.normal(0, volatility * 0.5)
            
            next_price = prices[-1] + daily_trend + noise
            prices.append(max(next_price, prices[-1] * 0.8))  # Prevent unrealistic drops
        
        return np.array(prices[1:])  # Exclude initial actual price
    
    def _generate_prediction_dates(self):
        """Generate future trading dates"""
        last_date = self.data.index[-1]
        dates = []
        
        current_date = last_date
        days_added = 0
        
        while days_added < self.prediction_days:
            current_date += timedelta(days=1)
            
            # Skip weekends (basic calendar, doesn't account for holidays)
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current_date)
                days_added += 1
        
        return dates
    
    def _create_meta_ensemble(self):
        """Create ensemble of all model predictions"""
        if len(self.predictions) < 2:
            return
        
        print("🔮 Creating meta-ensemble...")
        
        # Weight models based on their type (ensemble gets higher weight)
        weights = {
            'ensemble': 0.4,
            'mslstma': 0.25,
            'mslstm': 0.20,
            'baseline': 0.15
        }
        
        # Normalize weights for available models
        available_models = list(self.predictions.keys())
        total_weight = sum(weights.get(model, 0.1) for model in available_models)
        normalized_weights = {model: weights.get(model, 0.1) / total_weight 
                            for model in available_models}
        
        # Create weighted ensemble
        ensemble_prices = np.zeros(self.prediction_days)
        ensemble_confidence = 0
        
        for model_type, weight in normalized_weights.items():
            if model_type in self.predictions:
                pred = self.predictions[model_type]
                ensemble_prices += weight * pred['prices']
                ensemble_confidence += weight * pred['confidence']
        
        self.predictions['meta_ensemble'] = {
            'prices': ensemble_prices,
            'confidence': ensemble_confidence,
            'last_actual': self.data['Close'].iloc[-1],
            'prediction_dates': self._generate_prediction_dates(),
            'weights': normalized_weights
        }
        
        print(f"✅ Meta-ensemble created with weights: {normalized_weights}")
    
    def create_prediction_dashboard(self, save_path=None):
        """Create comprehensive prediction dashboard"""
        if not self.predictions:
            print("❌ No predictions available. Please generate predictions first.")
            return
        
        print("📊 Creating prediction dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Price Predictions (All Models)',
                'Model Confidence Levels',
                'Historical vs Predicted',
                'Prediction Ranges',
                'Volume Analysis',
                'Model Weights (Meta Ensemble)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Color palette for models
        colors = {
            'baseline': '#1f77b4',
            'mslstm': '#ff7f0e',
            'mslstma': '#2ca02c',
            'ensemble': '#d62728',
            'meta_ensemble': '#9467bd'
        }
        
        # 1. Price Predictions
        historical_dates = self.data.index[-60:]  # Last 60 days
        historical_prices = self.data['Close'].iloc[-60:].values
        
        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='black', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Plot predictions
        for model_type, pred in self.predictions.items():
            if model_type != 'meta_ensemble':  # Handle separately
                fig.add_trace(
                    go.Scatter(
                        x=pred['prediction_dates'],
                        y=pred['prices'],
                        name=model_type.title(),
                        line=dict(color=colors.get(model_type, 'gray'), width=2),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Highlight meta ensemble
        if 'meta_ensemble' in self.predictions:
            meta_pred = self.predictions['meta_ensemble']
            fig.add_trace(
                go.Scatter(
                    x=meta_pred['prediction_dates'],
                    y=meta_pred['prices'],
                    name='Meta Ensemble',
                    line=dict(color=colors['meta_ensemble'], width=3, dash='dash'),
                    opacity=1.0
                ),
                row=1, col=1
            )
        
        # 2. Model Confidence Levels
        for model_type, pred in self.predictions.items():
            if model_type != 'meta_ensemble':
                confidence_level = pred['confidence'] if np.isscalar(pred['confidence']) else np.mean(pred['confidence'])
                fig.add_trace(
                    go.Bar(
                        x=[model_type.title()],
                        y=[confidence_level * 100],
                        name=f'{model_type} Confidence',
                        marker_color=colors.get(model_type, 'gray'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Historical vs Predicted (Recent performance)
        last_30_days = self.data['Close'].iloc[-30:].values
        last_30_dates = self.data.index[-30:]
        
        fig.add_trace(
            go.Scatter(
                x=last_30_dates,
                y=last_30_days,
                name='Recent Actual',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        # Add prediction starting point
        fig.add_trace(
            go.Scatter(
                x=[self.data.index[-1]] + self.predictions['meta_ensemble']['prediction_dates'][:5],
                y=[self.data['Close'].iloc[-1]] + list(self.predictions['meta_ensemble']['prices'][:5]),
                name='Meta Prediction (5d)',
                line=dict(color=colors['meta_ensemble'], width=3, dash='dot'),
                mode='lines+markers'
            ),
            row=2, col=1
        )
        
        # Volume analysis
        volume_dates = self.data.index[-30:]
        volumes = self.data['Volume'].iloc[-30:].values
        
        fig.add_trace(
            go.Bar(
                x=volume_dates,
                y=volumes,
                name='Volume',
                marker_color='lightblue',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # 4. Prediction Ranges (confidence intervals)
        if 'meta_ensemble' in self.predictions:
            meta_pred = self.predictions['meta_ensemble']
            
            # Create confidence bands (±1 standard deviation)
            volatility = np.std(np.diff(self.data['Close'].values[-30:]))
            upper_bound = meta_pred['prices'] + volatility
            lower_bound = meta_pred['prices'] - volatility
            
            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=meta_pred['prediction_dates'],
                    y=upper_bound,
                    line=dict(color='rgba(148, 103, 189, 0)'),
                    name='Upper Bound',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=meta_pred['prediction_dates'],
                    y=lower_bound,
                    fill='tonexty',
                    fillcolor='rgba(148, 103, 189, 0.3)',
                    line=dict(color='rgba(148, 103, 189, 0)'),
                    name='Confidence Band',
                    showlegend=True
                ),
                row=2, col=2
            )
            
            # Center line
            fig.add_trace(
                go.Scatter(
                    x=meta_pred['prediction_dates'],
                    y=meta_pred['prices'],
                    line=dict(color=colors['meta_ensemble'], width=3),
                    name='Meta Ensemble',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 5. Volume Analysis (recent)
        recent_volume = self.data['Volume'].iloc[-20:].values
        recent_dates = self.data.index[-20:]
        
        fig.add_trace(
            go.Scatter(
                x=recent_dates,
                y=recent_volume,
                mode='lines+markers',
                name='Recent Volume',
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Model Weights (Pie chart)
        if 'meta_ensemble' in self.predictions:
            weights = self.predictions['meta_ensemble']['weights']
            fig.add_trace(
                go.Pie(
                    labels=[name.title() for name in weights.keys()],
                    values=list(weights.values()),
                    name="Model Weights",
                    marker_colors=[colors.get(model, 'gray') for model in weights.keys()],
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'MSLSTM Prediction Dashboard - {self.ticker}',
            height=1200,
            showlegend=True,
            font=dict(size=10),
            template='plotly_white'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Confidence (%)", row=1, col=2)
        fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Price (₹)", row=2, col=2)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        # Update x-axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 Dashboard saved to: {save_path}")
        
        fig.show()
        
        return fig
    
    def export_predictions(self, filename=None):
        """Export next-day predictions to CSV"""
        if not self.predictions:
            print("❌ No predictions available.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_next_day_predictions_{timestamp}.csv"
        
        print(f"💾 Exporting next-day predictions to {filename}...")
        
        # Create export data
        export_data = []
        current_price = self.data['Close'].iloc[-1]
        next_trading_day = self._get_next_trading_day()
        
        for model_type, pred in self.predictions.items():
            confidence_pct = pred['confidence'] * 100 if pred['confidence'] <= 1 else pred['confidence']
            
            export_data.append({
                'Ticker': self.ticker,
                'Current_Date': self.data.index[-1].strftime('%Y-%m-%d'),
                'Prediction_Date': next_trading_day.strftime('%Y-%m-%d'),
                'Model': model_type.title(),
                'Current_Price': current_price,
                'Predicted_Price': pred['next_day_price'],
                'Price_Change': pred['price_change'],
                'Price_Change_Pct': pred['price_change_pct'],
                'Confidence_Pct': confidence_pct,
                'Direction': 'UP' if pred['price_change'] > 0 else 'DOWN'
            })
        
        # Save to CSV
        export_df = pd.DataFrame(export_data)
        export_path = os.path.join(PATHS['results'], filename)
        export_df.to_csv(export_path, index=False)
        
        print(f"✅ Next-day predictions exported to: {export_path}")
        
        return export_df
    
    def run_prediction_pipeline(self, retrain=False):
        """Run complete next-day prediction pipeline"""
        print(f"\n🚀 Starting NEXT-DAY prediction pipeline for {self.ticker}...")
        
        # Step 1: Download latest data
        self.download_latest_data()
        
        # Step 2: Load or train models
        if retrain:
            print("🔄 Retraining all models...")
            for model_type in ['baseline', 'mslstm', 'mslstma', 'ensemble']:
                try:
                    trainer = PaperModelTrainer(self.ticker, model_type)
                    trainer.load_and_prepare_data()
                    trainer.train(epochs=50)
                    trainer.save_model()
                    self.models[model_type] = trainer
                    print(f"✅ {model_type} retrained")
                except Exception as e:
                    print(f"❌ Failed to retrain {model_type}: {e}")
        else:
            self.load_trained_models()
        
        # Step 3: Generate next-day predictions
        self.generate_predictions()
        
        # Step 4: Export predictions
        self.export_predictions()
        
        print(f"\n🎉 Next-day prediction pipeline complete for {self.ticker}!")


def run_single_prediction(ticker='TCS.NS', prediction_days=1, retrain=False):
    """Run next-day prediction for a single ticker"""
    predictor = ModelPredictor(ticker=ticker, prediction_days=prediction_days)
    predictor.run_prediction_pipeline(retrain=retrain)
    return predictor


def run_portfolio_predictions(tickers=None, prediction_days=1):
    """Run next-day predictions for multiple tickers"""
    if tickers is None:
        tickers = ['TCS.NS', 'INFY.NS', 'WIPRO.NS']  # Default tech stocks
    
    print(f"\n{'='*80}")
    print(f"PORTFOLIO NEXT-DAY PREDICTIONS")
    print(f"Tickers: {', '.join(tickers)}")
    print('='*80)
    
    predictors = {}
    
    for ticker in tickers:
        try:
            print(f"\n{'─'*60}")
            print(f"Processing {ticker}...")
            print('─'*60)
            
            predictor = ModelPredictor(ticker=ticker, prediction_days=prediction_days)
            predictor.run_prediction_pipeline(retrain=False)
            predictors[ticker] = predictor
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    # Create portfolio summary
    create_portfolio_summary(predictors)
    
    return predictors


def create_portfolio_summary(predictors):
    """Create consolidated portfolio next-day summary"""
    print(f"\n{'='*100}")
    print("PORTFOLIO NEXT-DAY PREDICTION SUMMARY")
    print('='*100)
    
    summary_data = []
    
    for ticker, predictor in predictors.items():
        if predictor and predictor.predictions:
            # Get the model with highest confidence instead of meta_ensemble
            best_model = max(predictor.predictions.items(), key=lambda x: x[1]['confidence'])
            model_name, best_pred = best_model
            current_price = predictor.data['Close'].iloc[-1]
            
            summary_data.append({
                'Ticker': ticker,
                'Model': model_name.upper(),
                'Current_Price': current_price,
                'Next_Day_Prediction': best_pred['next_day_price'],
                'Price_Change': best_pred['price_change'],
                'Change_Pct': best_pred['price_change_pct'],
                'Direction': 'UP' if best_pred['price_change'] > 0 else 'DOWN',
                'Confidence': best_pred['confidence'] * 100 if best_pred['confidence'] <= 1 else best_pred['confidence']
            })
    
    if summary_data:
        portfolio_df = pd.DataFrame(summary_data)
        
        # Sort by expected return (absolute change %)
        portfolio_df['Abs_Change_Pct'] = portfolio_df['Change_Pct'].abs()
        portfolio_df = portfolio_df.sort_values('Abs_Change_Pct', ascending=False)
        
        # Display formatted table
        print(f"{'Ticker':<10} {'Model':<10} {'Current':<10} {'Predicted':<10} {'Change':<10} {'Change%':<8} {'Direction':<9} {'Confidence':<10}")
        print('─' * 95)
        
        for _, row in portfolio_df.iterrows():
            print(f"{row['Ticker']:<10} "
                  f"{row['Model']:<10} "
                  f"₹{row['Current_Price']:<9.2f} "
                  f"₹{row['Next_Day_Prediction']:<9.2f} "
                  f"₹{row['Price_Change']:<9.2f} "
                  f"{row['Change_Pct']:<7.2f}% "
                  f"{row['Direction']:<9} "
                  f"{row['Confidence']:<9.1f}%")
        
        # Save portfolio summary
        portfolio_path = os.path.join(PATHS['results'], 'portfolio_next_day_predictions.csv')
        portfolio_df.drop('Abs_Change_Pct', axis=1).to_csv(portfolio_path, index=False)
        print(f"\n💾 Portfolio summary saved: {portfolio_path}")
        
        # Trading recommendations
        print(f"\n💡 TRADING RECOMMENDATIONS:")
        print('─' * 50)
        
        bullish_stocks = portfolio_df[portfolio_df['Direction'] == 'UP']
        bearish_stocks = portfolio_df[portfolio_df['Direction'] == 'DOWN']
        
        if len(bullish_stocks) > 0:
            top_bullish = bullish_stocks.iloc[0]
            print(f"📈 MOST BULLISH: {top_bullish['Ticker']} ({top_bullish['Change_Pct']:+.2f}%, {top_bullish['Confidence']:.1f}% confidence)")
        
        if len(bearish_stocks) > 0:
            top_bearish = bearish_stocks.iloc[0]
            print(f"📉 MOST BEARISH: {top_bearish['Ticker']} ({top_bearish['Change_Pct']:+.2f}%, {top_bearish['Confidence']:.1f}% confidence)")
        
        high_confidence = portfolio_df[portfolio_df['Confidence'] > 70]
        if len(high_confidence) > 0:
            print(f"\n🎯 HIGH CONFIDENCE PICKS:")
            for _, stock in high_confidence.iterrows():
                print(f"   • {stock['Ticker']}: {stock['Direction']} {stock['Change_Pct']:+.2f}% ({stock['Confidence']:.1f}% confidence)")
    
    print(f"\n✅ Portfolio analysis complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MSLSTM Prediction Framework')
    parser.add_argument('--ticker', type=str, default='TCS.NS', 
                       help='Stock ticker to predict')
    parser.add_argument('--days', type=int, default=1, 
                       help='Number of days to predict (default: 1 for next-day)')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain all models before prediction')
    parser.add_argument('--portfolio', nargs='+', default=None,
                       help='Run predictions for multiple tickers')
    
    args = parser.parse_args()
    
    if args.portfolio:
        run_portfolio_predictions(args.portfolio, args.days)
    else:
        run_single_prediction(args.ticker, args.days, args.retrain)
