#!/usr/bin/env python3
"""
Real Trading Simulation with Actual Model Integration
Daily investment of ₹25,000 per stock based on model predictions
Tracks actual profit/loss with real price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import INDIAN_STOCKS_BY_SECTOR

class RealTradingSimulator:
    def __init__(self, investment_per_stock: float = 25000, stocks: List[str] = None):
        """
        Initialize real trading simulator with actual model integration
        
        Args:
            investment_per_stock: Amount to invest per stock per day (₹25,000)
            stocks: Specific list of stocks to trade, if None uses all stocks
        """
        self.investment_per_stock = investment_per_stock
        self.target_stocks = stocks  # Store the target stocks list
        self.results_dir = Path("trading_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Portfolio tracking
        self.portfolio = {
            'cash': 0,
            'holdings': {},  # {ticker: {'shares': float, 'avg_price': float}}
            'total_invested': 0,
            'total_profit_loss': 0,
            'trade_history': []
        }
        
        # Results file
        self.excel_file = self.results_dir / f"real_trading_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        print(f"🏦 Real Trading Simulator Initialized")
        print(f"💰 Daily Investment per Stock: ₹{self.investment_per_stock:,.0f}")
        print(f"📊 Results file: {self.excel_file}")
    
    def get_stock_list(self) -> List[str]:
        """Get list of stocks to trade"""
        if self.target_stocks is not None:
            return self.target_stocks  # Use specific stocks if provided
        
        # Otherwise, use all stocks from sectors
        stocks = []
        for sector_stocks in INDIAN_STOCKS_BY_SECTOR.values():
            stocks.extend(sector_stocks)
        return stocks
    
    def get_historical_data(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """Get historical data for a stock"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=f"{days}d")
            return data
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return pd.DataFrame()
    
    def make_prediction_with_real_model(self, ticker: str, model_type: str = 'ensemble') -> dict:
        """
        Make actual prediction using your trained models
        """
        try:
            # Import prediction framework
            from MSLSTM_prediction_framework import ModelPredictor
            
            # Initialize predictor
            predictor = ModelPredictor(ticker=ticker, prediction_days=1)
            
            # Run prediction pipeline
            predictor.run_prediction_pipeline(retrain=False)
            
            # Get ensemble prediction (or best available model)
            if predictor.predictions:
                # Try ensemble first, then fall back to best model
                if 'ensemble' in predictor.predictions:
                    prediction_data = predictor.predictions['ensemble']
                else:
                    # Get the model with highest confidence
                    best_model = max(predictor.predictions.items(), key=lambda x: x[1]['confidence'])
                    model_name, prediction_data = best_model
                
                current_price = prediction_data['current_price']
                predicted_price = prediction_data['next_day_price']
                predicted_change = prediction_data['price_change']
                predicted_change_pct = prediction_data['price_change_pct']
                
                # Determine direction
                predicted_direction = 1 if predicted_price > current_price else 0
                confidence = prediction_data.get('confidence', 0.5)
                
                return {
                    'success': True,
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predicted_change': predicted_change,
                    'predicted_change_pct': predicted_change_pct,
                    'predicted_direction': predicted_direction,
                    'confidence': confidence,
                    'model_type': model_type,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                print(f"❌ Prediction failed for {ticker}: No predictions available")
                return {'success': False, 'ticker': ticker, 'error': 'No predictions available'}
                
        except Exception as e:
            print(f"❌ Error making prediction for {ticker}: {e}")
            return {'success': False, 'ticker': ticker, 'error': str(e)}
    
    def should_invest(self, prediction: dict) -> bool:
        """
        Decide whether to invest based on prediction
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            True if should invest, False otherwise
        """
        if not prediction.get('success', False):
            return False
        
        # Investment strategy: Only invest if prediction is UP
        if prediction['predicted_direction'] == 1:  # UP
            return True
        
        return False
    
    def execute_daily_trading(self, trading_date: str = None) -> dict:
        """
        Execute daily trading for all stocks
        
        Args:
            trading_date: Date to simulate trading for (YYYY-MM-DD)
            
        Returns:
            Dictionary with daily trading results
        """
        if trading_date is None:
            trading_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n🎯 Executing Daily Trading - {trading_date}")
        print("=" * 60)
        
        stocks = self.get_stock_list()
        daily_results = {
            'date': trading_date,
            'trades': [],
            'total_invested': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'buy_trades': 0,
            'hold_decisions': 0
        }
        
        for i, ticker in enumerate(stocks, 1):
            print(f"\n[{i}/{len(stocks)}] Analyzing {ticker}...")
            
            try:
                # Get prediction
                prediction = self.make_prediction_with_real_model(ticker, 'ensemble')
                
                if not prediction['success']:
                    daily_results['failed_predictions'] += 1
                    print(f"  ❌ Prediction failed: {prediction.get('error', 'Unknown')}")
                    continue
                
                daily_results['successful_predictions'] += 1
                current_price = prediction['current_price']
                predicted_price = prediction['predicted_price']
                predicted_direction = prediction['predicted_direction']
                
                # Decide whether to invest
                should_buy = self.should_invest(prediction)
                
                if should_buy:
                    # Calculate shares to buy
                    shares = self.investment_per_stock / current_price
                    
                    trade = {
                        'date': trading_date,
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'investment': self.investment_per_stock,
                        'predicted_price': predicted_price,
                        'predicted_direction': 'UP',
                        'confidence': prediction['confidence'],
                        'model_type': prediction['model_type']
                    }
                    
                    daily_results['trades'].append(trade)
                    daily_results['total_invested'] += self.investment_per_stock
                    daily_results['buy_trades'] += 1
                    
                    print(f"  ✅ BUY: {shares:.2f} shares at ₹{current_price:.2f}")
                    print(f"  💰 Investment: ₹{self.investment_per_stock:,.0f}")
                    print(f"  📈 Predicted: ₹{predicted_price:.2f} (UP)")
                    
                else:
                    # Record hold decision
                    trade = {
                        'date': trading_date,
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'ticker': ticker,
                        'action': 'HOLD',
                        'shares': 0,
                        'price': current_price,
                        'investment': 0,
                        'predicted_price': predicted_price,
                        'predicted_direction': 'DOWN',
                        'confidence': prediction['confidence'],
                        'model_type': prediction['model_type']
                    }
                    
                    daily_results['trades'].append(trade)
                    daily_results['hold_decisions'] += 1
                    
                    print(f"  ⏸️  HOLD: Predicted DOWN (₹{predicted_price:.2f})")
                    print(f"  💸 No investment")
                
            except Exception as e:
                print(f"  ❌ Error processing {ticker}: {e}")
                daily_results['failed_predictions'] += 1
        
        # Print daily summary
        print(f"\n📊 Daily Trading Summary - {trading_date}")
        print(f"  Stocks Analyzed: {len(stocks)}")
        print(f"  Successful Predictions: {daily_results['successful_predictions']}")
        print(f"  Failed Predictions: {daily_results['failed_predictions']}")
        print(f"  Buy Trades: {daily_results['buy_trades']}")
        print(f"  Hold Decisions: {daily_results['hold_decisions']}")
        print(f"  Total Invested: ₹{daily_results['total_invested']:,.0f}")
        
        return daily_results
    
    def evaluate_previous_day_trades(self, previous_trades: list, current_date: str) -> list:
        """
        Evaluate trades from previous day with current prices
        
        Args:
            previous_trades: List of trades from previous day
            current_date: Current date (YYYY-MM-DD)
            
        Returns:
            List of evaluated trades with profit/loss
        """
        print(f"\n💹 Evaluating Previous Day Trades - {current_date}")
        print("=" * 50)
        
        evaluated_trades = []
        total_profit_loss = 0
        
        for trade in previous_trades:
            if trade['action'] != 'BUY':
                continue  # Skip HOLD trades
            
            ticker = trade['ticker']
            
            try:
                # Get current price
                stock_data = self.get_historical_data(ticker, days=5)
                if stock_data.empty:
                    print(f"❌ No current price data for {ticker}")
                    continue
                
                current_price = float(stock_data['Close'].iloc[-1])
                buy_price = trade['price']
                shares = trade['shares']
                
                # Calculate profit/loss
                profit_loss = (current_price - buy_price) * shares
                return_pct = ((current_price - buy_price) / buy_price) * 100
                
                # Check prediction accuracy
                actual_direction = 'UP' if current_price > buy_price else 'DOWN'
                prediction_correct = actual_direction == trade['predicted_direction']
                
                evaluated_trade = {
                    **trade,
                    'evaluation_date': current_date,
                    'sell_price': current_price,
                    'profit_loss': profit_loss,
                    'return_percentage': return_pct,
                    'actual_direction': actual_direction,
                    'prediction_correct': prediction_correct,
                    'accuracy': 'CORRECT' if prediction_correct else 'WRONG'
                }
                
                evaluated_trades.append(evaluated_trade)
                total_profit_loss += profit_loss
                
                print(f"  {ticker}: ₹{profit_loss:+,.2f} ({return_pct:+.2f}%) - {evaluated_trade['accuracy']}")
                
            except Exception as e:
                print(f"❌ Error evaluating {ticker}: {e}")
        
        print(f"\n💰 Total Profit/Loss: ₹{total_profit_loss:+,.2f}")
        return evaluated_trades
    
    def save_results_to_excel(self, daily_trades: list, evaluated_trades: list):
        """
        Save all results to Excel file
        
        Args:
            daily_trades: List of daily trading results
            evaluated_trades: List of evaluated trades with P&L
        """
        print(f"\n💾 Saving results to Excel...")
        
        with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
            
            # Sheet 1: Daily Trading Summary
            daily_summary = []
            for day_data in daily_trades:
                daily_summary.append({
                    'Date': day_data['date'],
                    'Stocks_Analyzed': len(day_data['trades']),
                    'Successful_Predictions': day_data['successful_predictions'],
                    'Failed_Predictions': day_data['failed_predictions'],
                    'Buy_Trades': day_data['buy_trades'],
                    'Hold_Decisions': day_data['hold_decisions'],
                    'Total_Invested': day_data['total_invested']
                })
            
            if daily_summary:
                df_daily = pd.DataFrame(daily_summary)
                df_daily.to_excel(writer, sheet_name='Daily_Summary', index=False)
            
            # Sheet 2: All Trades
            all_trades = []
            for day_data in daily_trades:
                all_trades.extend(day_data['trades'])
            
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                df_trades.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # Sheet 3: Evaluated Trades (with P&L)
            if evaluated_trades:
                df_evaluated = pd.DataFrame(evaluated_trades)
                df_evaluated.to_excel(writer, sheet_name='Profit_Loss_Analysis', index=False)
                
                # Sheet 4: Performance Summary
                total_investment = df_evaluated['investment'].sum()
                total_profit_loss = df_evaluated['profit_loss'].sum()
                total_return_pct = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
                
                correct_predictions = len(df_evaluated[df_evaluated['prediction_correct'] == True])
                total_predictions = len(df_evaluated)
                accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                
                performance_summary = pd.DataFrame([{
                    'Total_Investment': total_investment,
                    'Total_Profit_Loss': total_profit_loss,
                    'Total_Return_Percentage': total_return_pct,
                    'Prediction_Accuracy': accuracy,
                    'Correct_Predictions': correct_predictions,
                    'Total_Predictions': total_predictions,
                    'Profitable_Trades': len(df_evaluated[df_evaluated['profit_loss'] > 0]),
                    'Loss_Trades': len(df_evaluated[df_evaluated['profit_loss'] < 0])
                }])
                
                performance_summary.to_excel(writer, sheet_name='Performance_Summary', index=False)
        
        print(f"✅ Results saved to: {self.excel_file}")
    
    def run_simulation(self, days: int = 5):
        """
        Run complete trading simulation
        
        Args:
            days: Number of days to simulate
        """
        print(f"\n🚀 Starting {days}-Day Real Trading Simulation")
        print(f"💰 Investment per stock: ₹{self.investment_per_stock:,.0f}")
        print("=" * 60)
        
        all_daily_trades = []
        all_evaluated_trades = []
        
        # Start simulation using historical dates (so we can validate outcomes)
        # Use dates from 2-7 days ago to ensure we have actual next-day data
        start_days_ago = days + 5  # Start simulation from 6+ days ago
        for day in range(days):
            current_date = (datetime.now() - timedelta(days=start_days_ago-day)).strftime('%Y-%m-%d')
            print(f"\n{'🗓️ ' * 3} DAY {day+1}/{days} - {current_date} {'🗓️ ' * 3}")
            
            # Execute daily trading
            daily_result = self.execute_daily_trading(current_date)
            all_daily_trades.append(daily_result)
            
            # Evaluate previous day's trades (if any)
            if day > 0:
                previous_trades = all_daily_trades[day-1]['trades']
                evaluated = self.evaluate_previous_day_trades(previous_trades, current_date)
                all_evaluated_trades.extend(evaluated)
            
            print(f"\n⏳ Waiting before next day...")
            import time
            time.sleep(2)  # Small delay between days
        
        # Final evaluation of last day's trades using next trading day
        if all_daily_trades:
            # Use the day after the last simulation day for final evaluation
            final_date = (datetime.now() - timedelta(days=start_days_ago-days)).strftime('%Y-%m-%d')
            last_day_trades = all_daily_trades[-1]['trades']
            final_evaluated = self.evaluate_previous_day_trades(last_day_trades, final_date)
            all_evaluated_trades.extend(final_evaluated)
        
        # Save all results
        self.save_results_to_excel(all_daily_trades, all_evaluated_trades)
        
        # Print final summary
        if all_evaluated_trades:
            total_invested = sum(trade['investment'] for trade in all_evaluated_trades)
            total_profit_loss = sum(trade['profit_loss'] for trade in all_evaluated_trades)
            total_return = (total_profit_loss / total_invested) * 100 if total_invested > 0 else 0
            
            correct = len([t for t in all_evaluated_trades if t['prediction_correct']])
            accuracy = (correct / len(all_evaluated_trades)) * 100
            
            print(f"\n🎯 FINAL SIMULATION RESULTS")
            print("=" * 50)
            print(f"📅 Simulation Period: {days} days")
            print(f"💰 Total Invested: ₹{total_invested:,.0f}")
            print(f"💹 Total Profit/Loss: ₹{total_profit_loss:+,.0f}")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"🎯 Prediction Accuracy: {accuracy:.1f}%")
            print(f"📊 Results saved to: {self.excel_file}")
    
    def save_daily_predictions(self, predictions: List[dict], date: str):
        """Save daily predictions to JSON file for later evaluation"""
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True)
        
        filename = f"predictions_{date}.json"
        filepath = predictions_dir / filename
        
        import json
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f"💾 Predictions saved: {filepath}")
    
    def load_previous_predictions(self, date: str) -> List[dict]:
        """Load predictions from a specific date"""
        predictions_dir = Path("predictions")
        filename = f"predictions_{date}.json"
        filepath = predictions_dir / filename
        
        if not filepath.exists():
            print(f"❌ No predictions found for {date}")
            return []
        
        import json
        with open(filepath, 'r') as f:
            predictions = json.load(f)
        
        print(f"📂 Loaded {len(predictions)} predictions from {date}")
        return predictions
    
    def make_predictions_only(self) -> dict:
        """Make predictions for tomorrow without validation"""
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n🔮 Making Predictions for Tomorrow ({tomorrow})")
        print("=" * 60)
        
        stocks = self.get_stock_list()
        predictions = []
        
        for i, ticker in enumerate(stocks, 1):
            print(f"\n[{i}/{len(stocks)}] Predicting {ticker}...")
            
            try:
                # Get prediction for tomorrow
                prediction = self.make_prediction_with_real_model(ticker, 'ensemble')
                
                if not prediction['success']:
                    print(f"  ❌ Prediction failed: {prediction.get('error', 'Unknown')}")
                    continue
                
                # Store prediction with metadata
                pred_data = {
                    'ticker': ticker,
                    'prediction_date': today,
                    'target_date': tomorrow,
                    'current_price': prediction['current_price'],
                    'predicted_price': prediction['predicted_price'],
                    'predicted_change': prediction['predicted_change'],
                    'predicted_change_pct': prediction['predicted_change_pct'],
                    'confidence': prediction['confidence'],
                    'model_used': 'ensemble'
                }
                
                predictions.append(pred_data)
                
                print(f"  ✅ Current: ₹{prediction['current_price']:.2f} → "
                      f"Predicted: ₹{prediction['predicted_price']:.2f} "
                      f"({prediction['predicted_change_pct']:+.2f}%)")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Save predictions
        if predictions:
            self.save_daily_predictions(predictions, today)
            
            print(f"\n✅ Made {len(predictions)} predictions for {tomorrow}")
            print(f"💡 Run with --evaluate-only tomorrow to validate these predictions!")
        
        return {'predictions': predictions, 'date': today}
    
    def evaluate_previous_predictions(self) -> dict:
        """Evaluate yesterday's predictions against today's actual prices"""
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n📊 Evaluating Predictions from {yesterday}")
        print("=" * 60)
        
        # Load yesterday's predictions
        predictions = self.load_previous_predictions(yesterday)
        if not predictions:
            return {'evaluated': [], 'summary': {}}
        
        evaluated = []
        
        for pred in predictions:
            ticker = pred['ticker']
            print(f"\n📈 Evaluating {ticker}...")
            
            try:
                # Get today's actual price
                stock_data = self.get_historical_data(ticker, days=5)
                if stock_data.empty:
                    print(f"  ❌ No data available")
                    continue
                
                # Get today's closing price
                actual_price = stock_data['Close'].iloc[-1]
                predicted_price = pred['predicted_price']
                current_price = pred['current_price']
                
                # Calculate actual change
                actual_change = actual_price - current_price
                actual_change_pct = (actual_change / current_price) * 100
                
                # Calculate prediction accuracy
                price_error = abs(actual_price - predicted_price)
                price_error_pct = (price_error / actual_price) * 100
                
                # Direction accuracy
                predicted_direction = "UP" if pred['predicted_change'] > 0 else "DOWN"
                actual_direction = "UP" if actual_change > 0 else "DOWN"
                direction_correct = predicted_direction == actual_direction
                
                eval_data = {
                    **pred,
                    'actual_price': float(actual_price),
                    'actual_change': float(actual_change),
                    'actual_change_pct': float(actual_change_pct),
                    'price_error': float(price_error),
                    'price_error_pct': float(price_error_pct),
                    'direction_correct': direction_correct,
                    'predicted_direction': predicted_direction,
                    'actual_direction': actual_direction
                }
                
                evaluated.append(eval_data)
                
                direction_emoji = "✅" if direction_correct else "❌"
                print(f"  {direction_emoji} Predicted: ₹{predicted_price:.2f} | "
                      f"Actual: ₹{actual_price:.2f} | "
                      f"Error: {price_error_pct:.2f}% | "
                      f"Direction: {direction_correct}")
                
            except Exception as e:
                print(f"  ❌ Error evaluating {ticker}: {e}")
        
        # Calculate summary statistics
        if evaluated:
            avg_error = np.mean([e['price_error_pct'] for e in evaluated])
            direction_accuracy = np.mean([e['direction_correct'] for e in evaluated]) * 100
            
            summary = {
                'total_predictions': len(evaluated),
                'avg_price_error_pct': float(avg_error),
                'direction_accuracy': float(direction_accuracy),
                'date': yesterday
            }
            
            print(f"\n📊 EVALUATION SUMMARY")
            print("=" * 40)
            print(f"🎯 Predictions Evaluated: {summary['total_predictions']}")
            print(f"📈 Average Price Error: {avg_error:.2f}%")
            print(f"🎯 Direction Accuracy: {direction_accuracy:.1f}%")
            
            # Save evaluation results
            eval_dir = Path("evaluations")
            eval_dir.mkdir(exist_ok=True)
            
            import json
            eval_file = eval_dir / f"evaluation_{yesterday}.json"
            with open(eval_file, 'w') as f:
                json.dump({'evaluated': evaluated, 'summary': summary}, f, indent=2, default=str)
            
            print(f"💾 Evaluation saved: {eval_file}")
        
        return {'evaluated': evaluated, 'summary': summary if evaluated else {}}

    # ...existing code...
def main():
    """Run the real trading simulation with command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real Trading Simulator - Automated Stock Trading with MSLSTM Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --ticker TCS.NS --days 5                    # Single stock, 5 days
  %(prog)s --portfolio --days 3                        # All stocks, 3 days  
  %(prog)s --tickers TCS.NS INFY.NS WIPRO.NS --days 7  # Specific stocks, 7 days
  %(prog)s --investment 50000 --portfolio --days 1     # Custom investment amount
        ''')
    
    # Main arguments
    parser.add_argument('--ticker', type=str, 
                       help='Single stock ticker symbol (e.g., TCS.NS)')
    parser.add_argument('--tickers', nargs='+', 
                       help='Multiple stock tickers (e.g., TCS.NS INFY.NS WIPRO.NS)')
    parser.add_argument('--portfolio', action='store_true',
                       help='Run simulation for all stocks in portfolio')
    parser.add_argument('--days', type=int, default=1,
                       help='Number of days to simulate (default: 1)')
    parser.add_argument('--investment', type=float, default=25000,
                       help='Investment amount per stock per day (default: ₹25,000)')
    
    # Optional arguments
    parser.add_argument('--sector', type=str,
                       help='Run simulation for specific sector (Technology, Banking, etc.)')
    parser.add_argument('--output', type=str,
                       help='Custom output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--predict-only', action='store_true',
                       help='Make predictions for tomorrow only (no validation)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Evaluate yesterday\'s predictions against today\'s actual prices')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.ticker, args.tickers, args.portfolio, args.sector]) and not args.evaluate_only:
        print("❌ Error: Must specify --ticker, --tickers, --portfolio, or --sector")
        parser.print_help()
        return
    
    # Handle evaluation-only mode
    if args.evaluate_only:
        simulator = RealTradingSimulator(investment_per_stock=args.investment)
        try:
            result = simulator.evaluate_previous_predictions()
            print(f"\n✅ Evaluation completed successfully!")
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        return
    
    # Determine stocks to simulate
    stocks = None
    
    if args.ticker:
        stocks = [args.ticker]
        
    elif args.tickers:
        stocks = args.tickers
        
    elif args.sector:
        if args.sector in INDIAN_STOCKS_BY_SECTOR:
            stocks = INDIAN_STOCKS_BY_SECTOR[args.sector]
        else:
            print(f"❌ Error: Sector '{args.sector}' not found")
            print(f"Available sectors: {', '.join(INDIAN_STOCKS_BY_SECTOR.keys())}")
            return
            
    elif args.portfolio:
        stocks = None  # Use all stocks
    
    # Initialize simulator
    simulator = RealTradingSimulator(investment_per_stock=args.investment, stocks=stocks)
    
    # Handle predict-only mode
    if args.predict_only:
        try:
            result = simulator.make_predictions_only()
            print(f"\n✅ Predictions completed successfully!")
        except Exception as e:
            print(f"\n❌ Prediction failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        return
    
    # Display simulation info
    if stocks:
        if len(stocks) == 1:
            print(f"🎯 Single Stock Simulation: {stocks[0]}")
        else:
            print(f"📈 Multi-Stock Simulation: {', '.join(stocks)}")
    else:
        print(f"💼 Full Portfolio Simulation: All stocks")
    
    # Run simulation
    try:
        if args.predict_only:
            simulator.make_predictions_only()
        elif args.evaluate_only:
            simulator.evaluate_previous_predictions()
        else:
            simulator.run_simulation(days=args.days)
        
        print(f"\n✅ Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Simulation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
