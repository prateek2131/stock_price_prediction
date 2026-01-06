#!/usr/bin/env python3
"""
Daily Stock Predictions Automation
Runs predictions for all stocks and saves results to files (Git operations handled by CI/CD)
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import INDIAN_STOCKS_BY_SECTOR
from real_trading_simulator import RealTradingSimulator

class DailyPredictionAutomator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.predictions_dir = self.base_dir / "daily_predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.predictions_dir / "json").mkdir(exist_ok=True)
        (self.predictions_dir / "csv").mkdir(exist_ok=True)
        (self.predictions_dir / "summary").mkdir(exist_ok=True)
        
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
    def get_all_stocks(self):
        """Get all stocks from all sectors"""
        stocks = []
        for sector_stocks in INDIAN_STOCKS_BY_SECTOR.values():
            stocks.extend(sector_stocks)
        return sorted(list(set(stocks)))  # Remove duplicates and sort
    
    def run_daily_predictions(self):
        """Run predictions for all stocks"""
        print(f"🚀 Running Daily Predictions for {self.tomorrow}")
        print(f"📅 Prediction Date: {self.today}")
        print("=" * 70)
        
        all_stocks = self.get_all_stocks()
        
        # Initialize simulator for all stocks
        simulator = RealTradingSimulator(stocks=all_stocks)
        
        # Run predictions
        result = simulator.make_predictions_only()
        
        if result['predictions']:
            # Save detailed JSON
            json_file = self.predictions_dir / "json" / f"predictions_{self.today}.json"
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Create CSV summary
            self.create_csv_summary(result['predictions'])
            
            # Note: Daily summary will be created later with evaluation data
            
            print(f"\n✅ Predictions completed for {len(result['predictions'])} stocks")
            return True, result['predictions']
        else:
            print("\n❌ No predictions generated")
            return False, []
    
    def run_trading_evaluation(self):
        """Evaluate yesterday's predictions if available"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n📊 Evaluating Trading Performance from {yesterday}")
        print("=" * 70)
        
        # Check if we have yesterday's predictions to evaluate
        predictions_file = self.predictions_dir / "json" / f"predictions_{yesterday}.json"
        
        if predictions_file.exists():
            # Initialize simulator for evaluation
            simulator = RealTradingSimulator()
            
            # Run evaluation
            eval_result = simulator.evaluate_previous_predictions()
            
            if eval_result['evaluated']:
                print(f"✅ Evaluated {len(eval_result['evaluated'])} predictions")
                return True, eval_result
            else:
                print("⚠️  No evaluations could be performed")
                return False, {}
        else:
            print(f"⚠️  No predictions from {yesterday} found - skipping evaluation")
            return False, {}
    
    def run_complete_daily_automation(self):
        """Run both predictions and trading evaluation"""
        print("🤖 COMPLETE DAILY AUTOMATION")
        print("=" * 50)
        
        results = {
            'predictions': None,
            'evaluation': None,
            'date': self.today
        }
        
        # Step 1: Evaluate yesterday's predictions (trading performance)
        eval_success, eval_data = self.run_trading_evaluation()
        if eval_success:
            results['evaluation'] = eval_data
        
        # Step 2: Make today's predictions for tomorrow
        pred_success, pred_data = self.run_daily_predictions()
        if pred_success:
            results['predictions'] = pred_data
            
            # Create daily summary with both prediction and evaluation data
            evaluation_data = results.get('evaluation', None)
            self.create_daily_summary(pred_data, evaluation_data)
        
        return results
    
    def create_csv_summary(self, predictions):
        """Create CSV file with predictions from all models"""
        # Flatten the predictions to include all models
        flattened_predictions = []
        
        for pred in predictions:
            ticker = pred['ticker']
            prediction_date = pred['prediction_date']
            target_date = pred['target_date']
            
            # Add a row for each model
            for model_name, model_pred in pred['models'].items():
                row = {
                    'ticker': ticker,
                    'model': model_name,
                    'prediction_date': prediction_date,
                    'target_date': target_date,
                    'current_price': model_pred['current_price'],
                    'predicted_price': model_pred['predicted_price'],
                    'predicted_change': model_pred['predicted_change'],
                    'predicted_change_pct': model_pred['predicted_change_pct'],
                    'predicted_direction': model_pred['predicted_direction'],
                    'confidence': model_pred['confidence']
                }
                flattened_predictions.append(row)
        
        df = pd.DataFrame(flattened_predictions)
        
        # Reorder columns for better readability
        columns = [
            'ticker', 'model', 'current_price', 'predicted_price', 
            'predicted_change', 'predicted_change_pct', 'predicted_direction',
            'confidence', 'prediction_date', 'target_date'
        ]
        
        df = df.reindex(columns=columns)
        
        # Save CSV
        csv_file = self.predictions_dir / "csv" / f"predictions_{self.today}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"📊 CSV saved: {csv_file} ({len(flattened_predictions)} model predictions)")
        
        # Also create a summary CSV with just ensemble predictions for quick reference
        ensemble_predictions = []
        for pred in predictions:
            if 'ensemble' in pred['models']:
                ensemble_pred = pred['models']['ensemble']
                row = {
                    'ticker': pred['ticker'],
                    'current_price': ensemble_pred['current_price'],
                    'predicted_price': ensemble_pred['predicted_price'],
                    'predicted_change': ensemble_pred['predicted_change'],
                    'predicted_change_pct': ensemble_pred['predicted_change_pct'],
                    'confidence': ensemble_pred['confidence']
                }
                ensemble_predictions.append(row)
        
        if ensemble_predictions:
            ensemble_df = pd.DataFrame(ensemble_predictions)
            ensemble_file = self.predictions_dir / "csv" / f"ensemble_summary_{self.today}.csv"
            ensemble_df.to_csv(ensemble_file, index=False)
            print(f"📊 Ensemble summary CSV: {ensemble_file}")
    
    def create_daily_summary(self, predictions, evaluation_data=None):
        """Create human-readable daily summary"""
        summary_file = self.predictions_dir / "summary" / f"summary_{self.today}.md"
        
        # Get ensemble predictions for statistics
        ensemble_predictions = []
        for pred in predictions:
            if 'ensemble' in pred['models']:
                ensemble_pred = pred['models']['ensemble']
                ensemble_pred['ticker'] = pred['ticker']
                ensemble_predictions.append(ensemble_pred)
        
        # Calculate statistics based on ensemble predictions
        total_stocks = len(ensemble_predictions)
        bullish = len([p for p in ensemble_predictions if p['predicted_change'] > 0])
        bearish = len([p for p in ensemble_predictions if p['predicted_change'] < 0])
        neutral = total_stocks - bullish - bearish
        
        avg_change = sum(p['predicted_change_pct'] for p in ensemble_predictions) / total_stocks if ensemble_predictions else 0
        
        # Sort by predicted change percentage (ensemble)
        sorted_ensemble = sorted(ensemble_predictions, key=lambda x: x['predicted_change_pct'], reverse=True)
        
        # Create markdown content
        content = f"""# Daily Stock Predictions - {self.today}

## Summary
- **Prediction Date**: {self.today}
- **Target Date**: {self.tomorrow}
- **Total Stocks**: {total_stocks}
- **Models Used**: Baseline LSTM, MSLSTM, MSLSTMA, Stacked Ensemble
- **Bullish Predictions**: {bullish} ({bullish/total_stocks*100:.1f}%)
- **Bearish Predictions**: {bearish} ({bearish/total_stocks*100:.1f}%)
- **Neutral Predictions**: {neutral} ({neutral/total_stocks*100:.1f}%)
- **Average Expected Change**: {avg_change:+.2f}%

## Top 10 Bullish Predictions
| Ticker | Current Price | Predicted Price | Expected Change | Change % |
|--------|---------------|-----------------|-----------------|----------|
"""
        
        # Add top 10 bullish
        for pred in sorted_ensemble[:10]:
            content += f"| {pred['ticker']} | ₹{pred['current_price']:.2f} | ₹{pred['predicted_price']:.2f} | ₹{pred['predicted_change']:+.2f} | {pred['predicted_change_pct']:+.2f}% |\n"
        
        content += f"""
## Top 10 Bearish Predictions
| Ticker | Current Price | Predicted Price | Expected Change | Change % |
|--------|---------------|-----------------|-----------------|----------|
"""
        
        # Add top 10 bearish
        for pred in sorted_ensemble[-10:]:
            content += f"| {pred['ticker']} | ₹{pred['current_price']:.2f} | ₹{pred['predicted_price']:.2f} | ₹{pred['predicted_change']:+.2f} | {pred['predicted_change_pct']:+.2f}% |\n"
        
        content += f"""
## Model Comparison (Sample - First 5 Stocks)
| Ticker | Baseline | MSLSTM | MSLSTMA | Ensemble |
|--------|----------|---------|----------|----------|
"""
        
        # Add model comparison for first 5 stocks
        for i, pred in enumerate(predictions[:5]):
            ticker = pred['ticker']
            row = f"| {ticker} | "
            for model in ['baseline', 'mslstm', 'mslstma', 'ensemble']:
                if model in pred['models']:
                    change_pct = pred['models'][model]['predicted_change_pct']
                    row += f"{change_pct:+.2f}% | "
                else:
                    row += "N/A | "
            content += row + "\n"
        
        content += f"""
## Sector-wise Breakdown (Based on Ensemble Predictions)
"""
        
        # Group by sector
        sector_summary = {}
        for sector, sector_stocks in INDIAN_STOCKS_BY_SECTOR.items():
            sector_preds = [p for p in ensemble_predictions if p['ticker'] in sector_stocks]
            if sector_preds:
                avg_sector_change = sum(p['predicted_change_pct'] for p in sector_preds) / len(sector_preds)
                sector_summary[sector] = {
                    'count': len(sector_preds),
                    'avg_change': avg_sector_change,
                    'bullish': len([p for p in sector_preds if p['predicted_change'] > 0])
                }
        
        for sector, stats in sorted(sector_summary.items(), key=lambda x: x[1]['avg_change'], reverse=True):
            content += f"- **{sector}**: {stats['count']} stocks, {stats['avg_change']:+.2f}% avg, {stats['bullish']} bullish\n"
        
        # Add evaluation results if available
        if evaluation_data and 'summary' in evaluation_data and evaluation_data['summary']:
            summary = evaluation_data['summary']
            content += f"""
## Previous Day Trading Performance

**Evaluation Date**: {summary.get('date', 'Unknown')}  
**Total Predictions Evaluated**: {summary.get('total_predictions', 0)}  
**Overall ROI**: {summary.get('overall_roi_pct', 0):+.2f}%  
**Direction Accuracy**: {summary.get('direction_accuracy', 0):.1f}%  

### Model Performance Comparison
| Model | Predictions | P/L (₹) | ROI (%) | Accuracy (%) |
|-------|------------|---------|---------|--------------|
"""
            
            if 'model_performance' in summary:
                for model_name, stats in summary['model_performance'].items():
                    content += f"| {model_name.title()} | {stats['predictions']} | ₹{stats['profit_loss']:+,.2f} | {stats['roi_pct']:+.2f}% | {stats['direction_accuracy']:.1f}% |\n"
            
            content += f"""
**Best Performing Model**: {max(summary.get('model_performance', {}).items(), key=lambda x: x[1]['roi_pct'])[0].title() if summary.get('model_performance') else 'N/A'}  
**Total P/L**: ₹{summary.get('total_profit_loss', 0):+,.2f}  
"""
        
        content += f"""
---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by MSLSTM Prediction Framework*
"""
        
        # Save summary
        with open(summary_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Summary saved: {summary_file}")

def main():
    """Main function to run daily automation"""
    print("🤖 DAILY PREDICTIONS & TRADING AUTOMATION")
    print("=" * 60)
    
    automator = DailyPredictionAutomator()
    
    # Run complete automation (predictions + evaluation)
    results = automator.run_complete_daily_automation()
    
    # Report results
    has_results = results['predictions'] or results['evaluation']
    
    if has_results:
        print(f"\n🎉 Daily automation completed successfully!")
        if results['predictions']:
            print(f"📈 Generated predictions for {len(results['predictions'])} stocks")
        if results['evaluation']:
            print(f"📊 Evaluated {len(results['evaluation']['evaluated'])} previous predictions")
        print(f"� All results saved to files (Git operations handled by CI/CD)")
    else:
        print(f"\n❌ No results generated - automation failed")

if __name__ == "__main__":
    main()
