#!/usr/bin/env python3
"""
Daily Stock Predictions Automation
Runs predictions for all stocks and saves results to Git
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
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
            
            # Create daily summary
            self.create_daily_summary(result['predictions'])
            
            print(f"\n✅ Predictions completed for {len(result['predictions'])} stocks")
            return True
        else:
            print("\n❌ No predictions generated")
            return False
    
    def create_csv_summary(self, predictions):
        """Create CSV file with predictions"""
        df = pd.DataFrame(predictions)
        
        # Reorder columns for better readability
        columns = [
            'ticker', 'current_price', 'predicted_price', 
            'predicted_change', 'predicted_change_pct', 
            'confidence', 'prediction_date', 'target_date'
        ]
        
        df = df.reindex(columns=columns)
        
        # Save CSV
        csv_file = self.predictions_dir / "csv" / f"predictions_{self.today}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"📊 CSV saved: {csv_file}")
    
    def create_daily_summary(self, predictions):
        """Create human-readable daily summary"""
        summary_file = self.predictions_dir / "summary" / f"summary_{self.today}.md"
        
        # Calculate statistics
        total_stocks = len(predictions)
        bullish = len([p for p in predictions if p['predicted_change'] > 0])
        bearish = len([p for p in predictions if p['predicted_change'] < 0])
        neutral = total_stocks - bullish - bearish
        
        avg_change = sum(p['predicted_change_pct'] for p in predictions) / total_stocks if predictions else 0
        
        # Sort by predicted change percentage
        sorted_predictions = sorted(predictions, key=lambda x: x['predicted_change_pct'], reverse=True)
        
        # Create markdown content
        content = f"""# Daily Stock Predictions - {self.today}

## Summary
- **Prediction Date**: {self.today}
- **Target Date**: {self.tomorrow}
- **Total Stocks**: {total_stocks}
- **Bullish Predictions**: {bullish} ({bullish/total_stocks*100:.1f}%)
- **Bearish Predictions**: {bearish} ({bearish/total_stocks*100:.1f}%)
- **Neutral Predictions**: {neutral} ({neutral/total_stocks*100:.1f}%)
- **Average Expected Change**: {avg_change:+.2f}%

## Top 10 Bullish Predictions
| Ticker | Current Price | Predicted Price | Expected Change | Change % |
|--------|---------------|-----------------|-----------------|----------|
"""
        
        # Add top 10 bullish
        for pred in sorted_predictions[:10]:
            content += f"| {pred['ticker']} | ₹{pred['current_price']:.2f} | ₹{pred['predicted_price']:.2f} | ₹{pred['predicted_change']:+.2f} | {pred['predicted_change_pct']:+.2f}% |\n"
        
        content += f"""
## Top 10 Bearish Predictions
| Ticker | Current Price | Predicted Price | Expected Change | Change % |
|--------|---------------|-----------------|-----------------|----------|
"""
        
        # Add top 10 bearish
        for pred in sorted_predictions[-10:]:
            content += f"| {pred['ticker']} | ₹{pred['current_price']:.2f} | ₹{pred['predicted_price']:.2f} | ₹{pred['predicted_change']:+.2f} | {pred['predicted_change_pct']:+.2f}% |\n"
        
        content += f"""
## Sector-wise Breakdown
"""
        
        # Group by sector
        sector_summary = {}
        for sector, sector_stocks in INDIAN_STOCKS_BY_SECTOR.items():
            sector_preds = [p for p in predictions if p['ticker'] in sector_stocks]
            if sector_preds:
                avg_sector_change = sum(p['predicted_change_pct'] for p in sector_preds) / len(sector_preds)
                sector_summary[sector] = {
                    'count': len(sector_preds),
                    'avg_change': avg_sector_change,
                    'bullish': len([p for p in sector_preds if p['predicted_change'] > 0])
                }
        
        for sector, stats in sorted(sector_summary.items(), key=lambda x: x[1]['avg_change'], reverse=True):
            content += f"- **{sector}**: {stats['count']} stocks, {stats['avg_change']:+.2f}% avg, {stats['bullish']} bullish\n"
        
        content += f"""
---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by MSLSTM Prediction Framework*
"""
        
        # Save summary
        with open(summary_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Summary saved: {summary_file}")
    
    def commit_to_git(self):
        """Commit and push results to Git"""
        try:
            os.chdir(self.base_dir)
            
            # Add all prediction files
            subprocess.run(['git', 'add', 'daily_predictions/'], check=True)
            
            # Commit with date
            commit_msg = f"Daily predictions for {self.tomorrow} (generated {self.today})"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            # Push to remote
            subprocess.run(['git', 'push'], check=True)
            
            print(f"✅ Committed and pushed to Git: {commit_msg}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in Git operation: {e}")
            return False

def main():
    """Main function to run daily automation"""
    print("🤖 DAILY PREDICTIONS AUTOMATION")
    print("=" * 50)
    
    automator = DailyPredictionAutomator()
    
    # Run predictions
    success = automator.run_daily_predictions()
    
    if success:
        # Commit to Git
        git_success = automator.commit_to_git()
        
        if git_success:
            print(f"\n🎉 Daily automation completed successfully!")
            print(f"📅 Predictions for {automator.tomorrow} are now in Git")
        else:
            print(f"\n⚠️  Predictions generated but Git commit failed")
    else:
        print(f"\n❌ Daily automation failed")

if __name__ == "__main__":
    main()
