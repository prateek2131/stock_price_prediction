#!/usr/bin/env python3
"""
Comprehensive Model Performance & Daily Prediction Analysis
=========================================================
Analyze both overall model performance and daily trading performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import glob

# Add project directory to path
project_dir = Path(__file__).parent.absolute()
sys.path.append(str(project_dir))

from config import INDIAN_STOCKS_BY_SECTOR

def load_overall_results():
    """Load overall paper implementation results"""
    results_file = project_dir / "results" / "paper_implementation_results.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    return pd.read_csv(results_file)

def load_daily_predictions():
    """Load all daily prediction files"""
    daily_predictions = []
    ensemble_summaries = []
    
    # Load prediction CSV files
    csv_dir = project_dir / "daily_predictions" / "csv"
    
    # Get prediction files
    prediction_files = sorted(glob.glob(str(csv_dir / "predictions_*.csv")))
    ensemble_files = sorted(glob.glob(str(csv_dir / "ensemble_summary_*.csv")))
    
    for file_path in prediction_files:
        df = pd.read_csv(file_path)
        df['file_date'] = Path(file_path).stem.split('_')[1]
        daily_predictions.append(df)
    
    for file_path in ensemble_files:
        df = pd.read_csv(file_path)
        df['file_date'] = Path(file_path).stem.split('_')[2]
        ensemble_summaries.append(df)
    
    if daily_predictions:
        daily_df = pd.concat(daily_predictions, ignore_index=True)
    else:
        daily_df = pd.DataFrame()
        
    if ensemble_summaries:
        ensemble_df = pd.concat(ensemble_summaries, ignore_index=True)
    else:
        ensemble_df = pd.DataFrame()
    
    return daily_df, ensemble_df

def analyze_overall_performance(df):
    """Analyze overall model performance from paper implementation"""
    print("🎯 OVERALL MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Total Stocks Analyzed: {df['Ticker'].nunique()}")
    print(f"Sectors Covered: {df['Sector'].nunique()}")
    print(f"Models Evaluated: {', '.join(df['Model'].unique())}")
    print()
    
    # Group by model and calculate average metrics
    model_performance = df.groupby('Model').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'MAPE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'Directional_Accuracy': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    model_performance.columns = [f"{col[0]}_{col[1]}" for col in model_performance.columns]
    
    # Calculate rankings
    model_performance['R2_Rank'] = model_performance['R2_mean'].rank(ascending=False)
    model_performance['MAPE_Rank'] = model_performance['MAPE_mean'].rank(ascending=True)
    model_performance['Dir_Acc_Rank'] = model_performance['Directional_Accuracy_mean'].rank(ascending=False)
    
    print("📊 MODEL PERFORMANCE SUMMARY:")
    print("-" * 50)
    for model in model_performance.index:
        print(f"\n🔸 {model.upper()}:")
        print(f"   R² Score: {model_performance.loc[model, 'R2_mean']:.4f} ± {model_performance.loc[model, 'R2_std']:.4f}")
        print(f"   MAPE: {model_performance.loc[model, 'MAPE_mean']:.2f}% ± {model_performance.loc[model, 'MAPE_std']:.2f}%")
        print(f"   Directional Accuracy: {model_performance.loc[model, 'Directional_Accuracy_mean']:.1f}% ± {model_performance.loc[model, 'Directional_Accuracy_std']:.1f}%")
    
    # Best model by R²
    best_r2_model = model_performance['R2_mean'].idxmax()
    print(f"\n🏆 BEST R² PERFORMANCE: {best_r2_model} ({model_performance.loc[best_r2_model, 'R2_mean']:.4f})")
    
    # Best model by MAPE (lowest)
    best_mape_model = model_performance['MAPE_mean'].idxmin()
    print(f"🏆 BEST MAPE PERFORMANCE: {best_mape_model} ({model_performance.loc[best_mape_model, 'MAPE_mean']:.2f}%)")
    
    # Best model by directional accuracy
    best_dir_model = model_performance['Directional_Accuracy_mean'].idxmax()
    print(f"🏆 BEST DIRECTIONAL ACCURACY: {best_dir_model} ({model_performance.loc[best_dir_model, 'Directional_Accuracy_mean']:.1f}%)")
    
    return model_performance

def analyze_sector_performance(df):
    """Analyze performance by sector"""
    print(f"\n🏭 SECTOR-WISE PERFORMANCE ANALYSIS")
    print("-" * 45)
    
    # Create sector mapping for easier analysis
    sector_mapping = {}
    for sector, tickers in INDIAN_STOCKS_BY_SECTOR.items():
        for ticker in tickers:
            sector_mapping[ticker] = sector
    
    # Calculate sector performance for each model
    sector_performance = df.groupby(['Sector', 'Model']).agg({
        'R2': 'mean',
        'MAPE': 'mean',
        'Directional_Accuracy': 'mean'
    }).round(4)
    
    print("\n R² SCORES BY SECTOR AND MODEL:")
    r2_pivot = sector_performance['R2'].unstack()
    print(r2_pivot.to_string())
    
    print("\n📉 MAPE BY SECTOR AND MODEL:")
    mape_pivot = sector_performance['MAPE'].unstack()
    print(mape_pivot.to_string())
    
    # Best performing sectors
    print(f"\n🎖️ SECTOR RANKINGS (by average R² across all models):")
    sector_avg_r2 = df.groupby('Sector')['R2'].mean().sort_values(ascending=False)
    for i, (sector, r2) in enumerate(sector_avg_r2.items(), 1):
        print(f"{i:2d}. {sector:15s}: R² = {r2:.4f}")
    
    return sector_performance

def analyze_daily_predictions(daily_df, ensemble_df):
    """Analyze daily prediction performance"""
    print(f"\n📅 DAILY PREDICTION ANALYSIS")
    print("-" * 35)
    
    if daily_df.empty:
        print(" No daily prediction data available")
        return None, None
    
    print(f"📊 Daily Prediction Data Summary:")
    print(f"   Prediction Days: {daily_df['file_date'].nunique()}")
    print(f"   Total Predictions: {len(daily_df)}")
    print(f"   Stocks Covered: {daily_df['ticker'].nunique()}")
    print(f"   Models Used: {', '.join(daily_df['model'].unique())}")
    
    # Daily performance by model
    daily_model_perf = daily_df.groupby(['model', 'file_date']).agg({
        'predicted_change_pct': ['mean', 'std', 'count'],
        'confidence': 'mean'
    }).round(4)
    
    print(f"\n DAILY PREDICTION TRENDS BY MODEL:")
    for model in daily_df['model'].unique():
        model_data = daily_df[daily_df['model'] == model]
        
        print(f"\n🔸 {model.upper()}:")
        daily_stats = model_data.groupby('file_date').agg({
            'predicted_change_pct': ['mean', 'std'],
            'predicted_direction': ['mean', 'count']
        }).round(4)
        
        for date in daily_stats.index:
            avg_change = daily_stats.loc[date, ('predicted_change_pct', 'mean')]
            bullish_pct = daily_stats.loc[date, ('predicted_direction', 'mean')] * 100
            total_predictions = daily_stats.loc[date, ('predicted_direction', 'count')]
            
            print(f"   {date}: Avg Change: {avg_change:+.2f}%, Bullish: {bullish_pct:.1f}%, Predictions: {total_predictions}")
    
    # Ensemble confidence analysis
    if not ensemble_df.empty:
        print(f"\n🎯 ENSEMBLE CONFIDENCE ANALYSIS:")
        
        for date in ensemble_df['file_date'].unique():
            date_data = ensemble_df[ensemble_df['file_date'] == date]
            
            avg_confidence = date_data['confidence'].mean()
            high_conf_count = (date_data['confidence'] > 0.7).sum()
            bullish_count = (date_data['predicted_change_pct'] > 0).sum()
            
            print(f"   {date}:")
            print(f"      Average Confidence: {avg_confidence:.3f}")
            print(f"      High Confidence (>70%): {high_conf_count}/{len(date_data)} stocks")
            print(f"      Bullish Predictions: {bullish_count}/{len(date_data)} stocks")
            print(f"      Expected Portfolio Return: {date_data['predicted_change_pct'].mean():+.2f}%")
    
    return daily_df, ensemble_df

def analyze_prediction_consistency(daily_df):
    """Analyze prediction consistency across models and days"""
    print(f"\n🔄 PREDICTION CONSISTENCY ANALYSIS")
    print("-" * 40)
    
    if daily_df.empty:
        return
    
    # Model agreement analysis
    print("🤝 MODEL AGREEMENT ANALYSIS:")
    
    for date in daily_df['file_date'].unique():
        date_data = daily_df[daily_df['file_date'] == date]
        
        print(f"\n📅 {date}:")
        
        # Calculate agreement on direction for each stock
        stock_agreements = []
        for ticker in date_data['ticker'].unique():
            ticker_data = date_data[date_data['ticker'] == ticker]
            
            if len(ticker_data) >= 3:  # Need at least 3 models
                directions = ticker_data['predicted_direction'].values
                agreement = np.mean(directions)  # Proportion agreeing (since 1=bullish, 0=bearish)
                stock_agreements.append(agreement)
        
        if stock_agreements:
            avg_agreement = np.mean(stock_agreements)
            high_agreement = np.mean([(a > 0.8 or a < 0.2) for a in stock_agreements])
            
            print(f"   Average Model Agreement: {avg_agreement:.3f}")
            print(f"   High Agreement Stocks: {high_agreement*100:.1f}%")
    
    # Volatility of predictions
    print(f"\n📊 PREDICTION VOLATILITY BY MODEL:")
    
    model_volatility = daily_df.groupby('model').agg({
        'predicted_change_pct': ['std', 'mean'],
    }).round(4)
    
    for model in model_volatility.index:
        std = model_volatility.loc[model, ('predicted_change_pct', 'std')]
        mean = model_volatility.loc[model, ('predicted_change_pct', 'mean')]
        cv = std / abs(mean) if mean != 0 else float('inf')
        
        print(f"   {model.upper()}: StdDev = {std:.2f}%, CV = {cv:.2f}")

def trading_performance_simulation(ensemble_df):
    """Simulate trading performance based on ensemble predictions"""
    print(f"\n💰 TRADING PERFORMANCE SIMULATION")
    print("-" * 40)
    
    if ensemble_df.empty:
        print(" No ensemble data for trading simulation")
        return
    
    # Trading parameters
    initial_capital = 25000  # ₹25,000 per stock
    
    print(f"📊 Trading Simulation Parameters:")
    print(f"   Initial Capital per Stock: ₹{initial_capital:,}")
    print(f"   Trading Strategy: Buy/Sell based on ensemble predictions")
    
    total_results = []
    
    for date in sorted(ensemble_df['file_date'].unique()):
        date_data = ensemble_df[ensemble_df['file_date'] == date]
        
        print(f"\n📅 {date} Trading Simulation:")
        
        # Filter predictions by confidence (optional)
        high_conf_data = date_data[date_data['confidence'] > 0.3]  # Minimum 30% confidence
        
        if len(high_conf_data) == 0:
            print("    No high-confidence predictions")
            continue
        
        # Simulate trades
        total_investment = len(high_conf_data) * initial_capital
        daily_returns = []
        
        for _, row in high_conf_data.iterrows():
            predicted_return = row['predicted_change_pct'] / 100
            position_value = initial_capital * (1 + predicted_return)
            profit_loss = position_value - initial_capital
            daily_returns.append(profit_loss)
        
        total_pnl = sum(daily_returns)
        portfolio_return = total_pnl / total_investment * 100
        
        # Risk metrics
        winning_trades = sum(1 for ret in daily_returns if ret > 0)
        win_rate = winning_trades / len(daily_returns) * 100
        
        avg_win = np.mean([ret for ret in daily_returns if ret > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([ret for ret in daily_returns if ret <= 0]) if winning_trades < len(daily_returns) else 0
        
        print(f"   Total Investment: ₹{total_investment:,}")
        print(f"   Total P&L: ₹{total_pnl:+,.2f}")
        print(f"   Portfolio Return: {portfolio_return:+.2f}%")
        print(f"   Win Rate: {win_rate:.1f}% ({winning_trades}/{len(daily_returns)})")
        print(f"   Avg Win: ₹{avg_win:+.2f}, Avg Loss: ₹{avg_loss:+.2f}")
        
        total_results.append({
            'date': date,
            'total_investment': total_investment,
            'total_pnl': total_pnl,
            'portfolio_return': portfolio_return,
            'win_rate': win_rate,
            'trades': len(daily_returns)
        })
    
    # Overall simulation results
    if total_results:
        print(f"\n🏆 OVERALL TRADING SIMULATION RESULTS:")
        total_invested = sum(r['total_investment'] for r in total_results)
        total_profit = sum(r['total_pnl'] for r in total_results)
        overall_return = total_profit / total_invested * 100
        avg_win_rate = np.mean([r['win_rate'] for r in total_results])
        
        print(f"   Total Capital Deployed: ₹{total_invested:,}")
        print(f"   Total Profit/Loss: ₹{total_profit:+,.2f}")
        print(f"   Overall Return: {overall_return:+.2f}%")
        print(f"   Average Win Rate: {avg_win_rate:.1f}%")
        print(f"   Total Trading Days: {len(total_results)}")

def create_comprehensive_visualizations(df, daily_df, ensemble_df):
    """Create comprehensive visualizations"""
    print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # Create subplot grid
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Model Performance (R²)
    ax1 = fig.add_subplot(gs[0, 0])
    model_r2 = df.groupby('Model')['R2'].mean().sort_values(ascending=False)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars1 = ax1.bar(model_r2.index, model_r2.values, color=colors)
    ax1.set_title('Average R² Score by Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(model_r2.values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. MAPE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    model_mape = df.groupby('Model')['MAPE'].mean().sort_values()
    bars2 = ax2.bar(model_mape.index, model_mape.values, color=colors)
    ax2.set_title('Average MAPE by Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAPE (%)')
    for i, v in enumerate(model_mape.values):
        ax2.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Directional Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    dir_acc = df.groupby('Model')['Directional_Accuracy'].mean().sort_values(ascending=False)
    bars3 = ax3.bar(dir_acc.index, dir_acc.values, color=colors)
    ax3.set_title('Directional Accuracy by Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    for i, v in enumerate(dir_acc.values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Sector-wise R² Performance
    ax4 = fig.add_subplot(gs[1, :])
    sector_model_r2 = df.groupby(['Sector', 'Model'])['R2'].mean().unstack()
    sector_model_r2.plot(kind='bar', ax=ax4, color=colors, width=0.8)
    ax4.set_title('R² Score by Sector and Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('R² Score')
    ax4.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Daily Prediction Trends (if available)
    if not daily_df.empty:
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Daily average predicted changes by model
        daily_trends = daily_df.groupby(['file_date', 'model'])['predicted_change_pct'].mean().unstack()
        daily_trends.plot(ax=ax5, marker='o', linewidth=2)
        ax5.set_title('Daily Average Predicted Changes by Model', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Predicted Change (%)')
        ax5.set_xlabel('Date')
        ax5.legend(title='Model')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Ensemble Confidence Distribution
        if not ensemble_df.empty:
            ax6 = fig.add_subplot(gs[2, 2])
            confidence_data = ensemble_df['confidence'].dropna()
            ax6.hist(confidence_data, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
            ax6.set_title('Ensemble Confidence Distribution', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Confidence Score')
            ax6.set_ylabel('Frequency')
            ax6.axvline(confidence_data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {confidence_data.mean():.3f}')
            ax6.legend()
    
    # 7. Model Performance Heatmap
    ax7 = fig.add_subplot(gs[3, :])
    
    # Create performance matrix
    metrics = ['R2', 'MAPE', 'Directional_Accuracy']
    perf_matrix = df.groupby('Model')[metrics].mean()
    
    # Normalize for better visualization (R² and Dir_Acc higher is better, MAPE lower is better)
    perf_matrix_norm = perf_matrix.copy()
    perf_matrix_norm['MAPE'] = 1 / perf_matrix_norm['MAPE']  # Invert MAPE
    
    # Normalize to 0-1 scale
    for col in perf_matrix_norm.columns:
        perf_matrix_norm[col] = (perf_matrix_norm[col] - perf_matrix_norm[col].min()) / \
                               (perf_matrix_norm[col].max() - perf_matrix_norm[col].min())
    
    sns.heatmap(perf_matrix_norm.T, annot=True, cmap='RdYlGn', ax=ax7, 
                cbar_kws={'label': 'Normalized Performance (Higher = Better)'})
    ax7.set_title('Model Performance Heatmap (Normalized)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Model')
    ax7.set_ylabel('Metric')
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    output_path = project_dir / "results" / "comprehensive_analysis_report.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Comprehensive visualization saved: {output_path}")
    plt.close()

def generate_executive_summary(df, daily_df, ensemble_df):
    """Generate executive summary with key insights"""
    print(f"\n📋 EXECUTIVE SUMMARY & KEY INSIGHTS")
    print("=" * 50)
    
    # Overall best performers
    best_r2_model = df.groupby('Model')['R2'].mean().idxmax()
    best_r2_score = df.groupby('Model')['R2'].mean().max()
    
    best_mape_model = df.groupby('Model')['MAPE'].mean().idxmin()
    best_mape_score = df.groupby('Model')['MAPE'].mean().min()
    
    best_dir_model = df.groupby('Model')['Directional_Accuracy'].mean().idxmax()
    best_dir_score = df.groupby('Model')['Directional_Accuracy'].mean().max()
    
    print(f"🎯 KEY PERFORMANCE HIGHLIGHTS:")
    print(f"   • Best Price Prediction (R²): {best_r2_model} ({best_r2_score:.4f})")
    print(f"   • Best Accuracy (MAPE): {best_mape_model} ({best_mape_score:.2f}%)")
    print(f"   • Best Direction Prediction: {best_dir_model} ({best_dir_score:.1f}%)")
    
    # Sector insights
    sector_performance = df.groupby('Sector')['R2'].mean().sort_values(ascending=False)
    best_sector = sector_performance.index[0]
    worst_sector = sector_performance.index[-1]
    
    print(f"\n🏭 SECTOR INSIGHTS:")
    print(f"   • Best Performing Sector: {best_sector} (R²: {sector_performance.iloc[0]:.4f})")
    print(f"   • Most Challenging Sector: {worst_sector} (R²: {sector_performance.iloc[-1]:.4f})")
    
    # Daily prediction insights
    if not daily_df.empty and not ensemble_df.empty:
        avg_confidence = ensemble_df['confidence'].mean()
        bullish_ratio = (ensemble_df['predicted_change_pct'] > 0).mean() * 100
        
        print(f"\n📅 DAILY PREDICTION INSIGHTS:")
        print(f"   • Average Ensemble Confidence: {avg_confidence:.3f}")
        print(f"   • Bullish Predictions: {bullish_ratio:.1f}% of total")
        print(f"   • Prediction Days Analyzed: {daily_df['file_date'].nunique()}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use {best_r2_model} for price target predictions")
    print(f"   2. Use {best_dir_model} for directional trading signals")
    print(f"   3. Focus on {best_sector} sector for higher accuracy")
    print(f"   4. Ensemble approach shows balanced performance across metrics")
    
    if not ensemble_df.empty:
        high_conf_threshold = ensemble_df['confidence'].quantile(0.75)
        print(f"   5. Consider trades only above {high_conf_threshold:.2f} confidence threshold")

def main():
    """Main comprehensive analysis function"""
    print("🚀 STARTING COMPREHENSIVE MODEL & DAILY PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Load all data
    print(" Loading data...")
    overall_df = load_overall_results()
    daily_df, ensemble_df = load_daily_predictions()
    
    if overall_df is None:
        print(" Cannot proceed without overall results data")
        return
    
    # Run comprehensive analysis
    print("\n" + "="*70)
    
    # 1. Overall Performance Analysis
    model_performance = analyze_overall_performance(overall_df)
    
    # 2. Sector Analysis
    sector_performance = analyze_sector_performance(overall_df)
    
    # 3. Daily Prediction Analysis
    daily_analysis = analyze_daily_predictions(daily_df, ensemble_df)
    
    # 4. Prediction Consistency Analysis
    analyze_prediction_consistency(daily_df)
    
    # 5. Trading Simulation
    trading_performance_simulation(ensemble_df)
    
    # 6. Create Visualizations
    create_comprehensive_visualizations(overall_df, daily_df, ensemble_df)
    
    # 7. Executive Summary
    generate_executive_summary(overall_df, daily_df, ensemble_df)
    
    print(f"\n COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📊 Visualizations saved to: results/comprehensive_analysis_report.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
