"""
MSLSTM Model Comparison Framework
=================================

Comprehensive comparison of all paper implementation models:
- Baseline LSTM (your original implementation)
- MSLSTM (Multi-Sequential LSTM from paper)
- MSLSTMA (Multi-Sequential LSTM Autoencoder from paper)
- Stacked Ensemble (TCN + WaveNet + LSTM + Attention-LSTM)

Features:
- Side-by-side training and evaluation
- Statistical significance testing
- Prediction visualization
- Performance analysis
- Export results for publication

Author: Research Implementation
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical testing
from scipy import stats
from scipy.stats import ttest_rel

# Model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MSLSTM_paper_implementation import (
    PaperModelTrainer, MSLSTMPaper, MSLSTMAPaper, 
    BaselineLSTM, StackedEnsemble, StockFeatureEngineer
)
from config import DATA_CONFIG, PATHS, INDIAN_STOCKS_BY_SECTOR

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelComparison:
    """
    Comprehensive model comparison framework
    """
    
    def __init__(self, ticker='TCS.NS', epochs=50, runs=5):
        self.ticker = ticker
        self.epochs = epochs
        self.runs = runs
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.models = {}
        
        # Model types
        self.model_types = ['baseline', 'mslstm', 'mslstma', 'ensemble']
        self.model_names = {
            'baseline': 'Baseline LSTM',
            'mslstm': 'MSLSTM (Paper)',
            'mslstma': 'MSLSTMA (Paper)',
            'ensemble': 'Stacked Ensemble'
        }
        
        print(f"\n{'='*80}")
        print(f"MSLSTM MODEL COMPARISON FRAMEWORK")
        print(f"Ticker: {ticker} | Epochs: {epochs} | Runs: {runs}")
        print('='*80)
    
    def train_all_models(self):
        """Train all models multiple times for statistical significance"""
        print(f"\n🚀 Training all models ({self.runs} runs each)...")
        
        for model_type in self.model_types:
            print(f"\n{'─'*60}")
            print(f"📊 Training {self.model_names[model_type]}")
            print('─'*60)
            
            self.results[model_type] = []
            self.predictions[model_type] = []
            
            for run in range(self.runs):
                print(f"\n--- Run {run+1}/{self.runs} ---")
                
                try:
                    # Create trainer
                    trainer = PaperModelTrainer(self.ticker, model_type)
                    
                    # Train model
                    trainer.load_and_prepare_data()
                    trainer.train(epochs=self.epochs)
                    
                    # Evaluate
                    metrics = trainer.evaluate()
                    
                    # Store results
                    self.results[model_type].append(metrics)
                    self.predictions[model_type].append({
                        'predictions': metrics['predictions'],
                        'actuals': metrics['actuals']
                    })
                    
                    # Store best model (from last run)
                    if run == self.runs - 1:
                        self.models[model_type] = trainer
                    
                    print(f"✓ Run {run+1} complete - Dir.Acc: {metrics['Directional_Accuracy']:.2f}%")
                    
                except Exception as e:
                    print(f" Run {run+1} failed: {e}")
                    # Add placeholder for failed run
                    self.results[model_type].append(None)
                    self.predictions[model_type].append(None)
        
        print(f"\n All models trained!")
    
    def calculate_statistics(self):
        """Calculate mean, std, and confidence intervals"""
        print(f"\n📊 Calculating statistics...")
        
        self.statistics = {}
        metrics_list = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
        
        for model_type in self.model_types:
            self.statistics[model_type] = {}
            
            # Get valid results (non-None)
            valid_results = [r for r in self.results[model_type] if r is not None]
            
            if not valid_results:
                print(f"⚠️  No valid results for {model_type}")
                continue
            
            for metric in metrics_list:
                if metric in valid_results[0]:
                    values = [r[metric] for r in valid_results]
                    
                    self.statistics[model_type][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'runs': len(values)
                    }
        
        print(" Statistics calculated!")
    
    def significance_testing(self):
        """Perform statistical significance testing between models"""
        print(f"\n🔬 Statistical significance testing...")
        
        self.pairwise_tests = {}
        
        # Compare each pair of models
        for i, model1 in enumerate(self.model_types):
            for j, model2 in enumerate(self.model_types[i+1:], i+1):
                pair_key = f"{model1}_vs_{model2}"
                self.pairwise_tests[pair_key] = {}
                
                # Get directional accuracy values
                acc1 = [r['Directional_Accuracy'] for r in self.results[model1] if r is not None]
                acc2 = [r['Directional_Accuracy'] for r in self.results[model2] if r is not None]
                
                if len(acc1) > 1 and len(acc2) > 1:
                    # Paired t-test (if same number of runs)
                    if len(acc1) == len(acc2):
                        statistic, p_value = ttest_rel(acc1, acc2)
                        test_type = "Paired t-test"
                    else:
                        # Independent t-test
                        statistic, p_value = stats.ttest_ind(acc1, acc2)
                        test_type = "Independent t-test"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(acc1)-1)*np.std(acc1)**2 + (len(acc2)-1)*np.std(acc2)**2) / 
                                       (len(acc1) + len(acc2) - 2))
                    effect_size = (np.mean(acc1) - np.mean(acc2)) / pooled_std
                    
                    self.pairwise_tests[pair_key] = {
                        'model1': self.model_names[model1],
                        'model2': self.model_names[model2],
                        'mean1': np.mean(acc1),
                        'mean2': np.mean(acc2),
                        'statistic': statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'test_type': test_type,
                        'significant': p_value < 0.05
                    }
        
        print(" Significance testing complete!")
    
    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        print(f"\n{'='*120}")
        print(f"MODEL COMPARISON RESULTS - {self.ticker}")
        print('='*120)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_type in self.model_types:
            if model_type in self.statistics:
                stats = self.statistics[model_type]
                
                row = {
                    'Model': self.model_names[model_type],
                    'Runs': stats.get('Directional_Accuracy', {}).get('runs', 0)
                }
                
                # Add metrics with mean ± std
                for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']:
                    if metric in stats:
                        mean_val = stats[metric]['mean']
                        std_val = stats[metric]['std']
                        
                        if metric in ['RMSE', 'MAE']:
                            row[f'{metric}'] = f"{mean_val:.4f} ± {std_val:.4f}"
                        elif metric in ['MAPE', 'Directional_Accuracy']:
                            row[f'{metric} (%)'] = f"{mean_val:.2f} ± {std_val:.2f}"
                        elif metric == 'R2':
                            row['R²'] = f"{mean_val:.4f} ± {std_val:.4f}"
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Statistical significance results
        if hasattr(self, 'pairwise_tests'):
            print(f"\n{'='*120}")
            print("STATISTICAL SIGNIFICANCE TESTING")
            print('='*120)
            print(f"{'Comparison':<40} {'Mean Diff':>12} {'p-value':>10} {'Significant':>12} {'Effect Size':>12}")
            print('─'*120)
            
            for pair_key, test in self.pairwise_tests.items():
                diff = test['mean1'] - test['mean2']
                significant = "YES" if test['significant'] else "No"
                
                print(f"{test['model1']} vs {test['model2']:<25} "
                      f"{diff:>12.2f} {test['p_value']:>10.4f} "
                      f"{significant:>12} {test['effect_size']:>12.2f}")
        
        return comparison_df
    
    def plot_results(self, save_path=None):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Comparison Results - {self.ticker}', fontsize=16, fontweight='bold')
        
        # 1. Directional Accuracy Box Plot
        acc_data = []
        labels = []
        for model_type in self.model_types:
            if model_type in self.results:
                accs = [r['Directional_Accuracy'] for r in self.results[model_type] if r is not None]
                if accs:
                    acc_data.append(accs)
                    labels.append(self.model_names[model_type])
        
        if acc_data:
            axes[0, 0].boxplot(acc_data, labels=labels)
            axes[0, 0].set_title('Directional Accuracy Distribution')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE Comparison
        rmse_means = []
        rmse_stds = []
        model_labels = []
        
        for model_type in self.model_types:
            if model_type in self.statistics and 'RMSE' in self.statistics[model_type]:
                rmse_means.append(self.statistics[model_type]['RMSE']['mean'])
                rmse_stds.append(self.statistics[model_type]['RMSE']['std'])
                model_labels.append(self.model_names[model_type])
        
        if rmse_means:
            x_pos = np.arange(len(model_labels))
            axes[0, 1].bar(x_pos, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(model_labels, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. R² Comparison
        r2_means = []
        r2_stds = []
        r2_labels = []
        
        for model_type in self.model_types:
            if model_type in self.statistics and 'R2' in self.statistics[model_type]:
                r2_means.append(self.statistics[model_type]['R2']['mean'])
                r2_stds.append(self.statistics[model_type]['R2']['std'])
                r2_labels.append(self.model_names[model_type])
        
        if r2_means:
            x_pos = np.arange(len(r2_labels))
            axes[0, 2].bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
            axes[0, 2].set_title('R² Comparison')
            axes[0, 2].set_ylabel('R²')
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels(r2_labels, rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Predictions vs Actuals (Best model from each type)
        best_model = max(self.model_types, 
                        key=lambda x: self.statistics.get(x, {}).get('Directional_Accuracy', {}).get('mean', 0))
        
        if best_model in self.predictions and self.predictions[best_model]:
            best_pred = self.predictions[best_model][-1]  # Last run
            if best_pred and best_pred['predictions'] is not None:
                actuals = best_pred['actuals']
                predictions = best_pred['predictions']
                
                # Plot time series
                time_idx = np.arange(len(actuals))
                axes[1, 0].plot(time_idx, actuals, label='Actual', linewidth=2, alpha=0.8)
                axes[1, 0].plot(time_idx, predictions, label='Predicted', linewidth=2, alpha=0.8)
                axes[1, 0].set_title(f'Price Predictions - {self.model_names[best_model]}')
                axes[1, 0].set_xlabel('Time Steps')
                axes[1, 0].set_ylabel('Price')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Scatter Plot (Predicted vs Actual)
        if best_model in self.predictions and self.predictions[best_model]:
            best_pred = self.predictions[best_model][-1]
            if best_pred and best_pred['predictions'] is not None:
                actuals = best_pred['actuals']
                predictions = best_pred['predictions']
                
                axes[1, 1].scatter(actuals, predictions, alpha=0.6)
                
                # Perfect prediction line
                min_val = min(actuals.min(), predictions.min())
                max_val = max(actuals.max(), predictions.max())
                axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
                
                axes[1, 1].set_title('Predicted vs Actual Prices')
                axes[1, 1].set_xlabel('Actual Price')
                axes[1, 1].set_ylabel('Predicted Price')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Comparison (Bar Chart instead of Radar)
        if len(self.statistics) >= 2:
            # Create a simple bar chart comparing key metrics
            metrics_data = []
            model_names_list = []
            
            for model_type in list(self.model_types)[:4]:  # Limit to 4 for readability
                if model_type in self.statistics:
                    stats = self.statistics[model_type]
                    
                    # Get key metrics
                    dir_acc = stats.get('Directional_Accuracy', {}).get('mean', 0)
                    r2 = stats.get('R2', {}).get('mean', 0) * 100  # Scale to percentage
                    
                    metrics_data.append([dir_acc, r2])
                    model_names_list.append(self.model_names[model_type])
            
            if metrics_data:
                metrics_array = np.array(metrics_data)
                
                # Create grouped bar chart
                x_pos = np.arange(len(model_names_list))
                width = 0.35
                
                axes[1, 2].bar(x_pos - width/2, metrics_array[:, 0], width, 
                              label='Directional Accuracy (%)', alpha=0.7, color='blue')
                axes[1, 2].bar(x_pos + width/2, metrics_array[:, 1], width, 
                              label='R² Score (×100)', alpha=0.7, color='red')
                
                axes[1, 2].set_title('Key Performance Metrics')
                axes[1, 2].set_xlabel('Models')
                axes[1, 2].set_ylabel('Score (%)')
                axes[1, 2].set_xticks(x_pos)
                axes[1, 2].set_xticklabels(model_names_list, rotation=45)
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        else:
            # If not enough data, show a simple text message
            axes[1, 2].text(0.5, 0.5, 'Not enough models\nfor comparison', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12)
            axes[1, 2].set_title('Performance Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, filename=None):
        """Save all results to files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_model_comparison_{timestamp}"
        
        results_dir = PATHS['results']
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Detailed results (all runs)
        detailed_path = os.path.join(results_dir, f"{filename}_detailed.csv")
        detailed_data = []
        
        for model_type in self.model_types:
            if model_type in self.results:
                for run_idx, result in enumerate(self.results[model_type]):
                    if result is not None:
                        row = {
                            'Model': self.model_names[model_type],
                            'Run': run_idx + 1,
                            'Ticker': self.ticker,
                            **result
                        }
                        # Remove complex objects
                        row.pop('predictions', None)
                        row.pop('actuals', None)
                        detailed_data.append(row)
        
        if detailed_data:
            pd.DataFrame(detailed_data).to_csv(detailed_path, index=False)
            print(f"💾 Detailed results saved: {detailed_path}")
        
        # 2. Summary statistics
        summary_path = os.path.join(results_dir, f"{filename}_summary.csv")
        summary_data = []
        
        for model_type in self.model_types:
            if model_type in self.statistics:
                row = {'Model': self.model_names[model_type], 'Ticker': self.ticker}
                
                for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']:
                    if metric in self.statistics[model_type]:
                        stats = self.statistics[model_type][metric]
                        row[f'{metric}_Mean'] = stats['mean']
                        row[f'{metric}_Std'] = stats['std']
                        row[f'{metric}_Min'] = stats['min']
                        row[f'{metric}_Max'] = stats['max']
                
                summary_data.append(row)
        
        if summary_data:
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            print(f"📊 Summary statistics saved: {summary_path}")
        
        # 3. Statistical significance tests
        if hasattr(self, 'pairwise_tests'):
            sig_path = os.path.join(results_dir, f"{filename}_significance.csv")
            sig_data = []
            
            for pair_key, test in self.pairwise_tests.items():
                sig_data.append({
                    'Comparison': f"{test['model1']} vs {test['model2']}",
                    'Model1_Mean': test['mean1'],
                    'Model2_Mean': test['mean2'],
                    'Mean_Difference': test['mean1'] - test['mean2'],
                    'Test_Statistic': test['statistic'],
                    'P_Value': test['p_value'],
                    'Effect_Size': test['effect_size'],
                    'Significant': test['significant'],
                    'Test_Type': test['test_type'],
                    'Ticker': self.ticker
                })
            
            if sig_data:
                pd.DataFrame(sig_data).to_csv(sig_path, index=False)
                print(f"🔬 Significance tests saved: {sig_path}")
        
        print(f"\n All results saved with prefix: {filename}")
    
    def run_full_comparison(self, save_plots=True):
        """Run complete comparison pipeline"""
        print(f"\n🚀 Starting full model comparison for {self.ticker}...")
        
        # Step 1: Train all models
        self.train_all_models()
        
        # Step 2: Calculate statistics
        self.calculate_statistics()
        
        # Step 3: Statistical significance testing
        self.significance_testing()
        
        # Step 4: Print results
        comparison_df = self.print_comparison_table()
        
        # Step 5: Create visualizations
        if save_plots:
            plot_path = os.path.join(PATHS['results'], 
                                   f"{self.ticker}_model_comparison.png")
            self.plot_results(save_path=plot_path)
        else:
            self.plot_results()
        
        # Step 6: Save results
        self.save_results()
        
        print(f"\n🎉 Model comparison complete for {self.ticker}!")
        return comparison_df


def run_single_ticker(ticker='TCS.NS', epochs=50, runs=3):
    """Run comparison for a single ticker"""
    comparison = ModelComparison(ticker=ticker, epochs=epochs, runs=runs)
    return comparison.run_full_comparison()


def run_sector_comparison(sector='Technology', epochs=50, runs=3):
    """Run comparison for all tickers in a sector"""
    if sector not in INDIAN_STOCKS_BY_SECTOR:
        print(f" Sector '{sector}' not found. Available sectors:")
        for s in INDIAN_STOCKS_BY_SECTOR.keys():
            print(f"   - {s}")
        return None
    
    tickers = INDIAN_STOCKS_BY_SECTOR[sector]
    print(f"\n{'='*80}")
    print(f"SECTOR COMPARISON: {sector.upper()}")
    print(f"Tickers: {', '.join(tickers)}")
    print('='*80)
    
    all_results = {}
    
    for ticker in tickers:
        print(f"\n{'─'*60}")
        print(f"Processing {ticker}...")
        print('─'*60)
        
        try:
            comparison = ModelComparison(ticker=ticker, epochs=epochs, runs=runs)
            result = comparison.run_full_comparison(save_plots=True)
            all_results[ticker] = result
            
        except Exception as e:
            print(f" Error processing {ticker}: {e}")
            all_results[ticker] = None
    
    # Create sector summary
    create_sector_summary(sector, all_results)
    
    return all_results


def create_sector_summary(sector, all_results):
    """Create consolidated summary for sector"""
    print(f"\n{'='*100}")
    print(f"SECTOR SUMMARY: {sector.upper()}")
    print('='*100)
    
    summary_data = []
    
    for ticker, results in all_results.items():
        if results is not None:
            # Find best model for this ticker
            best_idx = results['Directional_Accuracy (%)'].str.extract('(\d+\.?\d*)')[0].astype(float).idxmax()
            best_model = results.iloc[best_idx]
            
            summary_data.append({
                'Ticker': ticker,
                'Best_Model': best_model['Model'],
                'Direction_Accuracy': best_model['Directional_Accuracy (%)'],
                'RMSE': best_model['RMSE'],
                'R²': best_model['R²']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save sector summary
        sector_path = os.path.join(PATHS['results'], f"{sector}_sector_summary.csv")
        summary_df.to_csv(sector_path, index=False)
        print(f"\n💾 Sector summary saved: {sector_path}")
    
    print(f"\n Sector comparison complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MSLSTM Model Comparison Framework')
    parser.add_argument('--ticker', type=str, default='TCS.NS', 
                       help='Stock ticker to analyze')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Training epochs per model')
    parser.add_argument('--runs', type=int, default=3, 
                       help='Number of runs for statistical significance')
    parser.add_argument('--sector', type=str, default=None,
                       help='Run comparison for entire sector')
    parser.add_argument('--list-sectors', action='store_true',
                       help='List available sectors')
    
    args = parser.parse_args()
    
    if args.list_sectors:
        print("\nAvailable sectors:")
        for sector in INDIAN_STOCKS_BY_SECTOR.keys():
            print(f"  - {sector}")
        sys.exit(0)
    
    if args.sector:
        run_sector_comparison(args.sector, args.epochs, args.runs)
    else:
        run_single_ticker(args.ticker, args.epochs, args.runs)
