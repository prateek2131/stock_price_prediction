"""
Diebold-Mariano (DM) Test for Forecast Comparison

The Diebold-Mariano test is used to compare the forecast accuracy of two competing models.
It tests whether one forecast is significantly more accurate than another.

H0: Both models have equal forecast accuracy
H1: One model has significantly better forecast accuracy

Test Statistic: DM ~ N(0,1) (approximately normally distributed under H0)
- If |DM| > 1.96 (5% level), reject H0 - models are significantly different
- If |DM| < 1.96, fail to reject H0 - models are not significantly different
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


class DieboldMarianoTest:
    """Diebold-Mariano Test for Forecast Comparison"""
    
    def __init__(self, evaluations_dir='evaluations'):
        self.evaluations_dir = Path(evaluations_dir)
        self.model_errors = {}
        self.model_prices = {}
        self.evaluation_dates = []
        self.dm_results = {}
        
    def load_evaluations(self):
        """Load all evaluation files"""
        print("📂 Loading evaluation files...")
        files = sorted(self.evaluations_dir.glob('evaluation_*.json'))
        
        for f in files:
            with open(f) as file:
                data = json.load(file)
            
            date = f.stem.replace('evaluation_', '')
            self.evaluation_dates.append(date)
            
            # Initialize model errors for this date
            for model in ['baseline', 'mslstm', 'mslstma', 'ensemble']:
                if model not in self.model_errors:
                    self.model_errors[model] = []
            
            # Extract errors for each model
            for entry in data.get('evaluated', []):
                model = entry.get('model')
                if model:
                    error = entry.get('price_error_pct', 0)
                    self.model_errors[model].append(error)
        
        print(f"✅ Loaded {len(files)} evaluation files")
        
    def diebold_mariano_statistic(self, 
                                   errors1: np.ndarray, 
                                   errors2: np.ndarray,
                                   loss_type: str = 'squared') -> Tuple[float, float, float]:
        """
        Calculate Diebold-Mariano test statistic
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            loss_type: 'squared', 'absolute', or 'squared_directional'
            
        Returns:
            dm_statistic: DM test statistic
            p_value: Two-tailed p-value
            dm_ratio: Relative accuracy (model1 vs model2)
        """
        # Ensure same length
        min_len = min(len(errors1), len(errors2))
        errors1 = errors1[:min_len]
        errors2 = errors2[:min_len]
        
        # Calculate loss function
        if loss_type == 'squared':
            loss1 = errors1 ** 2
            loss2 = errors2 ** 2
        elif loss_type == 'absolute':
            loss1 = np.abs(errors1)
            loss2 = np.abs(errors2)
        elif loss_type == 'squared_directional':
            loss1 = errors1 ** 2
            loss2 = errors2 ** 2
        else:
            raise ValueError("Invalid loss type")
        
        # DM test
        d = loss1 - loss2
        d_mean = np.mean(d)
        
        # Calculate variance (using Newey-West HAC estimator for robustness)
        var_d = self._newey_west_variance(d, max_lag=4)
        
        # DM statistic
        dm_stat = d_mean / np.sqrt(var_d / len(d)) if var_d > 0 else 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        # Accuracy ratio (positive means model1 is better)
        dm_ratio = np.mean(loss2) - np.mean(loss1)
        
        return dm_stat, p_value, dm_ratio
    
    def _newey_west_variance(self, errors: np.ndarray, max_lag: int = 4) -> float:
        """Calculate Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) variance"""
        n = len(errors)
        gamma0 = np.mean(errors ** 2)
        
        gamma = np.zeros(max_lag + 1)
        gamma[0] = gamma0
        
        for k in range(1, max_lag + 1):
            gamma[k] = np.mean(errors[k:] * errors[:-k])
        
        # Newey-West weights
        var = gamma[0]
        for k in range(1, max_lag + 1):
            weight = 1 - k / (max_lag + 1)
            var += 2 * weight * gamma[k]
        
        return var
    
    def run_pairwise_tests(self, loss_type: str = 'squared') -> Dict:
        """Run DM tests for all model pairs"""
        print(f"\n🔬 Running Diebold-Mariano Tests (Loss Type: {loss_type})")
        print("=" * 80)
        
        models = ['baseline', 'mslstm', 'mslstma', 'ensemble']
        results = {}
        
        # Create comparison matrix
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                pair_key = f"{model1}_vs_{model2}"
                
                errors1 = np.array(self.model_errors[model1])
                errors2 = np.array(self.model_errors[model2])
                
                if len(errors1) > 0 and len(errors2) > 0:
                    dm_stat, p_value, dm_ratio = self.diebold_mariano_statistic(
                        errors1, errors2, loss_type
                    )
                    
                    results[pair_key] = {
                        'model1': model1,
                        'model2': model2,
                        'dm_statistic': dm_stat,
                        'p_value': p_value,
                        'significant_5pct': p_value < 0.05,
                        'significant_10pct': p_value < 0.10,
                        'dm_ratio': dm_ratio,
                        'better_model': model1 if dm_ratio > 0 else model2,
                        'better_model_edge': abs(dm_ratio),
                        'n_observations': len(errors1)
                    }
        
        self.dm_results = results
        return results
    
    def print_results(self):
        """Print test results in a readable format"""
        print("\n📊 DIEBOLD-MARIANO TEST RESULTS")
        print("=" * 100)
        print(f"{'Model1':<12} {'Model2':<12} {'DM Stat':>10} {'P-Value':>12} {'Sig (5%)':>10} {'Better Model':>15} {'Edge':>10}")
        print("-" * 100)
        
        for pair_key, result in sorted(self.dm_results.items()):
            sig_marker = "***" if result['significant_5pct'] else ("**" if result['significant_10pct'] else "")
            print(f"{result['model1']:<12} {result['model2']:<12} {result['dm_statistic']:>10.4f} "
                  f"{result['p_value']:>12.4f} {sig_marker:>10} "
                  f"{result['better_model']:>15} {result['better_model_edge']:>10.4f}")
        
        print("\nNote: *** p<0.05, ** p<0.10")
        print("Positive DM Stat means model1 is better; Negative means model2 is better")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of results"""
        data = []
        for pair_key, result in self.dm_results.items():
            data.append({
                'Comparison': pair_key,
                'Model 1': result['model1'],
                'Model 2': result['model2'],
                'DM Statistic': result['dm_statistic'],
                'P-Value': result['p_value'],
                'Significant (5%)': result['significant_5pct'],
                'Better Model': result['better_model'],
                'Edge': result['better_model_edge'],
                'Observations': result['n_observations']
            })
        
        return pd.DataFrame(data)
    
    def visualize_results(self, save_path: str = 'dm_test_results.png'):
        """Create visualizations of DM test results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Diebold-Mariano Test Results - Model Forecast Comparison', fontsize=16, fontweight='bold')
        
        df = self.create_summary_table()
        
        # 1. DM Statistics
        ax1 = axes[0, 0]
        colors = ['green' if x < 0 else 'red' for x in df['DM Statistic']]
        ax1.barh(df['Comparison'], df['DM Statistic'], color=colors, alpha=0.7)
        ax1.axvline(x=1.96, color='blue', linestyle='--', label='5% threshold', linewidth=2)
        ax1.axvline(x=-1.96, color='blue', linestyle='--', linewidth=2)
        ax1.set_xlabel('DM Statistic')
        ax1.set_title('DM Test Statistics\n(Negative: Model2 better, Positive: Model1 better)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. P-Values
        ax2 = axes[0, 1]
        sig_colors = ['red' if x < 0.05 else 'orange' if x < 0.10 else 'green' for x in df['P-Value']]
        ax2.barh(df['Comparison'], df['P-Value'], color=sig_colors, alpha=0.7)
        ax2.axvline(x=0.05, color='red', linestyle='--', label='5% level', linewidth=2)
        ax2.axvline(x=0.10, color='orange', linestyle='--', label='10% level', linewidth=2)
        ax2.set_xlabel('P-Value')
        ax2.set_title('Statistical Significance\n(Red: Significant at 5%, Orange: at 10%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model Advantage
        ax3 = axes[1, 0]
        advantage_data = []
        for _, row in df.iterrows():
            advantage_data.append({'Model': row['Model 1'], 'Edge': row['Edge'] if row['DM Statistic'] > 0 else -row['Edge']})
            advantage_data.append({'Model': row['Model 2'], 'Edge': -row['Edge'] if row['DM Statistic'] > 0 else row['Edge']})
        
        advantage_df = pd.DataFrame(advantage_data).groupby('Model')['Edge'].mean()
        colors_adv = ['green' if x > 0 else 'red' for x in advantage_df]
        ax3.barh(advantage_df.index, advantage_df.values, color=colors_adv, alpha=0.7)
        ax3.set_xlabel('Average Edge (Positive: Better)')
        ax3.set_title('Model Relative Advantage')
        ax3.grid(True, alpha=0.3)
        
        # 4. Mean Errors by Model
        ax4 = axes[1, 1]
        models = list(self.model_errors.keys())
        mean_errors = [np.mean(np.abs(self.model_errors[m])) for m in models]
        colors_err = ['green' if x == min(mean_errors) else 'lightblue' for x in mean_errors]
        ax4.bar(models, mean_errors, color=colors_err, alpha=0.7)
        ax4.set_ylabel('Mean Absolute Error (%)')
        ax4.set_title('Mean Absolute Forecast Error by Model')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {save_path}")
        plt.close()
    
    def generate_report(self, output_file: str = 'dm_test_report.txt'):
        """Generate a comprehensive report"""
        with open(output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("DIEBOLD-MARIANO TEST REPORT - FORECAST ACCURACY COMPARISON\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("TEST OVERVIEW:\n")
            f.write("-" * 100 + "\n")
            f.write("The Diebold-Mariano (DM) test compares forecast accuracy between two models.\n")
            f.write("Null Hypothesis (H0): Both models have equal forecast accuracy\n")
            f.write("Alternative Hypothesis (H1): One model is significantly more accurate\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 100 + "\n")
            df = self.create_summary_table()
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 100 + "\n")
            for pair_key, result in sorted(self.dm_results.items()):
                f.write(f"\n{pair_key}:\n")
                f.write(f"  DM Statistic: {result['dm_statistic']:.4f}\n")
                f.write(f"  P-Value: {result['p_value']:.4f}\n")
                
                if result['significant_5pct']:
                    f.write(f"  ✅ SIGNIFICANT at 5% level (p < 0.05)\n")
                    f.write(f"  ➡️  {result['better_model'].upper()} is significantly better with edge {result['better_model_edge']:.4f}\n")
                elif result['significant_10pct']:
                    f.write(f"  ⚠️  SIGNIFICANT at 10% level (p < 0.10)\n")
                    f.write(f"  ➡️  {result['better_model'].upper()} is marginally better with edge {result['better_model_edge']:.4f}\n")
                else:
                    f.write(f"  ❌ NOT SIGNIFICANT (p >= 0.10)\n")
                    f.write(f"  ➡️  No significant difference between {result['model1'].upper()} and {result['model2'].upper()}\n")
            
            f.write("\n\nMODEL ERROR STATISTICS:\n")
            f.write("-" * 100 + "\n")
            for model in sorted(self.model_errors.keys()):
                errors = np.array(self.model_errors[model])
                f.write(f"\n{model.upper()}:\n")
                f.write(f"  Mean Absolute Error: {np.mean(np.abs(errors)):.4f}%\n")
                f.write(f"  Median Absolute Error: {np.median(np.abs(errors)):.4f}%\n")
                f.write(f"  Std Dev: {np.std(np.abs(errors)):.4f}%\n")
                f.write(f"  Min Error: {np.min(np.abs(errors)):.4f}%\n")
                f.write(f"  Max Error: {np.max(np.abs(errors)):.4f}%\n")
                f.write(f"  Total Observations: {len(errors)}\n")
        
        print(f"✅ Report saved: {output_file}")


def main():
    """Run the Diebold-Mariano Test"""
    print("\n" + "=" * 80)
    print("DIEBOLD-MARIANO TEST FOR FORECAST COMPARISON")
    print("=" * 80)
    
    # Initialize and run test
    dm_test = DieboldMarianoTest()
    dm_test.load_evaluations()
    dm_test.run_pairwise_tests(loss_type='squared')
    dm_test.print_results()
    dm_test.visualize_results()
    dm_test.generate_report()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RECOMMENDATIONS")
    print("=" * 80)
    
    df = dm_test.create_summary_table()
    significant_results = df[df['Significant (5%)'] == True]
    
    if len(significant_results) > 0:
        print(f"\n✅ Found {len(significant_results)} SIGNIFICANT differences at 5% level:")
        for _, row in significant_results.iterrows():
            print(f"   • {row['Better Model'].upper()} significantly outperforms {row['Model 1'] if row['Model 1'] != row['Better Model'] else row['Model 2']}")
    else:
        print("\n⚠️  No SIGNIFICANT differences found at 5% level")
        marginal = df[df['Significant (10%)'] == True]
        if len(marginal) > 0:
            print(f"   However, {len(marginal)} differences are marginally significant at 10% level")
    
    print("\n✅ Test Complete!")


if __name__ == "__main__":
    main()
