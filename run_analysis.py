"""
Main script to run cryptocurrency exchange regulation analysis.
Implements exact formula: Exchange_Reg = (Listed) + License_Count + Incident Count + Compliance_Maturity + Country_Reg - BVI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from analysis_core import ExchangeAnalysis
from statistical_analysis import StatisticalAnalysis

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def run_analysis(data_file: str = "original_spreadsheet.csv"):
    """Run complete analysis pipeline"""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting cryptocurrency exchange regulation analysis")
    
    # Initialize analysis
    analyzer = ExchangeAnalysis(data_file)
    
    # Load and process data
    logger.info("Loading and processing data...")
    analyzer.load_data()
    processed_df = analyzer.run_analysis()
    
    # Get summary
    summary = analyzer.get_results_summary()
    
    # Statistical analysis
    logger.info("Running statistical analysis...")
    stats = StatisticalAnalysis()
    stats_results = stats.run_complete_analysis(processed_df)
    
    # Print results
    print_results(summary, stats_results)
    
    # Save outputs
    save_results(processed_df, summary, stats_results)
    
    logger.info("Analysis completed")
    
    return processed_df, summary, stats_results

def print_results(summary, stats_results):
    """Print analysis results"""
    
    print("\n" + "="*80)
    print("CRYPTOCURRENCY EXCHANGE REGULATION ANALYSIS RESULTS")
    print("="*80)
    
    # Basic statistics
    print(f"Total exchanges analyzed: {summary['total_exchanges']}")
    
    reg_stats = summary['exchange_reg_stats']
    print(f"Exchange_Reg score - Mean: {reg_stats['mean']:.2f}, "
          f"Std: {reg_stats['std']:.2f}, Range: {reg_stats['min']:.1f}-{reg_stats['max']:.1f}")
    
    # Component statistics
    comp_stats = summary['component_stats']
    print(f"\nComponent Statistics:")
    print(f"  Listed rate: {comp_stats['listed_rate']:.1%}")
    print(f"  Average license count: {comp_stats['avg_license_count']:.1f}")
    print(f"  Average incident count: {comp_stats['avg_incident_count']:.1f}")
    print(f"  Average compliance maturity: {comp_stats['avg_compliance_maturity']:.1f}")
    print(f"  Average country regulation: {comp_stats['avg_country_reg']:.1f}")
    print(f"  BVI rate: {comp_stats['bvi_rate']:.1%}")
    
    # Top exchanges
    print(f"\nTop 10 Exchanges by Regulation Score:")
    print("-" * 80)
    for i, exchange in enumerate(summary['top_10_exchanges'], 1):
        print(f"{i:2d}. {exchange['Crypto Exchange']:<25} "
              f"Score: {exchange['Exchange_Reg']:5.1f} "
              f"(L:{exchange['Listed']}, LC:{exchange['License_Count']}, "
              f"I:{exchange['Incident_Count']}, C:{exchange['Compliance_Maturity']:.1f}, "
              f"CR:{exchange['Country_Reg']:.1f}, B:{exchange['BVI']})")
    
    # Statistical results summary
    stats = StatisticalAnalysis()
    stats.results = stats_results
    stats.print_summary()

def save_results(processed_df, summary, stats_results):
    """Save analysis results"""
    
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save processed data
    key_columns = [
        'Crypto Exchange', 'Listed', 'License_Count', 'Incident_Count',
        'Compliance_Maturity', 'Country_Reg', 'Primary_Country', 'BVI', 'Exchange_Reg'
    ]
    
    available_columns = [col for col in key_columns if col in processed_df.columns]
    output_df = processed_df[available_columns].copy()
    
    # Add ranking
    output_df['Rank'] = output_df['Exchange_Reg'].rank(ascending=False, method='min').astype(int)
    output_df = output_df.sort_values('Exchange_Reg', ascending=False)
    
    # Save main results
    output_df.to_csv(output_dir / "exchange_regulation_results.csv", index=False)
    
    # Save model results
    linear_results = stats_results['linear_regression']
    
    model_summary = pd.DataFrame({
        'Component': list(linear_results['coefficients'].keys()) + ['Intercept'],
        'Coefficient': list(linear_results['coefficients'].values()) + [linear_results['intercept']],
        'Correlation': [stats_results['component_analysis'][comp]['correlation_with_target'] 
                       if comp in stats_results['component_analysis'] else np.nan 
                       for comp in linear_results['coefficients'].keys()] + [np.nan]
    })
    
    model_summary.to_csv(output_dir / "model_coefficients.csv", index=False)
    
    # Save performance metrics
    performance = pd.DataFrame({
        'Metric': ['R² (train)', 'R² (test)', 'R² (CV mean)', 'R² (CV std)', 'RMSE (test)', 'MAE (test)'],
        'Value': [
            linear_results['r2_train'],
            linear_results['r2_test'], 
            linear_results['cv_r2_mean'],
            linear_results['cv_r2_std'],
            linear_results['rmse_test'],
            linear_results['mae_test']
        ]
    })
    
    performance.to_csv(output_dir / "model_performance.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - exchange_regulation_results.csv (main results)")
    print(f"  - model_coefficients.csv (regression coefficients)")
    print(f"  - model_performance.csv (model performance metrics)")

if __name__ == "__main__":
    processed_df, summary, stats_results = run_analysis()