import pandas as pd
import numpy as np
from exchange_analysis import ExchangeAnalysis
from statistical_models import StatisticalModels

def run_analysis(data_file: str = "original_spreadsheet.csv"):
    analyzer = ExchangeAnalysis(data_file)
    processed_df = analyzer.run_analysis()
    
    stats = StatisticalModels()
    results = stats.run_models(processed_df)
    
    print("Exchange Regulation Analysis")
    print("=" * 50)
    
    formula_components = ['Listed', 'License_Count', 'Incident_Count', 
                         'Compliance_Maturity', 'Country_Reg', 'BVI']
    
    available_components = [c for c in formula_components if c in processed_df.columns]
    
    print(f"Total exchanges: {len(processed_df)}")
    print(f"Exchange_Reg range: {processed_df['Exchange_Reg'].min():.1f} - {processed_df['Exchange_Reg'].max():.1f}")
    print(f"Exchange_Reg mean: {processed_df['Exchange_Reg'].mean():.2f}")
    
    print("\nFormula components:")
    for comp in available_components:
        non_zero_rate = (processed_df[comp] != 0).mean()
        print(f"  {comp}: {processed_df[comp].mean():.2f} (non-zero: {non_zero_rate:.1%})")
    
    if 'error' not in results:
        print(f"\nPredictive features: {results['feature_names']}")
        
        linear = results['linear_regression']
        print(f"Linear R²: {linear['r2_test']:.3f} (CV: {linear['cv_r2_mean']:.3f} ± {linear['cv_r2_std']:.3f})")
        
        rf = results['random_forest']
        print(f"Random Forest R²: {rf['r2_test']:.3f} (CV: {rf['cv_r2_mean']:.3f} ± {rf['cv_r2_std']:.3f})")
        
        print("\nCorrelations:")
        for feat, corr in results['correlations'].items():
            print(f"  {feat}: {corr:.3f}")
    
    top_10 = processed_df.nlargest(10, 'Exchange_Reg')[
        ['Crypto Exchange', 'Exchange_Reg'] + available_components
    ]
    
    print("\nTop 10 exchanges:")
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['Crypto Exchange']:<25} {row['Exchange_Reg']:5.1f}")
    
    output_df = processed_df[[
        'Crypto Exchange', 'Listed', 'License_Count', 'Incident_Count',
        'Compliance_Maturity', 'Country_Reg', 'BVI', 'Exchange_Reg'
    ]].copy()
    
    output_df['Rank'] = output_df['Exchange_Reg'].rank(ascending=False, method='min').astype(int)
    output_df = output_df.sort_values('Exchange_Reg', ascending=False)
    
    output_df.to_csv('exchange_regulation_results.csv', index=False)
    print(f"\nResults saved to: exchange_regulation_results.csv")
    
    return processed_df, results

if __name__ == "__main__":
    processed_df, results = run_analysis()