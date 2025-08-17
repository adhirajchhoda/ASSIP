"""
Corrected analysis following original methodology.
Tests additional features (Product_Complexity, etc.) for predicting Exchange_Reg score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Tuple, Any

from analysis_core import ExchangeAnalysis

class CorrectedStatisticalAnalysis:
    """Statistical analysis following original methodology."""
    
    def __init__(self):
        self.results = {}
        
    def prepare_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare additional features beyond the formula components."""
        result = df.copy()
        
        # Product complexity from text parsing
        product_text_cols = [
            'Products Offered (spot, futures, options, etc.)',
            'Product Launch Dates'
        ]
        
        # Simple product complexity scoring
        def calculate_product_complexity(row):
            text = ""
            for col in product_text_cols:
                if col in row and isinstance(row[col], str):
                    text += " " + row[col].lower()
            
            score = 0
            if 'option' in text:
                score += 3
            if 'future' in text or 'perpetual' in text:
                score += 3
            if 'margin' in text or 'leverage' in text:
                score += 2
            if 'staking' in text:
                score += 1
            if 'spot' in text:
                score += 0.5
                
            return score
        
        def count_products(row):
            text = ""
            for col in product_text_cols:
                if col in row and isinstance(row[col], str):
                    text += " " + row[col].lower()
            
            products = ['spot', 'futures', 'options', 'margin', 'staking', 'p2p', 'nft', 'lending']
            return sum(1 for p in products if p in text)
        
        result['Product_Complexity'] = result.apply(calculate_product_complexity, axis=1)
        result['Num_Products'] = result.apply(count_products, axis=1)
        
        # Enhanced incident severity from original notebook
        def incident_severity_score(text):
            if not isinstance(text, str):
                return 0.0
            
            text_lower = text.lower()
            if 'criminal' in text_lower or 'fraud' in text_lower or 'felony' in text_lower:
                return 3.0
            elif 'class action' in text_lower or 'settlement' in text_lower:
                return 2.5
            elif 'fine' in text_lower or 'penalty' in text_lower or 'sanction' in text_lower:
                return 2.0
            elif 'warning' in text_lower or 'notice' in text_lower:
                return 1.0
            else:
                return 0.0
        
        incident_col = 'Regulatory Incidents (fines, violations, etc.)'
        if incident_col in result.columns:
            result['Incident_Severity'] = result[incident_col].apply(incident_severity_score)
        else:
            result['Incident_Severity'] = 0.0
            
        return result
    
    def test_predictive_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test additional features for predicting Exchange_Reg (original methodology)."""
        
        # Prepare additional features
        model_df = self.prepare_additional_features(df)
        model_df = model_df.dropna(subset=['Exchange_Reg'])
        
        # Additional features to test (NOT the formula components)
        feature_cols = ['Product_Complexity', 'Num_Products', 'Incident_Severity']
        available_features = [c for c in feature_cols if c in model_df.columns and model_df[c].var() > 0]
        
        if not available_features:
            return {"error": "No features with variance available"}
        
        # Prepare data
        X = model_df[available_features].fillna(0)
        y = model_df['Exchange_Reg']
        
        # Test 1: Simple correlation analysis (baseline)
        correlations = {}
        for feat in available_features:
            correlations[feat] = y.corr(X[feat])
        
        # Test 2: Linear regression with just these additional features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred = linear_model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring='r2')
        
        # Statsmodels for detailed output
        try:
            formula = 'Exchange_Reg ~ ' + ' + '.join([f'Q("{c}")' for c in available_features])
            ols_model = smf.ols(formula=formula, data=model_df).fit()
            ols_summary = str(ols_model.summary())
        except Exception as e:
            ols_summary = f"OLS failed: {e}"
        
        # Random Forest for comparison
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
        
        results = {
            'data_shape': X.shape,
            'target_stats': {
                'mean': y.mean(),
                'std': y.std(),
                'range': [y.min(), y.max()]
            },
            'correlations': correlations,
            'linear_regression': {
                'r2_test': r2_score(y_test, y_pred),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae_test': mean_absolute_error(y_test, y_pred),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'coefficients': dict(zip(available_features, linear_model.coef_)),
                'intercept': linear_model.intercept_
            },
            'random_forest': {
                'r2_test': r2_score(y_test, rf_pred),
                'rmse_test': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'cv_r2_mean': rf_cv_scores.mean(),
                'cv_r2_std': rf_cv_scores.std(),
                'feature_importance': dict(zip(available_features, rf_model.feature_importances_))
            },
            'ols_summary': ols_summary,
            'feature_names': available_features
        }
        
        return results
    
    def analyze_formula_components(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution and characteristics of formula components."""
        
        formula_components = ['Listed', 'License_Count', 'Incident_Count', 
                            'Compliance_Maturity', 'Country_Reg', 'BVI']
        
        available_components = [c for c in formula_components if c in df.columns]
        
        component_stats = {}
        for comp in available_components:
            series = df[comp]
            component_stats[comp] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'non_zero_rate': (series != 0).mean(),
                'unique_values': sorted(series.unique())[:10]  # First 10 unique values
            }
        
        # Verify formula calculation
        if all(c in df.columns for c in formula_components):
            manual_exchange_reg = (
                2.0 * df['Listed'] +
                df['License_Count'] +
                df['Incident_Count'] +
                df['Compliance_Maturity'] +
                df['Country_Reg'] -
                df['BVI']
            )
            
            correlation_check = df['Exchange_Reg'].corr(manual_exchange_reg)
            mean_diff = (df['Exchange_Reg'] - manual_exchange_reg).abs().mean()
        else:
            correlation_check = None
            mean_diff = None
        
        return {
            'component_statistics': component_stats,
            'formula_verification': {
                'correlation_with_manual': correlation_check,
                'mean_absolute_difference': mean_diff,
                'formula_correct': correlation_check and correlation_check > 0.999
            },
            'exchange_reg_distribution': {
                'mean': df['Exchange_Reg'].mean(),
                'std': df['Exchange_Reg'].std(),
                'range': [df['Exchange_Reg'].min(), df['Exchange_Reg'].max()],
                'quartiles': df['Exchange_Reg'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        }
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete corrected analysis."""
        
        print("Running corrected analysis following original methodology...")
        
        # Analyze formula components
        component_analysis = self.analyze_formula_components(df)
        
        # Test predictive models with additional features
        predictive_analysis = self.test_predictive_models(df)
        
        # Get top exchanges
        top_10 = df.nlargest(10, 'Exchange_Reg')[
            ['Crypto Exchange', 'Exchange_Reg'] + 
            [c for c in ['Listed', 'License_Count', 'Incident_Count', 
                        'Compliance_Maturity', 'Country_Reg', 'BVI'] if c in df.columns]
        ]
        
        all_results = {
            'component_analysis': component_analysis,
            'predictive_analysis': predictive_analysis,
            'top_exchanges': top_10.to_dict('records'),
            'methodology_note': (
                "This analysis tests whether additional features (Product_Complexity, "
                "Num_Products, Incident_Severity) can predict the Exchange_Reg score "
                "calculated from the original formula."
            )
        }
        
        self.results = all_results
        return all_results
    
    def print_results_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        results = self.results
        
        print("\n" + "="*80)
        print("CRYPTOCURRENCY EXCHANGE REGULATION ANALYSIS")
        print("="*80)
        
        # Formula verification
        formula_check = results['component_analysis']['formula_verification']
        print(f"Formula verification: {'CORRECT' if formula_check['formula_correct'] else 'INCORRECT'}")
        
        # Exchange_Reg distribution
        reg_dist = results['component_analysis']['exchange_reg_distribution']
        print(f"Exchange_Reg scores: Mean={reg_dist['mean']:.2f}, Std={reg_dist['std']:.2f}")
        print(f"Range: {reg_dist['range'][0]:.1f} to {reg_dist['range'][1]:.1f}")
        
        # Component statistics
        print(f"\nFormula component statistics:")
        comp_stats = results['component_analysis']['component_statistics']
        for comp, stats in comp_stats.items():
            print(f"  {comp}: Mean={stats['mean']:.2f}, Non-zero rate={stats['non_zero_rate']:.1%}")
        
        # Predictive analysis
        if 'error' not in results['predictive_analysis']:
            pred_results = results['predictive_analysis']
            
            print(f"\nPredictive analysis using additional features:")
            print(f"Features tested: {pred_results['feature_names']}")
            
            # Linear regression results
            linear = pred_results['linear_regression']
            print(f"Linear regression R²: {linear['r2_test']:.3f} (CV: {linear['cv_r2_mean']:.3f} ± {linear['cv_r2_std']:.3f})")
            
            # Random Forest results
            rf = pred_results['random_forest']
            print(f"Random Forest R²: {rf['r2_test']:.3f} (CV: {rf['cv_r2_mean']:.3f} ± {rf['cv_r2_std']:.3f})")
            
            # Correlations
            print(f"\nFeature correlations with Exchange_Reg:")
            for feat, corr in pred_results['correlations'].items():
                print(f"  {feat}: {corr:.3f}")
            
            # Feature importance (Random Forest)
            print(f"\nRandom Forest feature importance:")
            for feat, importance in rf['feature_importance'].items():
                print(f"  {feat}: {importance:.3f}")
        
        # Top exchanges
        print(f"\nTop 10 exchanges by regulation score:")
        for i, exchange in enumerate(results['top_exchanges'][:10], 1):
            print(f"{i:2d}. {exchange['Crypto Exchange']:<25} Score: {exchange['Exchange_Reg']:5.1f}")
        
        print("="*80)

def run_corrected_analysis(data_file: str = "original_spreadsheet.csv"):
    """Run the corrected analysis pipeline."""
    
    # Load and process data using the core analyzer
    analyzer = ExchangeAnalysis(data_file)
    processed_df = analyzer.run_analysis()
    
    # Run corrected statistical analysis
    stats = CorrectedStatisticalAnalysis()
    results = stats.run_complete_analysis(processed_df)
    
    # Print results
    stats.print_results_summary()
    
    # Save results
    output_df = processed_df[[
        'Crypto Exchange', 'Listed', 'License_Count', 'Incident_Count',
        'Compliance_Maturity', 'Country_Reg', 'BVI', 'Exchange_Reg'
    ]].copy()
    
    output_df['Rank'] = output_df['Exchange_Reg'].rank(ascending=False, method='min').astype(int)
    output_df = output_df.sort_values('Exchange_Reg', ascending=False)
    
    output_df.to_csv('cryptocurrency_exchange_regulation_analysis.csv', index=False)
    
    print(f"\nResults saved to: cryptocurrency_exchange_regulation_analysis.csv")
    
    return processed_df, results

if __name__ == "__main__":
    processed_df, results = run_corrected_analysis()