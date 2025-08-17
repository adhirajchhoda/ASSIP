"""
Statistical analysis of the Exchange_Reg formula using only the original 6 variables.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from typing import Dict, Tuple, Any

class StatisticalAnalysis:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_regression_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for regression using ONLY the 6 formula variables"""
        
        # Use only the exact variables from the formula
        feature_cols = ['Listed', 'License_Count', 'Incident_Count', 
                       'Compliance_Maturity', 'Country_Reg', 'BVI']
        
        # Check which columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) != 6:
            print(f"Warning: Only {len(available_cols)} of 6 formula variables available: {available_cols}")
        
        X = df[available_cols].fillna(0)
        y = df['Exchange_Reg'].fillna(df['Exchange_Reg'].mean())
        
        return X, y
    
    def fit_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit basic linear regression"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # OLS regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Statsmodels for detailed statistics
        X_sm = sm.add_constant(X)
        ols_model = sm.OLS(y, X_sm).fit()
        
        results = {
            'model': model,
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_,
            'ols_summary': ols_model,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        return results
    
    def fit_regularized_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit Ridge and Lasso regression to check for overfitting"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features for regularization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            
            y_pred_test = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'r2_test': r2_score(y_test, y_pred_test),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'coefficients': dict(zip(X.columns, model.coef_))
            }
        
        return results
    
    def test_formula_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test if the Exchange_Reg formula is being calculated correctly"""
        
        # Recalculate Exchange_Reg manually
        manual_calc = (
            2.0 * df['Listed'] +  # Listed weight = 2.0
            df['License_Count'] +
            df['Incident_Count'] +
            df['Compliance_Maturity'] +
            df['Country_Reg'] -
            df['BVI']
        )
        
        # Compare with existing Exchange_Reg
        correlation = df['Exchange_Reg'].corr(manual_calc)
        mean_diff = (df['Exchange_Reg'] - manual_calc).abs().mean()
        max_diff = (df['Exchange_Reg'] - manual_calc).abs().max()
        
        return {
            'correlation_with_manual': correlation,
            'mean_absolute_difference': mean_diff,
            'max_absolute_difference': max_diff,
            'is_formula_correct': correlation > 0.999 and mean_diff < 0.01
        }
    
    def analyze_components(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze each component of the formula"""
        
        components = ['Listed', 'License_Count', 'Incident_Count', 
                     'Compliance_Maturity', 'Country_Reg', 'BVI']
        
        component_stats = {}
        
        for comp in components:
            if comp in df.columns:
                component_stats[comp] = {
                    'mean': df[comp].mean(),
                    'std': df[comp].std(),
                    'min': df[comp].min(),
                    'max': df[comp].max(),
                    'non_zero_rate': (df[comp] != 0).mean(),
                    'correlation_with_target': df[comp].corr(df['Exchange_Reg'])
                }
        
        return component_stats
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete statistical analysis"""
        
        print("Running statistical analysis on original formula components...")
        
        # Check formula validity
        formula_check = self.test_formula_validity(df)
        print(f"Formula validity check - Correlation: {formula_check['correlation_with_manual']:.4f}")
        
        # Analyze components
        component_analysis = self.analyze_components(df)
        
        # Prepare regression data
        X, y = self.prepare_regression_data(df)
        print(f"Regression data shape: X={X.shape}, y={y.shape}")
        
        # Fit models
        linear_results = self.fit_linear_regression(X, y)
        regularized_results = self.fit_regularized_models(X, y)
        
        # Compile results
        all_results = {
            'formula_validity': formula_check,
            'component_analysis': component_analysis,
            'linear_regression': linear_results,
            'regularized_models': regularized_results,
            'data_info': {
                'n_observations': len(df),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'target_stats': {
                    'mean': y.mean(),
                    'std': y.std(),
                    'range': y.max() - y.min()
                }
            }
        }
        
        self.results = all_results
        return all_results
    
    def print_summary(self):
        """Print summary of results"""
        if not self.results:
            print("No results to summarize. Run analysis first.")
            return
        
        results = self.results
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*60)
        
        # Data info
        data_info = results['data_info']
        print(f"Observations: {data_info['n_observations']}")
        print(f"Features: {data_info['n_features']} {data_info['feature_names']}")
        print(f"Target (Exchange_Reg): Mean={data_info['target_stats']['mean']:.2f}, "
              f"Std={data_info['target_stats']['std']:.2f}")
        
        # Formula validity
        formula = results['formula_validity']
        print(f"\nFormula Validity: {'CORRECT' if formula['is_formula_correct'] else 'INCORRECT'}")
        print(f"Correlation with manual calculation: {formula['correlation_with_manual']:.4f}")
        
        # Model performance
        linear = results['linear_regression']
        print(f"\nLinear Regression Results:")
        print(f"  R² (train): {linear['r2_train']:.4f}")
        print(f"  R² (test): {linear['r2_test']:.4f}")
        print(f"  R² (CV): {linear['cv_r2_mean']:.4f} ± {linear['cv_r2_std']:.4f}")
        print(f"  RMSE (test): {linear['rmse_test']:.4f}")
        
        # Coefficients
        print(f"\nCoefficients:")
        for feature, coef in linear['coefficients'].items():
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {linear['intercept']:.4f}")
        
        # Component correlations
        print(f"\nComponent Correlations with Exchange_Reg:")
        comp_analysis = results['component_analysis']
        for comp, stats in comp_analysis.items():
            print(f"  {comp}: {stats['correlation_with_target']:.4f}")
        
        print("="*60)