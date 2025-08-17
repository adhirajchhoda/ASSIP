import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.formula.api as smf
from typing import Dict, Any

class StatisticalModels:
    
    def __init__(self):
        self.results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        product_text_cols = [
            'Products Offered (spot, futures, options, etc.)',
            'Product Launch Dates'
        ]
        
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
    
    def run_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        model_df = self.prepare_features(df)
        model_df = model_df.dropna(subset=['Exchange_Reg'])
        
        feature_cols = ['Product_Complexity', 'Num_Products', 'Incident_Severity']
        available_features = [c for c in feature_cols if c in model_df.columns and model_df[c].var() > 0]
        
        if not available_features:
            return {"error": "No features with variance available"}
        
        X = model_df[available_features].fillna(0)
        y = model_df['Exchange_Reg']
        
        correlations = {}
        for feat in available_features:
            correlations[feat] = y.corr(X[feat])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred = linear_model.predict(X_test)
        
        cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring='r2')
        
        try:
            formula = 'Exchange_Reg ~ ' + ' + '.join([f'Q("{c}")' for c in available_features])
            ols_model = smf.ols(formula=formula, data=model_df).fit()
            ols_summary = str(ols_model.summary())
        except Exception as e:
            ols_summary = f"OLS failed: {e}"
        
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
        
        self.results = results
        return results