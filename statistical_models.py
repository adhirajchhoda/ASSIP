"""
Statistical models and analysis for cryptocurrency exchange regulation data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ExchangeRegulationModels:
    """
    Advanced statistical models for exchange regulation analysis.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Exchange_Reg',
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling by selecting relevant columns."""
        
        if exclude_cols is None:
            exclude_cols = [
                'Crypto Exchange', 'Primary_Country', 'Combined_Regulatory_Text',
                'Product_Complexity_Breakdown', 'Regulatory_Framework_Breakdown'
            ]
        
        feature_df = df.copy()
        
        feature_cols = []
        for col in feature_df.columns:
            if (col != target_col and 
                col not in exclude_cols and
                feature_df[col].dtype in ['int64', 'float64'] and
                feature_df[col].var() > 0):
                feature_cols.append(col)
        
        X = feature_df[feature_cols].fillna(0)
        y = feature_df[target_col].fillna(feature_df[target_col].mean())
        
        return X, y
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        X_enhanced = X.copy()
        
        interaction_pairs = [
            ('Product_Complexity', 'Regulatory_Framework_Score'),
            ('Listed', 'Country_Reg'),
            ('Incident_Count', 'License_Count'),
            ('Num_Products', 'Compliance_Maturity'),
            ('Product_Complexity', 'Incident_Severity'),
            ('Listed', 'Product_Complexity')
        ]
        
        for col1, col2 in interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                X_enhanced[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        geo_cols = [col for col in X.columns if col.startswith('Geo_')]
        product_cols = [col for col in X.columns if col.startswith('Product_') and col.endswith('_final')]
        reg_cols = [col for col in X.columns if col.startswith('Reg_')]
        
        if geo_cols:
            X_enhanced['Geographic_Diversity'] = X[geo_cols].sum(axis=1)
        if product_cols:
            X_enhanced['Product_Diversity'] = X[product_cols].sum(axis=1)
        if reg_cols:
            X_enhanced['Regulatory_Diversity'] = X[reg_cols].sum(axis=1)
        
        return X_enhanced
    
    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for multicollinearity detection."""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values('VIF', ascending=False)
    
    def fit_linear_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit various linear regression models with cross-validation."""
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )
        
        models_config = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet()
        }
        
        param_grids = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        }
        
        results = {}
        
        for name, model in models_config.items():
            if name in param_grids:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=5, scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = model.fit(X_train, y_train)
                best_params = {}
            
            y_pred = best_model.predict(X_test)
            
            cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
            
            results[name] = {
                'model': best_model,
                'best_params': best_params,
                'r2_test': r2_score(y_test, y_pred),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae_test': mean_absolute_error(y_test, y_pred),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            if hasattr(best_model, 'coef_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': best_model.coef_,
                    'abs_coefficient': np.abs(best_model.coef_)
                }).sort_values('abs_coefficient', ascending=False)
        
        return results
    
    def fit_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit ensemble models (Random Forest, Gradient Boosting, XGBoost)."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        models_config = {
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state)
        }
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        }
        
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBRegressor(random_state=self.random_state)
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.01]
            }
        
        results = {}
        
        for name, model in models_config.items():
            grid_search = GridSearchCV(
                model, param_grids[name], cv=3, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(X_test)
            
            cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
            
            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'r2_test': r2_score(y_test, y_pred),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae_test': mean_absolute_error(y_test, y_pred),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        return results
    
    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = 4) -> Dict[str, Any]:
        """Perform clustering analysis to identify exchange archetypes."""
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        
        kmeans_labels = kmeans.fit_predict(X_scaled)
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        results = {
            'kmeans_labels': kmeans_labels,
            'hierarchical_labels': hierarchical_labels,
            'pca_components': X_pca,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'kmeans_centers': kmeans.cluster_centers_,
            'feature_names': X.columns.tolist()
        }
        
        return results
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Perform feature selection using multiple methods."""
        
        selector_univariate = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector_univariate.fit_transform(X, y)
        selected_features = X.columns[selector_univariate.get_support()].tolist()
        
        lr = LinearRegression()
        selector_rfe = RFE(lr, n_features_to_select=min(k, X.shape[1]))
        selector_rfe.fit(X, y)
        rfe_features = X.columns[selector_rfe.get_support()].tolist()
        
        combined_features = list(set(selected_features + rfe_features))
        
        return X[combined_features], combined_features
    
    def correlation_analysis(self, df: pd.DataFrame, target_col: str = 'Exchange_Reg') -> pd.DataFrame:
        """Perform correlation analysis with the target variable."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for col in numeric_cols:
            if col != target_col and df[col].var() > 0:
                pearson_corr, pearson_p = pearsonr(df[col].fillna(0), df[target_col].fillna(df[target_col].mean()))
                spearman_corr, spearman_p = spearmanr(df[col].fillna(0), df[target_col].fillna(df[target_col].mean()))
                
                correlations.append({
                    'feature': col,
                    'pearson_corr': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'abs_pearson': abs(pearson_corr)
                })
        
        return pd.DataFrame(correlations).sort_values('abs_pearson', ascending=False)
    
    def run_comprehensive_analysis(self, df: pd.DataFrame, target_col: str = 'Exchange_Reg') -> Dict[str, Any]:
        """Run comprehensive analysis including all models and diagnostics."""
        
        X, y = self.prepare_features(df, target_col)
        
        X_enhanced = self.create_interaction_features(X)
        
        X_selected, selected_features = self.feature_selection(X_enhanced, y, k=25)
        
        linear_results = self.fit_linear_models(X_selected, y)
        ensemble_results = self.fit_ensemble_models(X_selected, y)
        clustering_results = self.perform_clustering(X_selected)
        correlation_results = self.correlation_analysis(df, target_col)
        
        try:
            vif_results = self.calculate_vif(X_selected)
        except Exception:
            vif_results = pd.DataFrame({'Feature': ['Error'], 'VIF': ['Could not calculate']})
        
        all_results = {
            'linear_models': linear_results,
            'ensemble_models': ensemble_results,
            'clustering': clustering_results,
            'correlations': correlation_results,
            'vif': vif_results,
            'selected_features': selected_features,
            'feature_importance': self.feature_importance
        }
        
        self.results = all_results
        return all_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison table of all models."""
        if not self.results:
            raise ValueError("No results available. Run comprehensive analysis first.")
        
        comparison_data = []
        
        for model_type in ['linear_models', 'ensemble_models']:
            if model_type in self.results:
                for name, result in self.results[model_type].items():
                    comparison_data.append({
                        'Model': name,
                        'Type': model_type.replace('_', ' ').title(),
                        'R2_Test': round(result['r2_test'], 4),
                        'RMSE_Test': round(result['rmse_test'], 4),
                        'MAE_Test': round(result['mae_test'], 4),
                        'CV_R2_Mean': round(result['cv_r2_mean'], 4),
                        'CV_R2_Std': round(result['cv_r2_std'], 4)
                    })
        
        return pd.DataFrame(comparison_data).sort_values('R2_Test', ascending=False)
    
    def get_top_features(self, n: int = 10) -> Dict[str, pd.DataFrame]:
        """Get top features from each model."""
        top_features = {}
        
        for model_name, importance_df in self.feature_importance.items():
            if 'coefficient' in importance_df.columns:
                top_features[model_name] = importance_df.head(n)[['feature', 'coefficient', 'abs_coefficient']]
            elif 'importance' in importance_df.columns:
                top_features[model_name] = importance_df.head(n)[['feature', 'importance']]
        
        return top_features