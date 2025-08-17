"""
Data processing module for cryptocurrency exchange regulation analysis.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, date
import logging

from config import (
    LICENSE_CANON_MAP, INCIDENT_SEVERITY_WEIGHTS, PRODUCT_WEIGHTS, PRODUCT_KEYWORDS,
    GEO_KEYWORD_MAP, REGULATION_KEYWORD_MAP, COMPLIANCE_KEYWORD_MAP, GOVERNANCE_KEYWORD_MAP,
    COLUMN_NAMES, EXCHANGE_REG_WEIGHTS, REGULATORY_WEIGHTS, CURRENT_YEAR, CURRENT_MONTH
)

class ExchangeRegulationAnalyzer:
    """
    Unified analyzer for cryptocurrency exchange regulation data.
    """
    
    def __init__(self, data_file: str = "original_spreadsheet.csv"):
        self.data_file = data_file
        self.df = None
        self.processed_df = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the cryptocurrency exchange data."""
        try:
            self.df = pd.read_csv(self.data_file)
            if COLUMN_NAMES['crypto_exchange_col'] in self.df.columns:
                self.df[COLUMN_NAMES['crypto_exchange_col']] = self.df[COLUMN_NAMES['crypto_exchange_col']].astype(str).str.strip()
            self.logger.info(f"Loaded {len(self.df)} exchanges from {self.data_file}")
            return self.df
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
            
    def create_indicator_columns(self, dataframe: pd.DataFrame, column_to_search: str, 
                               keyword_map: Dict[str, List[str]], new_prefix: str) -> pd.DataFrame:
        """Create indicator columns with negation detection."""
        if column_to_search not in dataframe.columns:
            self.logger.warning(f"Column '{column_to_search}' not found, skipping...")
            return dataframe
        
        df_result = dataframe.copy()
        
        for suffix, search_terms in keyword_map.items():
            new_column_name = f"{new_prefix}_{suffix}"
            df_result[new_column_name] = 0
            
            for idx, cell_value in df_result[column_to_search].items():
                if pd.isna(cell_value) or cell_value == '':
                    continue
                    
                cell_text = str(cell_value).lower()
                
                direct_negation_found = False
                for term in search_terms:
                    direct_negation_patterns = [
                        rf'\b(no|not|does\s+not|do\s+not|don\'t|doesn\'t|never|none|without|lacks|lacking|absent|missing|exempt|exemption|waived|waiver)\s+{re.escape(term.lower())}\b',
                        rf'\b{re.escape(term.lower())}\s+(not|never|none|exempt|exemption|waived|waiver)\b',
                        rf'\b(optional|voluntary|discretionary)\s+{re.escape(term.lower())}\b',
                        rf'\b{re.escape(term.lower())}\s+(optional|voluntary|discretionary)\b'
                    ]
                    
                    for pattern in direct_negation_patterns:
                        if re.search(pattern, cell_text):
                            df_result.loc[idx, new_column_name] = 0
                            direct_negation_found = True
                            break
                    
                    if direct_negation_found:
                        break
                
                if not direct_negation_found:
                    for term in search_terms:
                        pattern = r'\b' + re.escape(term.lower()) + r'\b'
                        if re.search(pattern, cell_text):
                            df_result.loc[idx, new_column_name] = 1
                            break
            
            matches = df_result[new_column_name].sum()
            self.logger.info(f"Created {new_column_name}: {matches} matches found")
        
        return df_result

    def calculate_bvi_listed_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BVI and Listed columns from governance structure text."""
        result = df.copy()
        
        bvi_col = 'Geo_BVI'
        if bvi_col in result.columns:
            result['BVI'] = result[bvi_col].fillna(0).astype(int)
        else:
            result['BVI'] = 0
        
        gov_col = COLUMN_NAMES['governance_col']
        if gov_col in result.columns:
            listed_series = result[gov_col].astype(str).str.lower()
            listed_flags = listed_series.str.contains(
                r'listed|public|ipo|nasdaq|nyse|lse|tsx|publicly', 
                regex=True, na=False
            ).astype(int)
            result['Listed'] = listed_flags
        else:
            result['Listed'] = 0
        
        return result

    def _normalize_license_tokens(self, text: str) -> set:
        """Normalize license tokens using canonical mapping."""
        if not isinstance(text, str) or not text.strip():
            return set()
        raw = [p.strip().lower() for p in re.split(r'[;,/\n\|]', text) if p.strip()]
        canon = set()
        for token in raw:
            for k, v in LICENSE_CANON_MAP.items():
                if k in token:
                    canon.add(v)
                    break
            else:
                canon.add(token)
        return canon

    def calculate_license_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count distinct licenses/registrations using canonical mapping."""
        result = df.copy()
        canon_sets = [set() for _ in range(len(result))]
        
        for col in COLUMN_NAMES['license_text_cols']:
            if col in result.columns:
                series_sets = result[col].astype(str).apply(self._normalize_license_tokens)
                canon_sets = [a.union(b) for a, b in zip(canon_sets, series_sets)]
        
        result['License_Count'] = pd.Series([len(s) for s in canon_sets], index=result.index).astype(int)
        return result

    def _kyc_strength(self, text: str) -> int:
        """Calculate KYC strength score."""
        if not isinstance(text, str):
            return 0
        t = text.lower()
        if 'full kyc' in t or 'mandatory kyc' in t:
            return 3
        if 'tiered kyc' in t:
            return 2
        if 'optional kyc' in t or 'partial kyc' in t:
            return 1
        if 'kyc' in t:
            return 1
        return 0

    def _por_strength(self, text: str) -> int:
        """Calculate proof of reserves strength."""
        if not isinstance(text, str):
            return 0
        t = text.lower()
        keywords = ['proof of reserves','attest','audit','audited','merkle','reserves']
        return int(any(k in t for k in keywords))

    def calculate_compliance_maturity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse compliance maturity from spreadsheet columns."""
        result = df.copy()
        
        comp_col = COLUMN_NAMES['compliance_text_cols'][0]
        
        if comp_col in result.columns:
            comp_text = result[comp_col].astype(str)
            
            result['KYC_Full'] = comp_text.apply(lambda x: int('full kyc' in x.lower() or 'mandatory kyc' in x.lower()))
            result['KYC_Tiered'] = comp_text.apply(lambda x: int('tiered kyc' in x.lower()))
            result['KYC_Optional'] = comp_text.apply(lambda x: int('optional kyc' in x.lower() or 'partial kyc' in x.lower()))
            result['Proof_of_Reserves'] = comp_text.apply(self._por_strength)
            result['KYC_Strength_Score'] = comp_text.apply(self._kyc_strength)
        else:
            result['KYC_Full'] = 0
            result['KYC_Tiered'] = 0
            result['KYC_Optional'] = 0
            result['Proof_of_Reserves'] = 0
            result['KYC_Strength_Score'] = 0
        
        result['Compliance_Maturity'] = result['KYC_Strength_Score'] + result['Proof_of_Reserves']
        return result

    def _count_incidents(self, text: str) -> int:
        """Count regulatory incidents."""
        if not isinstance(text, str) or not text.strip():
            return 0
        parts = [p.strip() for p in re.split(r'[;\n\|]', text) if p.strip()]
        return len(parts)

    def _incident_severity_score(self, text: str) -> float:
        """Calculate incident severity score."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        lower = text.lower()
        score = 0.0
        for pattern, weight in INCIDENT_SEVERITY_WEIGHTS:
            if re.search(pattern, lower):
                score = max(score, weight)
        return score

    def calculate_incident_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate incident count and severity scores."""
        result = df.copy()
        inc_counts = pd.Series(0, index=result.index)
        inc_severity = pd.Series(0.0, index=result.index)
        
        for col in COLUMN_NAMES['incident_text_cols']:
            if col in result.columns:
                vals = result[col].astype(str)
                inc_counts = np.maximum(inc_counts.values, vals.apply(self._count_incidents).values)
                inc_severity = np.maximum(inc_severity.values, vals.apply(self._incident_severity_score).values)
        
        result['Incident_Count'] = inc_counts.astype(int)
        result['Incident_Severity'] = inc_severity
        return result

    def _extract_primary_country(self, country_text: str) -> str:
        """Extract primary country from country/region text."""
        if not isinstance(country_text, str) or not country_text.strip():
            return 'Unknown'
        
        text = country_text.strip()
        
        if 'global' in text.lower() or 'worldwide' in text.lower():
            if 'hq' in text.lower() or 'headquarter' in text.lower():
                parts = re.split(r'hq|headquarter', text.lower())
                if len(parts) > 1:
                    hq_part = parts[1].strip()
                    if '(' in hq_part:
                        country = hq_part.split('(')[1].split(')')[0].strip()
                    elif ' in ' in hq_part:
                        country = hq_part.split(' in ')[1].split(',')[0].strip()
                    else:
                        country = hq_part.split(',')[0].strip()
                    return country.title()
        
        first_part = re.split(r'[,;&]', text)[0].strip()
        first_part = re.sub(r'^primarily\s+in\s+', '', first_part, flags=re.IGNORECASE)
        first_part = re.sub(r'^\s*but\s+', '', first_part, flags=re.IGNORECASE)
        
        return first_part.title()

    def calculate_country_regulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build per-country regulation baseline."""
        result = df.copy()
        
        if COLUMN_NAMES['hq_country_col'] not in result.columns:
            result['Country_Reg'] = 0.0
            return result
        
        for needed in ['License_Count','Compliance_Maturity']:
            if needed not in result.columns:
                result[needed] = 0
        
        result['Primary_Country'] = result[COLUMN_NAMES['hq_country_col']].apply(self._extract_primary_country)
        
        country_stats = (
            result.groupby('Primary_Country')[['License_Count','Compliance_Maturity']]
            .mean()
            .rename(columns={
                'License_Count': 'Country_License_Mean',
                'Compliance_Maturity': 'Country_Compliance_Mean'
            })
        )
        
        result = result.merge(
            country_stats,
            left_on='Primary_Country',
            right_index=True,
            how='left'
        )
        
        result[['Country_License_Mean','Country_Compliance_Mean']] = result[['Country_License_Mean','Country_Compliance_Mean']].fillna(0)
        
        result['Country_Reg'] = (
            0.5 * result['Country_License_Mean'] + 0.5 * result['Country_Compliance_Mean']
        )
        
        return result

    def infer_product_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer product flags using NLP on text columns."""
        result = df.copy()
        
        def row_text_concat(row: pd.Series) -> str:
            parts = []
            for c in COLUMN_NAMES['product_text_cols']:
                if c in row and isinstance(row[c], str) and row[c] != 'nan':
                    parts.append(row[c])
            return ' | '.join(parts).lower()
        
        text_series = result.apply(row_text_concat, axis=1)

        for col, patterns in PRODUCT_KEYWORDS.items():
            regex = re.compile('|'.join(patterns), flags=re.IGNORECASE)
            inferred = text_series.apply(lambda t: int(bool(regex.search(t))))
            result[f'{col}_inferred'] = inferred
            
            if col in result.columns and result[col].dropna().shape[0] > 0:
                explicit = pd.to_numeric(result[col], errors='coerce').fillna(0).astype(int)
                final = np.maximum(explicit, inferred)
            else:
                final = inferred
            result[f'{col}_final'] = final

        final_cols = [f'{c}_final' for c in PRODUCT_KEYWORDS.keys() if f'{c}_final' in result.columns]
        unknown_mask = (result[final_cols].sum(axis=1) == 0) if final_cols else pd.Series(True, index=result.index)
        result['Product_Inference_Unknown'] = unknown_mask.astype(int)
        return result

    def calculate_product_complexity_nlp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute product complexity using final flags after NLP inference."""
        result = self.infer_product_flags(df)
        final_cols = [f'{c}_final' for c in PRODUCT_KEYWORDS.keys() if f'{c}_final' in result.columns]
        
        if not final_cols:
            result['Product_Complexity'] = 0.0
            result['Num_Products'] = 0.0
            return result
        
        weights = {f'{k}_final': w for k, w in PRODUCT_WEIGHTS.items() if f'{k}_final' in result.columns}
        result['Product_Complexity'] = (result[list(weights.keys())] * pd.Series(weights)).sum(axis=1)
        result['Num_Products'] = result[final_cols].sum(axis=1)
        return result

    def calculate_exchange_regulation_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the final Exchange_Reg score using the formula."""
        step = df.copy()
        
        step = self.calculate_bvi_listed_score(step)
        step = self.calculate_license_score(step)
        step = self.calculate_compliance_maturity(step)
        step = self.calculate_incident_score(step)
        step = self.calculate_country_regulation(step)
        step = self.calculate_product_complexity_nlp(step)
        
        step['Exchange_Reg'] = (
            EXCHANGE_REG_WEIGHTS['listed_weight'] * step['Listed'].fillna(0)
            + step['License_Count'].fillna(0)
            + step['Incident_Count'].fillna(0)
            + step['Compliance_Maturity'].fillna(0)
            + step['Country_Reg'].fillna(0)
            - step['BVI'].fillna(0)
        )
        
        return step

    def process_all_features(self) -> pd.DataFrame:
        """Process all features and calculate regulation score."""
        if self.df is None:
            self.load_data()
        
        self.logger.info("Processing all features...")
        
        step = self.df.copy()
        
        if COLUMN_NAMES['hq_country_col'] in step.columns:
            step = self.create_indicator_columns(
                step, 
                COLUMN_NAMES['hq_country_col'],
                GEO_KEYWORD_MAP,
                'Geo'
            )
        
        reg_columns = COLUMN_NAMES['license_text_cols']
        combined_reg_text = pd.Series('', index=step.index)
        for col in reg_columns:
            if col in step.columns:
                combined_reg_text = combined_reg_text + ' ' + step[col].fillna('')
        
        step['Combined_Regulatory_Text'] = combined_reg_text
        step = self.create_indicator_columns(
            step,
            'Combined_Regulatory_Text',
            REGULATION_KEYWORD_MAP,
            'Reg'
        )
        
        if COLUMN_NAMES['compliance_text_cols'][0] in step.columns:
            step = self.create_indicator_columns(
                step,
                COLUMN_NAMES['compliance_text_cols'][0],
                COMPLIANCE_KEYWORD_MAP,
                'Compliance'
            )
        
        if COLUMN_NAMES['governance_col'] in step.columns:
            step = self.create_indicator_columns(
                step,
                COLUMN_NAMES['governance_col'],
                GOVERNANCE_KEYWORD_MAP,
                'Governance'
            )
        
        step = self.calculate_exchange_regulation_score(step)
        
        self.processed_df = step
        self.logger.info("Feature processing completed")
        
        return step

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the processed data."""
        if self.processed_df is None:
            raise ValueError("Data not processed yet. Call process_all_features() first.")
        
        stats = {}
        
        if 'Exchange_Reg' in self.processed_df.columns:
            stats['exchange_reg'] = {
                'mean': self.processed_df['Exchange_Reg'].mean(),
                'std': self.processed_df['Exchange_Reg'].std(),
                'min': self.processed_df['Exchange_Reg'].min(),
                'max': self.processed_df['Exchange_Reg'].max(),
                'median': self.processed_df['Exchange_Reg'].median()
            }
        
        for col in ['License_Count', 'Compliance_Maturity', 'Product_Complexity', 'Listed']:
            if col in self.processed_df.columns:
                non_zero_rate = (self.processed_df[col] > 0).mean()
                stats[f'{col.lower()}_non_zero_rate'] = non_zero_rate
        
        return stats

    def save_results(self, output_path: str = 'processed_exchange_data.csv') -> None:
        """Save processed results to CSV."""
        if self.processed_df is None:
            raise ValueError("Data not processed yet. Call process_all_features() first.")
        
        self.processed_df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")