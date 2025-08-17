"""
Core analysis implementation following the original plan:
Exchange_Reg = (Listed) + License_Count + Incident Count + Compliance_Maturity + Country_Reg - BVI
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

class ExchangeAnalysis:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.processed_df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_file)
        if 'Crypto Exchange' in self.df.columns:
            self.df['Crypto Exchange'] = self.df['Crypto Exchange'].astype(str).str.strip()
        return self.df
    
    def ved_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ved's task: BVI and Listed components"""
        result = df.copy()
        
        # BVI: Check for British Virgin Islands in country column
        country_col = 'Country/Region(s) of Operation'
        if country_col in result.columns:
            bvi_pattern = r'\b(bvi|british virgin islands)\b'
            result['BVI'] = result[country_col].astype(str).str.contains(
                bvi_pattern, regex=True, case=False, na=False
            ).astype(int)
        else:
            result['BVI'] = 0
        
        # Listed: Extract from governance structure, overweight for transparency
        gov_col = 'Ownership & Governance Structure'
        if gov_col in result.columns:
            listed_pattern = r'\b(listed|public|publicly|ipo|nasdaq|nyse|lse|tsx)\b'
            result['Listed'] = result[gov_col].astype(str).str.contains(
                listed_pattern, regex=True, case=False, na=False
            ).astype(int)
        else:
            result['Listed'] = 0
            
        return result
    
    def ethan_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ethan's task: License_Count and Compliance_Maturity"""
        result = df.copy()
        
        # License_Count: Count distinct licenses from regulatory exposure
        reg_exposure_col = 'Regulatory Exposure (licenses, jurisdictions)'
        reg_frameworks_col = 'Key Regulatory Frameworks (MiCA, BitLicense, etc.)'
        
        def count_licenses(text):
            if not isinstance(text, str) or not text.strip():
                return 0
            
            license_keywords = [
                'mica', 'bitlicense', 'fca', 'fincen', 'msb', 'vasp', 'fatf',
                'austrac', 'fintrac', 'sec', 'cftc', 'license', 'registration',
                'authorization', 'permit'
            ]
            
            text_lower = text.lower()
            unique_licenses = set()
            
            for keyword in license_keywords:
                if keyword in text_lower:
                    unique_licenses.add(keyword)
            
            # Also count by splitting on common delimiters
            parts = re.split(r'[;,\n\|]', text)
            license_parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
            
            return max(len(unique_licenses), len(license_parts))
        
        license_text = ''
        if reg_exposure_col in result.columns:
            license_text += ' ' + result[reg_exposure_col].fillna('')
        if reg_frameworks_col in result.columns:
            license_text += ' ' + result[reg_frameworks_col].fillna('')
            
        result['License_Count'] = pd.Series([count_licenses(text) for text in license_text])
        
        # Compliance_Maturity: KYC subcategories + proof of reserves
        compliance_col = 'Compliance Requirements (AML/KYC, disclosure, etc.)'
        
        def compliance_score(text):
            if not isinstance(text, str):
                return 0
            
            text_lower = text.lower()
            score = 0
            
            # KYC scoring
            if 'full kyc' in text_lower or 'mandatory kyc' in text_lower:
                score += 3
            elif 'tiered kyc' in text_lower:
                score += 2
            elif 'optional kyc' in text_lower or 'partial kyc' in text_lower:
                score += 1
            elif 'kyc' in text_lower:
                score += 1
            
            # Proof of reserves / audit scoring
            audit_keywords = ['proof of reserves', 'audit', 'attest', 'reserves', 'merkle']
            if any(keyword in text_lower for keyword in audit_keywords):
                score += 1
                
            return score
        
        if compliance_col in result.columns:
            result['Compliance_Maturity'] = result[compliance_col].apply(compliance_score)
        else:
            result['Compliance_Maturity'] = 0
            
        return result
    
    def adhiraj_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adhiraj's task: Incident_Count and Country_Reg"""
        result = df.copy()
        
        # Incident_Count: Count incidents from regulatory incidents text
        incident_col = 'Regulatory Incidents (fines, violations, etc.)'
        
        def count_incidents(text):
            if not isinstance(text, str) or not text.strip():
                return 0
            
            # Split by common delimiters and count non-empty parts
            parts = re.split(r'[;\n\|]', text)
            incidents = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
            return len(incidents)
        
        if incident_col in result.columns:
            result['Incident_Count'] = result[incident_col].apply(count_incidents)
        else:
            result['Incident_Count'] = 0
        
        # Country_Reg: Country-level regulation baseline
        country_col = 'Country/Region(s) of Operation'
        
        def extract_primary_country(text):
            if not isinstance(text, str) or not text.strip():
                return 'Unknown'
            
            # Extract first country mentioned
            text = text.strip()
            
            # Handle HQ patterns
            if 'hq' in text.lower():
                hq_match = re.search(r'hq[^,]*?([a-zA-Z\s]+)', text.lower())
                if hq_match:
                    return hq_match.group(1).strip().title()
            
            # Take first part before comma/semicolon
            first_part = re.split(r'[,;&]', text)[0].strip()
            return first_part.title()
        
        if country_col in result.columns:
            result['Primary_Country'] = result[country_col].apply(extract_primary_country)
            
            # Calculate country-level averages for regulation
            country_stats = result.groupby('Primary_Country').agg({
                'License_Count': 'mean',
                'Compliance_Maturity': 'mean'
            }).fillna(0)
            
            result = result.merge(
                country_stats.rename(columns={
                    'License_Count': 'Country_License_Avg',
                    'Compliance_Maturity': 'Country_Compliance_Avg'
                }),
                left_on='Primary_Country',
                right_index=True,
                how='left'
            )
            
            # Simple country regulation score
            result['Country_Reg'] = (
                0.5 * result['Country_License_Avg'].fillna(0) + 
                0.5 * result['Country_Compliance_Avg'].fillna(0)
            )
        else:
            result['Country_Reg'] = 0
            
        return result
    
    def calculate_exchange_reg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final Exchange_Reg score using original formula"""
        result = df.copy()
        
        # Apply all component calculations
        result = self.ved_components(result)
        result = self.ethan_components(result)  
        result = self.adhiraj_components(result)
        
        # Original formula with Listed overweighting
        listed_weight = 2.0  # Overweight for transparency
        
        result['Exchange_Reg'] = (
            listed_weight * result['Listed'] +
            result['License_Count'] +
            result['Incident_Count'] +
            result['Compliance_Maturity'] +
            result['Country_Reg'] -
            result['BVI']
        )
        
        return result
    
    def run_analysis(self) -> pd.DataFrame:
        """Run complete analysis following original plan"""
        if self.df is None:
            self.load_data()
        
        self.processed_df = self.calculate_exchange_reg(self.df)
        return self.processed_df
    
    def get_results_summary(self) -> Dict:
        """Get summary of results"""
        if self.processed_df is None:
            raise ValueError("Analysis not run yet")
            
        df = self.processed_df
        
        summary = {
            'total_exchanges': len(df),
            'exchange_reg_stats': {
                'mean': df['Exchange_Reg'].mean(),
                'std': df['Exchange_Reg'].std(),
                'min': df['Exchange_Reg'].min(),
                'max': df['Exchange_Reg'].max(),
                'median': df['Exchange_Reg'].median()
            },
            'component_stats': {
                'listed_rate': df['Listed'].mean(),
                'avg_license_count': df['License_Count'].mean(),
                'avg_incident_count': df['Incident_Count'].mean(),
                'avg_compliance_maturity': df['Compliance_Maturity'].mean(),
                'avg_country_reg': df['Country_Reg'].mean(),
                'bvi_rate': df['BVI'].mean()
            },
            'top_10_exchanges': df.nlargest(10, 'Exchange_Reg')[
                ['Crypto Exchange', 'Exchange_Reg', 'Listed', 'License_Count', 
                 'Incident_Count', 'Compliance_Maturity', 'Country_Reg', 'BVI']
            ].to_dict('records')
        }
        
        return summary