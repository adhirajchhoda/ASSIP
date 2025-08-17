import pandas as pd
import numpy as np
import re
from pathlib import Path

class ExchangeAnalysis:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_file)
        if 'Crypto Exchange' in self.df.columns:
            self.df['Crypto Exchange'] = self.df['Crypto Exchange'].astype(str).str.strip()
        return self.df
    
    def calculate_bvi_listed(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        country_col = 'Country/Region(s) of Operation'
        if country_col in result.columns:
            bvi_pattern = r'bvi|british virgin islands'
            result['BVI'] = result[country_col].astype(str).str.contains(
                bvi_pattern, regex=True, case=False, na=False
            ).astype(int)
        else:
            result['BVI'] = 0
        
        gov_col = 'Ownership & Governance Structure'
        if gov_col in result.columns:
            listed_pattern = r'listed|public|publicly|ipo|nasdaq|nyse|lse|tsx'
            result['Listed'] = result[gov_col].astype(str).str.contains(
                listed_pattern, regex=True, case=False, na=False
            ).astype(int)
        else:
            result['Listed'] = 0
            
        return result
    
    def calculate_license_count(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
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
            
            parts = re.split(r'[;,\n\|]', text)
            license_parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
            
            return max(len(unique_licenses), len(license_parts))
        
        license_text = ''
        if reg_exposure_col in result.columns:
            license_text += ' ' + result[reg_exposure_col].fillna('')
        if reg_frameworks_col in result.columns:
            license_text += ' ' + result[reg_frameworks_col].fillna('')
            
        result['License_Count'] = pd.Series([count_licenses(text) for text in license_text])
        return result
    
    def calculate_compliance_maturity(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        compliance_col = 'Compliance Requirements (AML/KYC, disclosure, etc.)'
        
        def compliance_score(text):
            if not isinstance(text, str):
                return 0
            
            text_lower = text.lower()
            score = 0
            
            if 'full kyc' in text_lower or 'mandatory kyc' in text_lower:
                score += 3
            elif 'tiered kyc' in text_lower:
                score += 2
            elif 'optional kyc' in text_lower or 'partial kyc' in text_lower:
                score += 1
            elif 'kyc' in text_lower:
                score += 1
            
            audit_keywords = ['proof of reserves', 'audit', 'attest', 'reserves', 'merkle']
            if any(keyword in text_lower for keyword in audit_keywords):
                score += 1
                
            return score
        
        if compliance_col in result.columns:
            result['Compliance_Maturity'] = result[compliance_col].apply(compliance_score)
        else:
            result['Compliance_Maturity'] = 0
            
        return result
    
    def calculate_incident_count(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        incident_col = 'Regulatory Incidents (fines, violations, etc.)'
        
        def count_incidents(text):
            if not isinstance(text, str) or not text.strip():
                return 0
            
            parts = re.split(r'[;\n\|]', text)
            incidents = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
            return len(incidents)
        
        if incident_col in result.columns:
            result['Incident_Count'] = result[incident_col].apply(count_incidents)
        else:
            result['Incident_Count'] = 0
            
        return result
    
    def calculate_country_reg(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        country_col = 'Country/Region(s) of Operation'
        
        def extract_primary_country(text):
            if not isinstance(text, str) or not text.strip():
                return 'Unknown'
            
            text = text.strip()
            
            if 'hq' in text.lower():
                hq_match = re.search(r'hq[^,]*?([a-zA-Z\s]+)', text.lower())
                if hq_match:
                    return hq_match.group(1).strip().title()
            
            first_part = re.split(r'[,;&]', text)[0].strip()
            return first_part.title()
        
        if country_col in result.columns:
            result['Primary_Country'] = result[country_col].apply(extract_primary_country)
            
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
            
            result['Country_Reg'] = (
                0.5 * result['Country_License_Avg'].fillna(0) + 
                0.5 * result['Country_Compliance_Avg'].fillna(0)
            )
        else:
            result['Country_Reg'] = 0
            
        return result
    
    def calculate_exchange_reg(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result = self.calculate_bvi_listed(result)
        result = self.calculate_license_count(result)  
        result = self.calculate_compliance_maturity(result)
        result = self.calculate_incident_count(result)
        result = self.calculate_country_reg(result)
        
        listed_weight = 2.0
        
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
        if self.df is None:
            self.load_data()
        
        processed_df = self.calculate_exchange_reg(self.df)
        return processed_df