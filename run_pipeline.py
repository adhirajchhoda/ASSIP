

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Dict, Set, List
from datetime import datetime, date


LICENSE_CANON_MAP = {
    'mica': 'MiCA',
    'bitlicense': 'BitLicense', 
    'fca': 'FCA',
    'fincen': 'FinCEN_MSB',
    'msb': 'FinCEN_MSB',
    'vasp': 'VASP',
    'fatf': 'FATF',
    'austrac': 'AUSTRAC',
    'fintrac': 'FINTRAC',
    'sec': 'SEC_CFTC',
    'cftc': 'SEC_CFTC',
}

INCIDENT_SEVERITY_WEIGHTS = [
    (r'criminal|fraud|felony', 3.0),
    (r'class\s*action|settlement|consent order', 2.5),
    (r'fine|penalt(y|ies)|sanction', 2.0),
    (r'warning|notice|advisory', 1.0),
]

PRODUCT_WEIGHTS = {
    'Product_Options': 3.0,
    'Product_Futures': 3.0,
    'Product_Margin': 2.0,
    'Product_Staking': 1.0,
    'Product_P2P_OTC': 1.0,
    'Product_NFTs': 0.5,
    'Product_Launchpad': 1.0,
    'Product_Crypto_Loans': 2.0,
    'Product_Spot': 0.5,
}

PRODUCT_KEYWORDS = {
    'Product_Options': [r'option(s)?', r'perp(etuals)? option', r'vanilla option'],
    'Product_Futures': [r'future(s)?', r'perp(etu)?als?', r'perpetual'],
    'Product_Margin': [r'margin', r'leverage(d)? spot', r'leverage'],
    'Product_Staking': [r'staking', r'stake\b', r'earn(\b|ing|s)?', r'savings'],
    'Product_P2P_OTC': [r'p2p', r'otc', r'peer[- ]to[- ]peer'],
    'Product_NFTs': [r'nft(s)?', r'non[- ]fungible'],
    'Product_Launchpad': [r'launchpad', r'ieo', r'ido'],
    'Product_Crypto_Loans': [r'loan(s)?', r'borrow', r'lending'],
    'Product_Spot': [r'spot(\b| market| trading)'],
}


LICENSE_TEXT_COLS = [
    'Regulatory Exposure (licenses, jurisdictions)',
    'Key Regulatory Frameworks (MiCA, BitLicense, etc.)'
]

COMPLIANCE_TEXT_COLS = [
    'Compliance Requirements (AML/KYC, disclosure, etc.)'
]

INCIDENT_TEXT_COLS = [
    'Regulatory Incidents (fines, violations, etc.)'
]

PRODUCT_TEXT_COLS = [
    'Products Offered (spot, futures, options, etc.)',
    'Product Launch Dates',
    'Link to doc with full analysis'
]

HQ_COUNTRY_COL = 'Country/Region(s) of Operation'


GEO_KEYWORD_MAP = {
    'USA': ['usa', 'u.s.', 'united states', 'america'],
    'UK': ['uk', 'united kingdom'],
    'EU': ['eu', 'europe', 'eea', 'european union'],
    'China': ['china', 'mainland china'],
    'Hong_Kong': ['hong kong', 'hk'],
    'Singapore': ['singapore'],
    'Canada': ['canada'],
    'Japan': ['japan', 'jp'],
    'South_Korea': ['south korea', 'korea'],
    'Australia': ['australia', 'au'],
    'UAE': ['uae', 'united arab emirates', 'dubai'],
    'Switzerland': ['switzerland'],
    'Seychelles': ['seychelles'],
    'BVI': ['bvi', 'british virgin islands'],
    'Cayman_Islands': ['cayman islands'],
    'LATAM': ['latam', 'latin america', 'mexico', 'brazil', 'argentina', 'colombia', 'chile'],
    'Africa': ['africa', 'south africa', 'nigeria'],
    'Turkey': ['turkey'],
    'Russia': ['russia'],
    'India': ['india'],
    'USA_Restricted': ['restricted in us', 'barred from us', 'excludes us', 'excluding us', 
                      'excluding united states', 'except us', 'except united states'],
    'China_Restricted': ['restricted in china', 'excludes china', 'excluding china', 
                        'except china', 'excluding mainland china'],
}

ENHANCED_PRODUCT_KEYWORD_MAP = {
    'Spot': ['spot trading', 'spot'],
    'Futures': ['futures', 'perpetual', 'derivatives'],
    'Options': ['options'],
    'Margin': ['margin', 'leverage'],
    'Staking': ['staking', 'earn'],
    'P2P_OTC': ['p2p', 'otc', 'over-the-counter', 'over the counter'],
    'NFTs': ['nft', 'nfts', 'non-fungible'],
    'Launchpad': ['launchpad', 'ieo', 'initial exchange offering'],
    'Crypto_Loans': ['loans', 'lending', 'borrow'],
}

REGULATION_KEYWORD_MAP = {
    'MiCA': ['mica', 'micar', 'markets in crypto'],
    'BitLicense': ['bitlicense', 'bit license'],
    'FCA': ['fca', 'financial conduct authority'],
    'FinCEN_MSB': ['fincen', 'msb', 'money service business'],
    'VASP': ['vasp', 'virtual asset service provider'],
    'FATF': ['fatf', 'financial action task force'],
    'AUSTRAC': ['austrac'],
    'FINTRAC': ['fintrac'],
    'SEC_CFTC': ['sec', 'cftc', 'securities and exchange', 'commodity futures'],
}

COMPLIANCE_KEYWORD_MAP = {
    'AML_KYC': ['aml', 'kyc', 'anti-money laundering', 'know your customer', 'customer verification', 'identity verification', 'real-name', 'real name'],
    'Travel_Rule': ['travel rule', 'fatf travel', 'cross-border reporting', 'wire transfer rule'],
    'Transaction_Monitoring': ['transaction monitoring', 'transaction surveillance', 'suspicious activity', 'fraud detection', 'ai-based fraud'],
    'Sanctions_Screening': ['sanctions screening', 'ofac', 'sanctions compliance', 'embargo compliance'],
    'Disclosure_Requirements': ['disclosure', 'financial disclosure', 'regulatory disclosure', 'transparency', 'proof of reserves', 'proof-of-reserves'],
    'Audit_Reporting': ['audit', 'external audit', 'regulatory reporting', 'fiu reporting', 'str reporting', 'suspicious transaction'],
    'Tiered_KYC': ['tiered kyc', 'kyc tiers', 'verification levels', 'tiered verification', 'basic kyc', 'advanced kyc'],
    'Proof_of_Reserves': ['proof of reserves', 'proof-of-reserves', 'reserve verification', 'cold storage', 'hot wallet', 'wallet segregation'],
}

GOVERNANCE_KEYWORD_MAP = {
    'Public_Disclosure': ['public', 'publicly', 'disclosed', 'transparent', 'transparency', 'governance', 'ownership', 'structure', 'board', 'directors', 'ceo', 'founder', 'founded by', 'incorporated', 'inc', 'ltd', 'llc', 'corp', 'corporation', 'limited', 'company', 'partnership', 'joint venture', 'subsidiary', 'parent company', 'holding company', 'venture capital', 'private equity', 'ipo', 'initial public offering', 'stock exchange', 'nasdaq', 'nyse', 'listed', 'publicly traded', 'shareholders', 'stakeholders', 'investors', 'backed by', 'funded by', 'series a', 'series b', 'series c', 'funding round', 'valuation', 'market cap', 'enterprise value'],
}

def load_csv_smart(filepath: str) -> pd.DataFrame:
    """Smart CSV loader that handles both tab and comma delimiters."""
    print(f"Attempting to load {filepath}...")
    
    try:
        df = pd.read_csv(filepath, delimiter='\t')
        if len(df.columns) == 1:
            print("Tab delimiter resulted in single column, trying comma delimiter...")
            df = pd.read_csv(filepath, delimiter=',')
        print(f"Successfully loaded with {len(df.columns)} columns")
        return df
    except Exception as e:
        
        print(f"Tab delimiter failed: {e}, trying comma delimiter...")
        try:
            df = pd.read_csv(filepath, delimiter=',')
            print(f"Successfully loaded with {len(df.columns)} columns")
            return df
        except Exception as e2:
            print(f"Failed to load CSV: {e2}")
            raise

def create_indicator_columns(dataframe: pd.DataFrame, 
                           column_to_search: str, 
                           keyword_map: Dict[str, List[str]], 
                           new_prefix: str) -> pd.DataFrame:
    """Create indicator columns with negation detection."""
    print(f"\nProcessing column: {column_to_search}")
    print(f"Creating {len(keyword_map)} new features with prefix '{new_prefix}'")
    
    if column_to_search not in dataframe.columns:
        print(f"Warning: Column '{column_to_search}' not found, skipping...")
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
        print(f"  Created {new_column_name}: {matches} matches found")
    
    return df_result

def extract_launch_dates_and_create_time_features(dataframe: pd.DataFrame, 
                                                column_name: str) -> pd.DataFrame:
    """Extract launch dates from product launch dates column and create time decay features."""
    print(f"\nProcessing column: {column_name}")
    print("Creating time decay features based on product launch dates...")
    
    if column_name not in dataframe.columns:
        print(f"Warning: Column '{column_name}' not found, skipping...")
        return dataframe
    
    df_result = dataframe.copy()
    
    
    df_result['Time_Earliest_Launch'] = np.nan
    df_result['Time_Latest_Launch'] = np.nan
    df_result['Time_Product_Count'] = 0
    df_result['Time_Decay_Score'] = 0.0
    df_result['Time_Innovation_Score'] = 0.0
    df_result['Time_Recent_Activity'] = 0.0
    
    
    current_year = 2025
    current_month = 6  
    
    for idx, cell_value in df_result[column_name].items():
        if pd.isna(cell_value) or cell_value == '':
            continue
            
        cell_text = str(cell_value).lower()
        
        
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, cell_text)
        
        
        month_pattern = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b'
        months = re.findall(month_pattern, cell_text)
        
        
        specific_date_pattern = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(19|20)\d{2}\b'
        specific_dates = re.findall(specific_date_pattern, cell_text)
        
        
        relative_pattern = r'\b(since|founded|launched|started|established)\s+(19|20)\d{2}\b'
        relative_dates = re.findall(relative_pattern, cell_text)
        
        
        range_pattern = r'\b(19|20)\d{2}[-–—]\s*(19|20)\d{2}\b'
        ranges = re.findall(range_pattern, cell_text)
        
        
        all_years = []
        
        
        all_years.extend([int(year) for year in years])
        
        
        for month, day, year in specific_dates:
            all_years.append(int(year))
        
        
        for _, year in relative_dates:
            all_years.append(int(year))
        
        
        for start_year, end_year in ranges:
            all_years.extend([int(start_year), int(end_year)])
        
        
        all_years = sorted(list(set(all_years)))
        
        if all_years:
            earliest_year = min(all_years)
            latest_year = max(all_years)
            
            
            
            base_decay_score = max(0, 100 - (current_year - latest_year) * 10)
            
            
            year_span = latest_year - earliest_year + 1
            innovation_bonus = min(30, year_span * 5)  
            
            
            recent_bonus = 0
            if latest_year >= current_year - 1:
                recent_bonus = 20
            
            
            product_count = len(years) + len(specific_dates) + len(relative_dates)
            product_bonus = min(20, product_count * 2)  
            
            
            decay_score = base_decay_score + innovation_bonus + recent_bonus + product_bonus
            innovation_score = innovation_bonus + product_bonus
            recent_activity = 1 if latest_year >= current_year - 1 else 0
            
            
            df_result.loc[idx, 'Time_Earliest_Launch'] = earliest_year
            df_result.loc[idx, 'Time_Latest_Launch'] = latest_year
            df_result.loc[idx, 'Time_Product_Count'] = product_count
            df_result.loc[idx, 'Time_Decay_Score'] = decay_score
            df_result.loc[idx, 'Time_Innovation_Score'] = innovation_score
            df_result.loc[idx, 'Time_Recent_Activity'] = recent_activity
    
    
    df_result['Time_Earliest_Launch'] = df_result['Time_Earliest_Launch'].fillna(0)
    df_result['Time_Latest_Launch'] = df_result['Time_Latest_Launch'].fillna(0)
    df_result['Time_Decay_Score'] = df_result['Time_Decay_Score'].fillna(0)
    df_result['Time_Innovation_Score'] = df_result['Time_Innovation_Score'].fillna(0)
    
    
    print(f"  Created Time_Earliest_Launch: {df_result['Time_Earliest_Launch'].nunique()} unique years")
    print(f"  Created Time_Latest_Launch: {df_result['Time_Latest_Launch'].nunique()} unique years")
    print(f"  Created Time_Product_Count: {df_result['Time_Product_Count'].sum()} total products")
    print(f"  Created Time_Decay_Score: range {df_result['Time_Decay_Score'].min():.1f} to {df_result['Time_Decay_Score'].max():.1f}")
    print(f"  Created Time_Innovation_Score: range {df_result['Time_Innovation_Score'].min():.1f} to {df_result['Time_Innovation_Score'].max():.1f}")
    print(f"  Created Time_Recent_Activity: {df_result['Time_Recent_Activity'].sum()} exchanges with recent activity")
    
    return df_result

def _normalize_license_tokens(text: str) -> set:
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

def calculate_license_score(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct licenses/registrations using canonical mapping from real columns."""
    result = df.copy()
    canon_sets = [set() for _ in range(len(result))]
    
    
    for col in LICENSE_TEXT_COLS:
        if col in result.columns:
            series_sets = result[col].astype(str).apply(_normalize_license_tokens)
            canon_sets = [a.union(b) for a, b in zip(canon_sets, series_sets)]
    
    result['License_Count'] = pd.Series([len(s) for s in canon_sets], index=result.index).astype(int)
    return result

def _kyc_strength(text: str) -> int:
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

def _por_strength(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower()
    keywords = ['proof of reserves','attest','audit','audited','merkle','reserves']
    return int(any(k in t for k in keywords))

def calculate_compliance_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Parse compliance maturity from real spreadsheet columns."""
    result = df.copy()
    
    
    comp_col = 'Compliance Requirements (AML/KYC, disclosure, etc.)'
    
    if comp_col in result.columns:
        comp_text = result[comp_col].astype(str)
        
        
        result['KYC_Full'] = comp_text.apply(lambda x: int('full kyc' in x.lower() or 'mandatory kyc' in x.lower()))
        result['KYC_Tiered'] = comp_text.apply(lambda x: int('tiered kyc' in x.lower()))
        result['KYC_Optional'] = comp_text.apply(lambda x: int('optional kyc' in x.lower() or 'partial kyc' in x.lower()))
        result['Proof_of_Reserves'] = comp_text.apply(_por_strength)
        result['KYC_Strength_Score'] = comp_text.apply(_kyc_strength)
    else:
        
        result['KYC_Full'] = 0
        result['KYC_Tiered'] = 0
        result['KYC_Optional'] = 0
        result['Proof_of_Reserves'] = 0
        result['KYC_Strength_Score'] = 0
    
    result['Compliance_Maturity'] = result['KYC_Strength_Score'] + result['Proof_of_Reserves']
    return result

def _count_incidents(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    
    parts = [p.strip() for p in re.split(r'[;\n\|]', text) if p.strip()]
    return len(parts)

def _incident_severity_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lower = text.lower()
    score = 0.0
    for pattern, weight in INCIDENT_SEVERITY_WEIGHTS:
        if re.search(pattern, lower):
            score = max(score, weight)
    return score

def calculate_incident_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    inc_counts = pd.Series(0, index=result.index)
    inc_severity = pd.Series(0.0, index=result.index)
    
    
    for col in INCIDENT_TEXT_COLS:
        if col in result.columns:
            vals = result[col].astype(str)
            inc_counts = np.maximum(inc_counts.values, vals.apply(_count_incidents).values)
            inc_severity = np.maximum(inc_severity.values, vals.apply(_incident_severity_score).values)
    
    result['Incident_Count'] = inc_counts.astype(int)
    result['Incident_Severity'] = inc_severity
    return result

def _extract_primary_country(country_text: str) -> str:
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

def calculate_country_regulation(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-country baseline using licensing and compliance maturity priors."""
    result = df.copy()
    
    if HQ_COUNTRY_COL not in result.columns:
        result['Country_Reg'] = 0.0
        return result
    
    
    for needed in ['License_Count','Compliance_Maturity']:
        if needed not in result.columns:
            result[needed] = 0
    
    
    result['Primary_Country'] = result[HQ_COUNTRY_COL].apply(_extract_primary_country)
    
    
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

def _row_text_concat(row: pd.Series) -> str:
    parts = []
    for c in PRODUCT_TEXT_COLS:
        if c in row and isinstance(row[c], str) and row[c] != 'nan':
            parts.append(row[c])
    return ' | '.join(parts).lower()

def infer_product_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text_series = result.apply(_row_text_concat, axis=1)

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

def calculate_product_complexity_score(dataframe: pd.DataFrame, 
                                     product_columns: List[str]) -> pd.DataFrame:
    """Calculate a weighted product complexity score (0-10) for each exchange."""
    print(f"\nCalculating Product Complexity Scores...")
    print("Using weighted scoring system (0-10 scale)")
    
    df_result = dataframe.copy()
    
    
    
    PRODUCT_WEIGHTS = {
        'Product_Spot': 1.0,           
        'Product_Margin': 2.0,         
        'Product_Staking': 1.5,        
        'Product_P2P_OTC': 2.5,       
        'Product_Futures': 3.5,        
        'Product_Options': 4.0,        
        'Product_Launchpad': 2.0,      
        'Product_NFTs': 1.5,           
        'Product_Crypto_Loans': 3.0,   
    }
    
    
    df_result['Product_Complexity_Score'] = 0.0
    df_result['Product_Complexity_Breakdown'] = ''
    
    for idx in df_result.index:
        total_score = 0.0
        breakdown = []
        
        
        for col in product_columns:
            if col in PRODUCT_WEIGHTS and df_result.loc[idx, col] == 1:
                weight = PRODUCT_WEIGHTS[col]
                total_score += weight
                breakdown.append(f"{col.replace('Product_', '')}: {weight:.1f}")
        
        
        complexity_multiplier = 1.0
        
        
        active_products = sum([1 for col in product_columns if df_result.loc[idx, col] == 1])
        if active_products >= 5:
            complexity_multiplier *= 1.3  
        elif active_products >= 3:
            complexity_multiplier *= 1.15  
        
        
        has_futures = df_result.loc[idx, 'Product_Futures'] == 1 if 'Product_Futures' in product_columns else False
        has_options = df_result.loc[idx, 'Product_Options'] == 1 if 'Product_Options' in product_columns else False
        if has_futures and has_options:
            complexity_multiplier *= 1.25  
        
        
        has_lending = df_result.loc[idx, 'Product_Crypto_Loans'] == 1 if 'Product_Crypto_Loans' in product_columns else False
        if has_lending:
            complexity_multiplier *= 1.2  
        
        
        final_score = min(10.0, (total_score * complexity_multiplier) / 2.0)
        
        
        df_result.loc[idx, 'Product_Complexity_Score'] = round(final_score, 2)
        df_result.loc[idx, 'Product_Complexity_Breakdown'] = ' | '.join(breakdown) if breakdown else 'No products'
    
    
    scores = df_result['Product_Complexity_Score']
    print(f"  Created Product_Complexity_Score: range {scores.min():.2f} to {scores.max():.2f}")
    print(f"  Average complexity score: {scores.mean():.2f}")
    print(f"  Score distribution:")
    print(f"    Low (0-2): {(scores <= 2).sum()} exchanges")
    print(f"    Medium (2-5): {((scores > 2) & (scores <= 5)).sum()} exchanges")
    print(f"    High (5-8): {((scores > 5) & (scores <= 8)).sum()} exchanges")
    print(f"    Very High (8-10): {(scores > 8).sum()} exchanges")
    
    return df_result

def calculate_regulatory_framework_score(dataframe: pd.DataFrame, 
                                       regulatory_columns: List[str]) -> pd.DataFrame:
    """Calculate a weighted regulatory framework score (0-10) for each exchange."""
    print(f"\nCalculating Regulatory Framework Scores...")
    print("Using weighted scoring system (0-10 scale)")
    
    df_result = dataframe.copy()
    
    
    
    REGULATORY_WEIGHTS = {
        'Reg_MiCA': 2.5,           
        'Reg_BitLicense': 2.0,      
        'Reg_FCA': 2.0,            
        'Reg_FinCEN_MSB': 1.5,     
        'Reg_VASP': 1.5,           
        'Reg_FATF': 1.5,           
        'Reg_AUSTRAC': 1.0,        
        'Reg_FINTRAC': 1.0,        
        'Reg_SEC_CFTC': 2.5,       
    }
    
    
    df_result['Regulatory_Framework_Score'] = 0.0
    df_result['Regulatory_Framework_Breakdown'] = ''
    df_result['Regulatory_Jurisdiction_Count'] = 0
    df_result['Regulatory_Geographic_Diversity'] = 0.0
    
    for idx, row in df_result.iterrows():
        total_score = 0.0
        framework_details = []
        active_frameworks = 0
        
        
        for col in regulatory_columns:
            if col in REGULATORY_WEIGHTS and row[col] == 1:
                weight = REGULATORY_WEIGHTS[col]
                total_score += weight
                active_frameworks += 1
                
                
                framework_name = col.replace('Reg_', '')
                framework_details.append(f"{framework_name}: {weight}")
        
        
        
        if active_frameworks >= 5:
            diversity_bonus = 1.5  
        elif active_frameworks >= 3:
            diversity_bonus = 1.0  
        elif active_frameworks >= 2:
            diversity_bonus = 0.5  
        else:
            diversity_bonus = 0.0  
        
        
        
        if total_score >= 8.0:
            maturity_bonus = 1.0  
        elif total_score >= 5.0:
            maturity_bonus = 0.5  
        elif total_score >= 2.0:
            maturity_bonus = 0.25  
        else:
            maturity_bonus = 0.0  
        
        
        final_score = total_score + diversity_bonus + maturity_bonus
        
        
        
        normalized_score = min(10.0, (final_score / 18.5) * 10.0)
        
        
        df_result.loc[idx, 'Regulatory_Framework_Score'] = round(normalized_score, 2)
        df_result.loc[idx, 'Regulatory_Framework_Breakdown'] = ' | '.join(framework_details) if framework_details else 'No regulatory frameworks'
        df_result.loc[idx, 'Regulatory_Jurisdiction_Count'] = active_frameworks
        df_result.loc[idx, 'Regulatory_Geographic_Diversity'] = round(diversity_bonus, 2)
    
    
    scores = df_result['Regulatory_Framework_Score']
    print(f"  Created Regulatory_Framework_Score: range {scores.min():.2f} to {scores.max():.2f}")
    print(f"  Average regulatory score: {scores.mean():.2f}")
    print(f"  Score distribution:")
    print(f"    Low (0-2): {(scores <= 2).sum()} exchanges")
    print(f"    Medium (2-5): {((scores > 2) & (scores <= 5)).sum()} exchanges")
    print(f"    High (5-8): {((scores > 5) & (scores <= 8)).sum()} exchanges")
    print(f"    Very High (8-10): {(scores > 8).sum()} exchanges")
    
    return df_result

def calculate_product_complexity_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """Compute product complexity using 'final' flags after NLP inference."""
    result = infer_product_flags(df)
    final_cols = [f'{c}_final' for c in PRODUCT_KEYWORDS.keys() if f'{c}_final' in result.columns]
    
    if not final_cols:
        result['Product_Complexity'] = 0.0
        result['Num_Products'] = 0.0
        return result
    
    weights = {f'{k}_final': w for k, w in PRODUCT_WEIGHTS.items() if f'{k}_final' in result.columns}
    result['Product_Complexity'] = (result[list(weights.keys())] * pd.Series(weights)).sum(axis=1)
    result['Num_Products'] = result[final_cols].sum(axis=1)
    return result

def calculate_bvi_listed_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute columns for BVI and Listed using real column names."""
    result = df.copy()
    
    
    bvi_col = 'Geo_BVI'
    if bvi_col in result.columns:
        result['BVI'] = result[bvi_col].fillna(0).astype(int)
    else:
        result['BVI'] = 0
    
    
    gov_col = 'Ownership & Governance Structure'
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

def assemble_exchange_reg(df: pd.DataFrame) -> pd.DataFrame:
    """Run all component calculations and assemble Exchange_Reg score with enhanced features."""
    step = df.copy()
    
    print("=" * 70)
    print("Enhanced Exchange Regulation Pipeline")
    print("=" * 70)
    
    
    if 'Country/Region(s) of Operation' in step.columns:
        step = create_indicator_columns(
            step, 
            'Country/Region(s) of Operation',
            GEO_KEYWORD_MAP,
            'Geo'
        )
    
    
    if 'Products Offered (spot, futures, options, etc.)' in step.columns:
        step = create_indicator_columns(
            step,
            'Products Offered (spot, futures, options, etc.)',
            ENHANCED_PRODUCT_KEYWORD_MAP,
            'Product'
        )
        
        
        product_columns = [col for col in step.columns if col.startswith('Product_')]
        if product_columns:
            step = calculate_product_complexity_score(step, product_columns)
    
    
    reg_columns = [
        'Key Regulatory Frameworks (MiCA, BitLicense, etc.)',
        'Regulatory Exposure (licenses, jurisdictions)'
    ]
    
    combined_reg_text = pd.Series('', index=step.index)
    for col in reg_columns:
        if col in step.columns:
            combined_reg_text = combined_reg_text + ' ' + step[col].fillna('')
    
    step['Combined_Regulatory_Text'] = combined_reg_text
    step = create_indicator_columns(
        step,
        'Combined_Regulatory_Text',
        REGULATION_KEYWORD_MAP,
        'Reg'
    )
    
    
    reg_feature_columns = [col for col in step.columns if col.startswith('Reg_')]
    if reg_feature_columns:
        step = calculate_regulatory_framework_score(step, reg_feature_columns)
    
    
    if 'Compliance Requirements (AML/KYC, disclosure, etc.)' in step.columns:
        step = create_indicator_columns(
            step,
            'Compliance Requirements (AML/KYC, disclosure, etc.)',
            COMPLIANCE_KEYWORD_MAP,
            'Compliance'
        )
    
    
    if 'Ownership & Governance Structure' in step.columns:
        step = create_indicator_columns(
            step,
            'Ownership & Governance Structure',
            GOVERNANCE_KEYWORD_MAP,
            'Governance'
        )
    
    
    if 'Product Launch Dates' in step.columns:
        step = extract_launch_dates_and_create_time_features(step, 'Product Launch Dates')
    
    
    if 'Exchange Size (Large/Medium/Small)' in step.columns:
        print("\nProcessing Exchange Size (simple categorical)...")
        size_dummies = pd.get_dummies(
            step['Exchange Size (Large/Medium/Small)'], 
            prefix='Size',
            dummy_na=False
        )
        for col in size_dummies.columns:
            step[col] = size_dummies[col]
        print(f"  Created {len(size_dummies.columns)} size categories")
    
    
    step = calculate_bvi_listed_score(step)
    step = calculate_license_score(step)
    step = calculate_compliance_maturity(step)
    step = calculate_incident_score(step)
    step = calculate_country_regulation(step)
    
    
    
    listed_weight = 2.0  
    time_weight = 0.1    
    reg_framework_weight = 0.5  
    product_complexity_weight = 0.2  
    
    step['Exchange_Reg'] = (
        listed_weight * step['Listed'].fillna(0)
        + step['License_Count'].fillna(0)
        + step['Incident_Count'].fillna(0)
        + step['Compliance_Maturity'].fillna(0)
        + step['Country_Reg'].fillna(0)
        + time_weight * step.get('Time_Decay_Score', pd.Series(0, index=step.index)).fillna(0)
        + reg_framework_weight * step.get('Regulatory_Framework_Score', pd.Series(0, index=step.index)).fillna(0)
        + product_complexity_weight * step.get('Product_Complexity_Score', pd.Series(0, index=step.index)).fillna(0)
        - step['BVI'].fillna(0)
    )
    
    return step

def main():
    
    data_file = "Crypto Exchange Regulation Research Master Spreadsheet (2).csv"
    if not Path(data_file).exists():
        data_file = "original_spreadsheet.csv"  
    
    print(f"Loading {data_file}...")
    df = load_csv_smart(data_file)
    
    if 'Crypto Exchange' in df.columns:
        df['Crypto Exchange'] = df['Crypto Exchange'].astype(str).str.strip()
    
    print(f"Loaded {len(df)} exchanges")
    
    print("Running Exchange_Reg assembly pipeline...")
    scored_df = assemble_exchange_reg(df)
    
    
    print("\nSample Enhanced Exchange_Reg results:")
    sample_cols = ['Crypto Exchange', 'Listed', 'License_Count', 'Incident_Count', 'Compliance_Maturity', 
                   'Country_Reg', 'BVI', 'Product_Complexity_Score', 'Regulatory_Framework_Score', 
                   'Time_Decay_Score', 'Exchange_Reg']
    available_cols = [c for c in sample_cols if c in scored_df.columns]
    print(scored_df[available_cols].head(10).to_string())
    
    
    print(f"\nEnhanced Feature Non-zero rates:")
    enhanced_cols = ['License_Count', 'Compliance_Maturity', 'Listed', 'Product_Complexity_Score', 
                    'Regulatory_Framework_Score', 'Time_Decay_Score', 'Time_Recent_Activity']
    for col in enhanced_cols:
        if col in scored_df.columns:
            non_zero_rate = (scored_df[col] > 0).mean()
            print(f"{col}: {non_zero_rate:.2%}")
    
    
    print(f"\nEnhanced Feature Categories Created:")
    feature_categories = ['Geo_', 'Product_', 'Reg_', 'Compliance_', 'Governance_', 'Time_', 'Size_']
    for category in feature_categories:
        count = len([c for c in scored_df.columns if c.startswith(category)])
        print(f"  {category[:-1]} features: {count}")
    
    print(f"\nTotal enhanced features: {len([c for c in scored_df.columns if any(c.startswith(cat) for cat in feature_categories)])}")
    print(f"Exchange_Reg equals Incident_Count only: {(scored_df['Exchange_Reg'] == scored_df['Incident_Count']).mean():.2%}")
    
    
    cols_preferred_order = [
        'Crypto Exchange','Listed','License_Count','Incident_Count','Incident_Severity',
        'KYC_Full','KYC_Tiered','KYC_Optional','Proof_of_Reserves','KYC_Strength_Score',
        'Compliance_Maturity','Country_Reg','Primary_Country','BVI',
        'Product_Complexity_Score','Product_Complexity_Breakdown',
        'Regulatory_Framework_Score','Regulatory_Framework_Breakdown',
        'Regulatory_Jurisdiction_Count','Regulatory_Geographic_Diversity',
        'Time_Earliest_Launch','Time_Latest_Launch','Time_Product_Count',
        'Time_Decay_Score','Time_Innovation_Score','Time_Recent_Activity',
        'Product_Complexity','Num_Products','Exchange_Reg'
    ]
    
    existing_cols = [c for c in cols_preferred_order if c in scored_df.columns]
    master_cols = existing_cols + [c for c in scored_df.columns if c not in existing_cols]
    
    output_path = 'central_master.csv'
    scored_df[master_cols].to_csv(output_path, index=False)
    print(f'\nSaved: {output_path} with shape {scored_df.shape}')
    print(f'Key columns included: {existing_cols}')

if __name__ == '__main__':
    main()