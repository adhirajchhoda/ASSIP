import pandas as pd
import numpy as np

def calculate_exchange_regulation_scores(df):
    result = df.copy()
    
    result['Listed'] = 0
    
    result['License_Count'] = (
        result[['Reg_MiCA', 'Reg_BitLicense', 'Reg_FCA', 'Reg_FinCEN_MSB', 
                'Reg_VASP', 'Reg_FATF', 'Reg_AUSTRAC', 'Reg_FINTRAC', 'Reg_SEC_CFTC']].sum(axis=1)
    )
    
    result['Incident_Count'] = 1
    
    result['Compliance_Maturity'] = np.random.randint(0, 5, len(result))
    
    country_weights = {
        'Geo_USA': 4.0, 'Geo_Singapore': 3.8, 'Geo_Hong_Kong': 3.6,
        'Geo_EU': 3.4, 'Geo_Canada': 3.2, 'Geo_Japan': 3.0,
        'Geo_Switzerland': 2.8, 'Geo_Australia': 2.6, 'Geo_UAE': 2.4,
        'Geo_Seychelles': 2.0, 'Geo_Cayman_Islands': 1.8, 'Geo_BVI': 1.0
    }
    
    result['Country_Reg'] = 0
    for geo_col, weight in country_weights.items():
        if geo_col in result.columns:
            result['Country_Reg'] += result[geo_col] * weight
    
    result['BVI'] = result.get('Geo_BVI', 0)
    
    result['Exchange_Reg'] = (
        2.0 * result['Listed'] +
        result['License_Count'] +
        result['Incident_Count'] +
        result['Compliance_Maturity'] +
        result['Country_Reg'] -
        result['BVI']
    )
    
    return result

def run_analysis():
    df = pd.read_csv("../crypto_exchanges_analysis_ready.csv")
    
    processed_df = calculate_exchange_regulation_scores(df)
    
    output_cols = ['Crypto Exchange', 'Listed', 'License_Count', 'Incident_Count',
                   'Compliance_Maturity', 'Country_Reg', 'BVI', 'Exchange_Reg']
    
    output_df = processed_df[output_cols].copy()
    output_df['Rank'] = output_df['Exchange_Reg'].rank(ascending=False, method='min').astype(int)
    output_df = output_df.sort_values('Exchange_Reg', ascending=False)
    
    output_df.to_csv('exchange_regulation_results.csv', index=False)
    
    return output_df

if __name__ == "__main__":
    results = run_analysis()