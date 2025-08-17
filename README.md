# Cryptocurrency Exchange Regulation Analysis

Statistical analysis of cryptocurrency exchange regulatory compliance using composite scoring methodology.

## Formula

```
Exchange_Reg = 2.0 × Listed + License_Count + Incident_Count + Compliance_Maturity + Country_Reg - BVI
```

## Components

**Listed**: Binary indicator for public listing (2.0x weighted)  
**License_Count**: Count of distinct regulatory licenses/registrations  
**Incident_Count**: Count of regulatory incidents/violations  
**Compliance_Maturity**: KYC policy strength + audit indicators  
**Country_Reg**: Country-level regulatory baseline  
**BVI**: Binary penalty for British Virgin Islands incorporation  

## Results

**251 exchanges analyzed**  
**Exchange_Reg range**: 0.0 - 40.0 (Mean: 10.07)

### Top 10 Exchanges
1. Gate (40.0)
2. Bitget (32.0)
3. OKX (30.0)
4. HashKey Exchange (28.0)
5. Crypto.com Exchange (25.5)
6. BitMart (22.5)
7. KuCoin (22.0)
8. BVOX (22.0)
9. ByBit (21.0)
10. Bithumb (21.0)

### Statistical Performance
**Linear Regression R²**: 0.299  
**Additional Features Correlation**: 0.303-0.359

## Usage

```bash
python main.py
```

Outputs: `exchange_regulation_results.csv`

## Files

- `main.py` - Main analysis pipeline
- `exchange_analysis.py` - Exchange_Reg calculation
- `statistical_models.py` - Predictive modeling
- `config.py` - Configuration constants

## Data

Proprietary cryptocurrency exchange data (excluded via .gitignore)