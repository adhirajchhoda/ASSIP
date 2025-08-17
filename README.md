# Cryptocurrency Exchange Regulation Analysis

Statistical analysis of cryptocurrency exchange regulatory compliance using composite scoring methodology.

## Research Methodology

Implementation of exchange regulation formula:
```
Exchange_Reg = 2.0 × Listed + License_Count + Incident_Count + Compliance_Maturity + Country_Reg - BVI
```

## Formula Components

### Ved's Components
- **BVI**: Binary indicator for British Virgin Islands incorporation
- **Listed**: Binary indicator for public listing (overweighted 2.0x for transparency)

### Ethan's Components  
- **License_Count**: Count of distinct regulatory licenses/registrations
- **Compliance_Maturity**: KYC policy strength + proof of reserves indicators

### Adhiraj's Components
- **Incident_Count**: Count of regulatory incidents/violations
- **Country_Reg**: Country-level regulatory baseline score

## Results

### Dataset Statistics
- **Total exchanges analyzed**: 251
- **Exchange_Reg score range**: 0.0 - 40.0 (Mean: 10.07, Std: 5.47)

### Formula Component Statistics
| Component | Mean | Non-zero Rate |
|-----------|------|---------------|
| Listed | 0.15 | 14.7% |
| License_Count | 4.14 | 96.8% |
| Incident_Count | 1.35 | 83.7% |
| Compliance_Maturity | 1.51 | 90.8% |
| Country_Reg | 2.82 | 98.8% |
| BVI | 0.03 | 3.2% |

### Top 10 Most Regulated Exchanges
1. **Gate** (40.0)
2. **Bitget** (32.0)
3. **OKX** (30.0)
4. **HashKey Exchange** (28.0)
5. **Crypto.com Exchange** (25.5)
6. **BitMart** (22.5)
7. **KuCoin** (22.0)
8. **BVOX** (22.0)
9. **ByBit** (21.0)
10. **Bithumb** (21.0)

### Predictive Analysis

Testing additional features for Exchange_Reg prediction:

| Feature | Correlation | Description |
|---------|-------------|-------------|
| Product_Complexity | 0.303 | Weighted product offering complexity |
| Num_Products | 0.342 | Count of distinct products |
| Incident_Severity | 0.359 | Severity weighting of incidents |

**Linear Regression Performance**: R² = 0.299 (CV: -0.137 ± 0.283)
**Random Forest Performance**: R² = -0.020 (CV: -0.288 ± 0.278)

## File Structure

```
├── analysis_core.py           # Core Exchange_Reg calculation
├── statistical_analysis.py    # Statistical modeling framework  
├── corrected_analysis.py      # Main analysis pipeline
├── run_analysis.py           # Alternative execution script
├── config.py                 # Configuration constants
├── .gitignore               # Excludes proprietary data
└── README.md               # This file
```

## Usage

```bash
python corrected_analysis.py
```

Outputs: `cryptocurrency_exchange_regulation_analysis.csv`

## Data Sources

Analysis uses proprietary cryptocurrency exchange data (excluded from repository via .gitignore).

## Statistical Notes

- Formula verification: Manual calculation correlation = 1.000 (correct implementation)
- Cross-validation shows potential overfitting due to limited sample size
- Additional features show moderate correlation with composite regulation score
- Listed exchanges show higher regulation scores due to 2.0x weighting

## Implementation Details

- BVI detection via country/region text parsing
- Listed status extracted from governance structure descriptions  
- License counting uses canonical mapping to avoid double-counting
- Compliance maturity combines KYC policy analysis with audit indicators
- Country regulation baseline calculated from exchange-level averages
- Product complexity uses weighted scoring of derivative vs. spot offerings