"""
Configuration settings for cryptocurrency exchange regulation analysis.
"""

import re
from typing import Dict, List, Tuple

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

COLUMN_NAMES = {
    'license_text_cols': [
        'Regulatory Exposure (licenses, jurisdictions)',
        'Key Regulatory Frameworks (MiCA, BitLicense, etc.)'
    ],
    'compliance_text_cols': [
        'Compliance Requirements (AML/KYC, disclosure, etc.)'
    ],
    'incident_text_cols': [
        'Regulatory Incidents (fines, violations, etc.)'
    ],
    'product_text_cols': [
        'Products Offered (spot, futures, options, etc.)',
        'Product Launch Dates',
        'Link to doc with full analysis'
    ],
    'hq_country_col': 'Country/Region(s) of Operation',
    'governance_col': 'Ownership & Governance Structure',
    'crypto_exchange_col': 'Crypto Exchange'
}

EXCHANGE_REG_WEIGHTS = {
    'listed_weight': 2.0,
    'time_weight': 0.1,
    'reg_framework_weight': 0.5,
    'product_complexity_weight': 0.2,
}

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

CURRENT_YEAR = 2025
CURRENT_MONTH = 6