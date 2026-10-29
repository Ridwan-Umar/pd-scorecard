"""
Step 2: Data Preparation for PD Scorecard
- Define default flag
- Handle missing values
- Remove target-leakage columns
"""
import pandas as pd
import numpy as np

# ── 1. Load raw data ──
# German Credit: 1,000 rows, space-separated, no header row.
# Column names come from the UCI documentation (german.doc).
cols = [
    'status_checking',   # Attr 1:  checking account status
    'duration',          # Attr 2:  loan duration in months
    'credit_history',    # Attr 3:  past credit behavior
    'purpose',           # Attr 4:  loan purpose
    'credit_amount',     # Attr 5:  loan amount in DM
    'savings',           # Attr 6:  savings account balance
    'employment',        # Attr 7:  years at current job
    'installment_rate',  # Attr 8:  installment as % of income
    'personal_status',   # Attr 9:  gender + marital status
    'other_debtors',     # Attr 10: guarantors / co-applicants
    'residence',         # Attr 11: years at current address
    'property',          # Attr 12: most valuable asset
    'age',               # Attr 13: age in years
    'other_plans',       # Attr 14: other installment plans
    'housing',           # Attr 15: rent / own / free
    'num_credits',       # Attr 16: # existing credits at this bank
    'job',               # Attr 17: employment skill level
    'dependents',        # Attr 18: # people to maintain
    'telephone',         # Attr 19: has registered phone?
    'foreign_worker',    # Attr 20: foreign worker?
    'target',            # 1 = Good, 2 = Bad (per UCI docs)
]

df = pd.read_csv('data/german.data', sep=' ', header=None, names=cols)
print(f"Raw shape: {df.shape}")

# ── 2. Define Default Flag ──
# UCI coding: 1 = Good, 2 = Bad.
# We remap to standard PD convention: 0 = Non-default, 1 = Default.
df['default'] = (df['target'] == 2).astype(int)
df.drop(columns=['target'], inplace=True)

print(f"\nDefault rate: {df['default'].mean():.1%}  ({df['default'].sum()} defaults out of {len(df)})")

# ── 3. Check for Missing Values ──
# The German Credit dataset uses codes like A14 = "no checking account"
# instead of NaN. There are no actual null values.
missing = df.isnull().sum()
print(f"\nTotal missing values: {missing.sum()}")
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found - dataset uses categorical codes for 'unknown'.")

# ── 4. Remove Target-Leakage Columns ──
# LEAKAGE ANALYSIS — which columns would NOT be available at the
# time of credit decision (i.e., at application)?
#
# DROPPED:
#   telephone   — whether phone is registered under customer's name.
#                 Not predictive of credit risk, just demographic noise.
#                 Including it risks encoding proxy discrimination.
#
#   foreign_worker — almost no variance (A201=963, A202=37).
#                    Also a protected characteristic in many jurisdictions.
#
# KEPT (all are known at application time):
#   status_checking, duration, credit_history, purpose, credit_amount,
#   savings, employment, installment_rate, personal_status, other_debtors,
#   residence, property, age, other_plans, housing, num_credits, job,
#   dependents
#
# NOTE: personal_status mixes gender + marital status (potential fair-
# lending concern). We keep it here for model performance but flag it
# as a limitation -- a production model would need disparate impact testing.

drop_cols = ['telephone', 'foreign_worker']
df.drop(columns=drop_cols, inplace=True)

print(f"\nDropped columns: {drop_cols}")
print(f"Remaining features: {df.shape[1] - 1}")  # minus default column
print(f"Final shape: {df.shape}")

# ── 5. Identify Feature Types ──
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df.select_dtypes(include='number').columns if c != 'default']

print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ── 6. Quick Sanity Check ──
print("\n-- Numerical summary --")
print(df[num_cols].describe().round(1))

print("\n-- Categorical value counts (first 3 features) --")
for col in cat_cols[:3]:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Save cleaned data for next step
df.to_csv('data/german_clean.csv', index=False)
print(f"\nSaved cleaned data to data/german_clean.csv")
