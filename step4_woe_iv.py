"""
Step 4: WoE (Weight of Evidence) Binning & IV (Information Value) Feature Selection

KEY CONCEPTS (for interview):
- WoE measures how much a bin separates defaults from non-defaults.
  Formula: WoE_i = ln(% of non-defaults in bin_i / % of defaults in bin_i)
  Positive WoE = more good customers; negative WoE = more bad customers.

- IV summarizes the total predictive power of a feature.
  Formula: IV = SUM[ (% non-default_i - % default_i) * WoE_i ]
  Rule of thumb:  <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 suspicious

- We fit WoE on TRAIN only, then apply the same mapping to TEST (no data leakage).
"""
import pandas as pd
import numpy as np

# ── Load train/test from Step 3 ──
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

y_train = train['default']
y_test  = test['default']

cat_cols = train.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in train.select_dtypes(include='number').columns if c != 'default']

print(f"Categorical: {cat_cols}")
print(f"Numerical:   {num_cols}")


# ── Helper: bin numerical features into quantiles ──
def bin_numerical(series, n_bins=5):
    """
    Bin a numerical column into n_bins roughly equal-sized groups.
    pd.qcut tries to make each bin have the same number of rows.
    duplicates='drop' handles cases where many values are the same.
    Returns a Series of bin labels like '(0.99, 12.0]'.
    """
    return pd.qcut(series, q=n_bins, duplicates='drop')


# ── Core function: compute WoE and IV for one feature ──
def calc_woe_iv(df, feature_col, target_col='default'):
    """
    Compute WoE per bin and IV for a single feature.

    Steps:
    1. Cross-tab: count defaults and non-defaults in each bin.
    2. Compute % of total defaults and % of total non-defaults per bin.
    3. WoE = ln(% non-default / % default) for each bin.
    4. IV  = sum of (% non-default - % default) * WoE across all bins.

    Returns: (woe_dict, iv_value)
      woe_dict maps each bin label to its WoE value.
      iv_value is a single number summarizing predictive power.
    """
    # Count events (default=1) and non-events (default=0) per bin
    crosstab = pd.crosstab(df[feature_col], df[target_col])

    # If a bin has 0 defaults or 0 non-defaults, WoE would be +/-inf.
    # We add 0.5 to each cell (Laplace smoothing) to avoid this.
    crosstab = crosstab + 0.5

    # Rename for clarity
    non_default = crosstab[0]
    default     = crosstab[1]

    # Percent of total non-defaults and defaults in each bin
    pct_non_default = non_default / non_default.sum()
    pct_default     = default / default.sum()

    # WoE per bin
    woe = np.log(pct_non_default / pct_default)

    # IV = sum of (pct_non_default - pct_default) * WoE
    iv = ((pct_non_default - pct_default) * woe).sum()

    # Build a dictionary: bin_label -> WoE value
    woe_dict = woe.to_dict()

    return woe_dict, iv


# ── Step A: Bin numerical features on TRAIN ──
# We store the bin edges so we can apply the SAME bins to TEST later.
bin_edges = {}   # feature_name -> list of bin edges from training
train_binned = train.copy()
test_binned  = test.copy()

for col in num_cols:
    # Create bins on train
    train_binned[col + '_bin'], edges = pd.qcut(
        train[col], q=5, duplicates='drop', retbins=True
    )
    bin_edges[col] = edges

    # Apply SAME bin edges to test (using pd.cut, not qcut)
    # Extend edges slightly so test values at boundaries are included
    edges_ext = edges.copy()
    edges_ext[0]  = -np.inf
    edges_ext[-1] =  np.inf
    test_binned[col + '_bin'] = pd.cut(test[col], bins=edges_ext)


# ── Step B: Compute WoE and IV for ALL features ──
woe_maps = {}    # feature_name -> {bin_label: woe_value}
iv_table = []    # list of (feature_name, iv_value)

print("\n" + "="*60)
print("INFORMATION VALUE (IV) TABLE")
print("="*60)

# Categorical features: use raw categories directly
for col in cat_cols:
    woe_dict, iv = calc_woe_iv(train, col)
    woe_maps[col] = woe_dict
    iv_table.append((col, iv))

# Numerical features: use the binned version
for col in num_cols:
    bin_col = col + '_bin'
    woe_dict, iv = calc_woe_iv(train_binned, bin_col)
    # Convert Interval keys to strings for easier mapping later
    woe_maps[col] = {str(k): v for k, v in woe_dict.items()}
    iv_table.append((col, iv))

# Sort by IV descending
iv_df = pd.DataFrame(iv_table, columns=['Feature', 'IV']).sort_values('IV', ascending=False)
iv_df['Strength'] = pd.cut(
    iv_df['IV'],
    bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
    labels=['Useless', 'Weak', 'Medium', 'Strong', 'Suspicious']
)
iv_df = iv_df.reset_index(drop=True)
print(iv_df.to_string(index=False))

# ── Step C: Select features with IV >= 0.02 ──
# Standard threshold: drop features with IV < 0.02 (no predictive power).
selected = iv_df[iv_df['IV'] >= 0.02]['Feature'].tolist()
dropped  = iv_df[iv_df['IV'] <  0.02]['Feature'].tolist()

print(f"\n--- Feature Selection ---")
print(f"Selected ({len(selected)}): {selected}")
print(f"Dropped  ({len(dropped)}):  {dropped}")

# ── Step D: Apply WoE transformation to train and test ──
# Replace each feature's raw values with their WoE values.
# This converts all features to numeric, which logistic regression needs.

def apply_woe(df, df_binned, feature, woe_dict, is_numerical=False):
    """
    Replace a feature's values with WoE scores.
    For categorical: map raw category -> WoE.
    For numerical:   map bin label (as string) -> WoE.
    Unknown categories get WoE = 0 (neutral, means no information).
    """
    if is_numerical:
        bin_col = feature + '_bin'
        return df_binned[bin_col].astype(str).map(woe_dict).fillna(0)
    else:
        return df[feature].map(woe_dict).fillna(0)


train_woe = pd.DataFrame()
test_woe  = pd.DataFrame()

for feat in selected:
    is_num = feat in num_cols
    train_woe[feat + '_woe'] = apply_woe(train, train_binned, feat, woe_maps[feat], is_num)
    test_woe[feat + '_woe']  = apply_woe(test, test_binned, feat, woe_maps[feat], is_num)

print(f"\nWoE-transformed train shape: {train_woe.shape}")
print(f"WoE-transformed test  shape: {test_woe.shape}")
print(f"\nTrain WoE sample (first 5 rows):")
print(train_woe.head())

# ── Save WoE-transformed data ──
train_woe['default'] = y_train.values
test_woe['default']  = y_test.values
train_woe.to_csv('data/train_woe.csv', index=False)
test_woe.to_csv('data/test_woe.csv', index=False)

# Save IV table for README
iv_df.to_csv('data/iv_table.csv', index=False)

print(f"\nSaved data/train_woe.csv, data/test_woe.csv, data/iv_table.csv")

