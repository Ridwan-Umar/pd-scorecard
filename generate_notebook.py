"""
Generate the Jupyter notebook for the PD Scorecard project.
Combines all steps into a single .ipynb file.
"""
import json

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "outputs": [],
        "execution_count": None
    }

cells = []

# ── Title ──
cells.append(md_cell("# Probability of Default (PD) Scorecard\n## German Credit Dataset — UCI\n\nA logistic regression PD model with WoE/IV feature selection, built on the UCI German Credit dataset (1,000 applicants)."))

# ── Step 1: Imports ──
cells.append(md_cell("## Step 1: Setup & Imports"))
cells.append(code_cell("""import os
# Ensure we run from the project root regardless of where the notebook lives
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
print(f"Working directory: {os.getcwd()}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print("All imports successful")"""))

# ── Step 2: Data Prep ──
cells.append(md_cell("## Step 2: Data Preparation\n\n**Dataset**: UCI German Credit — 1,000 rows, 20 features, binary default flag.  \n**Default flag**: UCI uses 1=Good, 2=Bad. We remap to 0=Non-default, 1=Default.  \n**Dropped columns**: `telephone` (not risk-relevant), `foreign_worker` (near-zero variance, protected characteristic)."))
cells.append(code_cell("""# Column names from UCI documentation
cols = [
    'status_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence', 'property', 'age', 'other_plans', 'housing', 'num_credits',
    'job', 'dependents', 'telephone', 'foreign_worker', 'target'
]

df = pd.read_csv('data/german.data', sep=' ', header=None, names=cols)
print(f"Raw shape: {df.shape}")

# Remap target: 2 (Bad) -> 1 (Default), 1 (Good) -> 0 (Non-default)
df['default'] = (df['target'] == 2).astype(int)
df.drop(columns=['target'], inplace=True)

# Check missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Drop non-predictive / leakage columns
df.drop(columns=['telephone', 'foreign_worker'], inplace=True)

print(f"Default rate: {df['default'].mean():.1%} ({df['default'].sum()}/{len(df)})")
print(f"Final shape: {df.shape} (18 features + target)")

cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df.select_dtypes(include='number').columns if c != 'default']
print(f"Categorical ({len(cat_cols)}): {cat_cols}")
print(f"Numerical ({len(num_cols)}): {num_cols}")"""))

# ── Step 3: Split ──
cells.append(md_cell("## Step 3: Stratified Train/Test Split\n\n70/30 split with `stratify=y` to preserve the 30% default rate in both sets.  \nNo out-of-time split because the dataset has no date column."))
cells.append(code_cell("""X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print(f"Train: {X_train.shape[0]} rows, default rate = {y_train.mean():.1%}")
print(f"Test:  {X_test.shape[0]} rows, default rate = {y_test.mean():.1%}")"""))

# ── Step 4: WoE/IV ──
cells.append(md_cell("## Step 4: WoE Binning & IV Feature Selection\n\n**WoE** (Weight of Evidence) measures how much each bin separates defaults from non-defaults:  \n`WoE_i = ln(% non-defaults in bin_i / % defaults in bin_i)`\n\n**IV** (Information Value) summarizes total predictive power:  \n`IV = SUM[(% non-default_i - % default_i) * WoE_i]`\n\n| IV Range | Strength |\n|----------|----------|\n| < 0.02 | Useless |\n| 0.02 - 0.1 | Weak |\n| 0.1 - 0.3 | Medium |\n| 0.3 - 0.5 | Strong |\n| > 0.5 | Suspicious |"))
cells.append(code_cell("""def calc_woe_iv(df, feature_col, target_col='default'):
    \"\"\"Compute WoE per bin and IV for a single feature.\"\"\"
    # Cross-tab of feature bins vs default flag
    crosstab = pd.crosstab(df[feature_col], df[target_col])
    # Laplace smoothing: add 0.5 to avoid log(0) or division by zero
    crosstab = crosstab + 0.5
    
    non_default = crosstab[0]
    default = crosstab[1]
    
    pct_non_default = non_default / non_default.sum()
    pct_default = default / default.sum()
    
    woe = np.log(pct_non_default / pct_default)
    iv = ((pct_non_default - pct_default) * woe).sum()
    
    return woe.to_dict(), iv

# Bin numerical features into quantiles on TRAIN only
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

bin_edges = {}
train_binned = train.copy()
test_binned = test.copy()

for col in num_cols:
    train_binned[col + '_bin'], edges = pd.qcut(train[col], q=5, duplicates='drop', retbins=True)
    bin_edges[col] = edges
    edges_ext = edges.copy()
    edges_ext[0] = -np.inf
    edges_ext[-1] = np.inf
    test_binned[col + '_bin'] = pd.cut(test[col], bins=edges_ext)

# Compute WoE and IV for all features
woe_maps = {}
iv_table = []

for col in cat_cols:
    woe_dict, iv = calc_woe_iv(train, col)
    woe_maps[col] = woe_dict
    iv_table.append((col, iv))

for col in num_cols:
    woe_dict, iv = calc_woe_iv(train_binned, col + '_bin')
    woe_maps[col] = {str(k): v for k, v in woe_dict.items()}
    iv_table.append((col, iv))

iv_df = pd.DataFrame(iv_table, columns=['Feature', 'IV']).sort_values('IV', ascending=False)
iv_df['Strength'] = pd.cut(iv_df['IV'], bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                           labels=['Useless', 'Weak', 'Medium', 'Strong', 'Suspicious'])
print(iv_df.to_string(index=False))

# Select features with IV >= 0.02
selected = iv_df[iv_df['IV'] >= 0.02]['Feature'].tolist()
dropped = iv_df[iv_df['IV'] < 0.02]['Feature'].tolist()
print(f"\\nSelected ({len(selected)}): {selected}")
print(f"Dropped ({len(dropped)}): {dropped}")"""))

cells.append(md_cell("### Apply WoE transformation\nReplace raw feature values with their WoE scores. This makes all features numeric and encodes their relationship with the default flag."))
cells.append(code_cell("""def apply_woe(df, df_binned, feature, woe_dict, is_numerical=False):
    \"\"\"Map raw values to WoE scores. Unknown categories get WoE=0 (neutral).\"\"\"
    if is_numerical:
        return df_binned[feature + '_bin'].astype(str).map(woe_dict).fillna(0)
    else:
        return df[feature].map(woe_dict).fillna(0)

train_woe = pd.DataFrame()
test_woe = pd.DataFrame()

for feat in selected:
    is_num = feat in num_cols
    train_woe[feat + '_woe'] = apply_woe(train, train_binned, feat, woe_maps[feat], is_num)
    test_woe[feat + '_woe'] = apply_woe(test, test_binned, feat, woe_maps[feat], is_num)

print(f"WoE train shape: {train_woe.shape}")
print(f"WoE test shape:  {test_woe.shape}")
train_woe.head()"""))

# ── Step 5: Model ──
cells.append(md_cell("## Step 5: Logistic Regression PD Model\n\n**Why logistic regression?**\n- Industry standard for PD scorecards — outputs calibrated probabilities.\n- Coefficients are directly interpretable (important for regulators).\n- WoE inputs make it a purely linear model — no interaction terms needed."))
cells.append(code_cell("""model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model.fit(train_woe, y_train)

# Coefficients
coef_df = pd.DataFrame({
    'Feature': [f.replace('_woe', '') for f in train_woe.columns],
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient')
print(f"Intercept: {model.intercept_[0]:.4f}\\n")
print(coef_df.to_string(index=False))

# Predicted PDs
train_pd = model.predict_proba(train_woe)[:, 1]
test_pd = model.predict_proba(test_woe)[:, 1]

print(f"\\nTrain mean PD: {train_pd.mean():.4f} (actual: {y_train.mean():.4f})")
print(f"Test  mean PD: {test_pd.mean():.4f} (actual: {y_test.mean():.4f})")"""))

# ── Step 6: Validation ──
cells.append(md_cell("## Step 6: Validation — AUC, KS, Gini, PSI"))
cells.append(code_cell("""# AUC
auc_train = roc_auc_score(y_train, train_pd)
auc_test = roc_auc_score(y_test, test_pd)

# Gini = 2*AUC - 1
gini_train = 2 * auc_train - 1
gini_test = 2 * auc_test - 1

# KS = max(TPR - FPR) from the ROC curve
def calc_ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)

ks_train = calc_ks(y_train, train_pd)
ks_test = calc_ks(y_test, test_pd)

# PSI: measures distribution shift between train and test PDs
def calc_psi(expected, actual, n_bins=10):
    breakpoints = np.linspace(0, 1, n_bins + 1)
    eps = 1e-4
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

psi = calc_psi(train_pd, test_pd)

print(f"{'Metric':<15} {'Train':>10} {'Test':>10}")
print(f"{'-'*37}")
print(f"{'AUC':<15} {auc_train:>10.4f} {auc_test:>10.4f}")
print(f"{'Gini':<15} {gini_train:>10.4f} {gini_test:>10.4f}")
print(f"{'KS':<15} {ks_train:>10.4f} {ks_test:>10.4f}")
print(f"{'PSI':<15} {psi:>10.4f}")"""))

cells.append(md_cell("### Validation Plots"))
cells.append(code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ROC Curve
fpr_tr, tpr_tr, _ = roc_curve(y_train, train_pd)
fpr_te, tpr_te, _ = roc_curve(y_test, test_pd)
axes[0].plot(fpr_tr, tpr_tr, label=f'Train (AUC={auc_train:.3f})')
axes[0].plot(fpr_te, tpr_te, label=f'Test (AUC={auc_test:.3f})')
axes[0].plot([0,1], [0,1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve'); axes[0].legend()

# KS Plot (test)
pd_def = test_pd[y_test == 1]
pd_nondef = test_pd[y_test == 0]
axes[1].plot(sorted(pd_def), np.linspace(0, 1, len(pd_def)), label='Defaults')
axes[1].plot(sorted(pd_nondef), np.linspace(0, 1, len(pd_nondef)), label='Non-defaults')
axes[1].set_xlabel('Predicted PD'); axes[1].set_ylabel('Cumulative %')
axes[1].set_title(f'KS Plot (Test KS={ks_test:.3f})'); axes[1].legend()

# PD Distribution (PSI)
bins = np.linspace(0, 1, 11)
axes[2].hist(train_pd, bins=bins, alpha=0.5, density=True, label='Train')
axes[2].hist(test_pd, bins=bins, alpha=0.5, density=True, label='Test')
axes[2].set_xlabel('Predicted PD'); axes[2].set_ylabel('Density')
axes[2].set_title(f'PD Distribution (PSI={psi:.4f})'); axes[2].legend()

plt.tight_layout()
plt.savefig('validation_plots.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── Build notebook ──
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

with open('notebooks/pd_scorecard.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook saved to notebooks/pd_scorecard.ipynb")

