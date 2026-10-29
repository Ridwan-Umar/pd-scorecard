"""
Step 6: Model Validation -- AUC, KS, Gini, and PSI

KEY CONCEPTS (for interview):
- AUC (Area Under ROC Curve): probability the model ranks a random default
  higher than a random non-default. 0.5 = coin flip, 1.0 = perfect.
  Industry: >0.7 acceptable, >0.8 good.

- KS (Kolmogorov-Smirnov): max separation between the cumulative distribution
  of defaults and non-defaults. Higher = better discrimination.
  Industry: >0.2 acceptable, >0.4 good.

- Gini = 2*AUC - 1. Rescales AUC to [0, 1]. Gini 0.5 = AUC 0.75.

- PSI (Population Stability Index): measures if the PD distribution shifted
  between train and test. Low PSI = stable model.
  Rule: <0.1 stable | 0.1-0.25 slight shift | >0.25 significant shift.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ── Load predictions from Step 5 ──
train_pred = pd.read_csv('data/train_predictions.csv')
test_pred  = pd.read_csv('data/test_predictions.csv')

y_train = train_pred['actual']
pd_train = train_pred['predicted_pd']
y_test = test_pred['actual']
pd_test = test_pred['predicted_pd']


# ── 1. AUC ──
auc_train = roc_auc_score(y_train, pd_train)
auc_test  = roc_auc_score(y_test, pd_test)
print(f"AUC  (Train): {auc_train:.4f}")
print(f"AUC  (Test):  {auc_test:.4f}")


# ── 2. Gini ──
gini_train = 2 * auc_train - 1
gini_test  = 2 * auc_test  - 1
print(f"Gini (Train): {gini_train:.4f}")
print(f"Gini (Test):  {gini_test:.4f}")


# ── 3. KS Statistic ──
def calc_ks(y_true, y_prob):
    """
    KS = max difference between cumulative % of defaults and non-defaults,
    when sorted by predicted PD. We use the ROC curve trick:
    tpr = cumulative % of defaults captured
    fpr = cumulative % of non-defaults captured
    KS  = max(tpr - fpr)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks

ks_train = calc_ks(y_train, pd_train)
ks_test  = calc_ks(y_test, pd_test)
print(f"KS   (Train): {ks_train:.4f}")
print(f"KS   (Test):  {ks_test:.4f}")


# ── 4. PSI (Population Stability Index) ──
def calc_psi(expected, actual, n_bins=10):
    """
    PSI measures distribution shift between two sets of predicted PDs.

    Steps:
    1. Bin the TRAIN predictions into n_bins equal-width buckets.
    2. Count what % of train and test predictions fall in each bucket.
    3. PSI = SUM[ (% test - % train) * ln(% test / % train) ]

    A small PSI means the model scores similarly on both sets.
    """
    # Create bins from the combined range of both distributions
    breakpoints = np.linspace(0, 1, n_bins + 1)

    # Count % in each bin (add small epsilon to avoid log(0))
    eps = 1e-4

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + eps
    actual_pct   = actual_counts   / len(actual)   + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

psi = calc_psi(pd_train, pd_test)
print(f"PSI (Train vs Test): {psi:.4f}")

if psi < 0.1:
    psi_verdict = "STABLE (< 0.1)"
elif psi < 0.25:
    psi_verdict = "SLIGHT SHIFT (0.1 - 0.25)"
else:
    psi_verdict = "SIGNIFICANT SHIFT (> 0.25)"
print(f"PSI Verdict: {psi_verdict}")


# ── 5. Summary Table ──
print(f"\n{'='*50}")
print(f"{'VALIDATION SUMMARY':^50}")
print(f"{'='*50}")
print(f"{'Metric':<25} {'Train':>10} {'Test':>10}")
print(f"{'-'*50}")
print(f"{'AUC':<25} {auc_train:>10.4f} {auc_test:>10.4f}")
print(f"{'Gini':<25} {gini_train:>10.4f} {gini_test:>10.4f}")
print(f"{'KS':<25} {ks_train:>10.4f} {ks_test:>10.4f}")
print(f"{'PSI (Train vs Test)':<25} {psi:>10.4f} {'':>10}")
print(f"{'-'*50}")


# ── 6. Plots ──

# ROC Curve
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, pd_train)
fpr_test,  tpr_test,  _ = roc_curve(y_test, pd_test)

axes[0].plot(fpr_train, tpr_train, label=f'Train (AUC={auc_train:.3f})')
axes[0].plot(fpr_test,  tpr_test,  label=f'Test (AUC={auc_test:.3f})')
axes[0].plot([0,1], [0,1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Plot 2: KS Plot (test set)
axes[1].plot(sorted(pd_test), np.linspace(0, 1, len(pd_test)), label='Overall')
# Separate defaults and non-defaults
pd_default = pd_test[y_test == 1]
pd_nondefault = pd_test[y_test == 0]
axes[1].plot(sorted(pd_default), np.linspace(0, 1, len(pd_default)), label='Defaults')
axes[1].plot(sorted(pd_nondefault), np.linspace(0, 1, len(pd_nondefault)), label='Non-defaults')
axes[1].set_xlabel('Predicted PD')
axes[1].set_ylabel('Cumulative %')
axes[1].set_title(f'KS Plot (Test KS={ks_test:.3f})')
axes[1].legend()

# Plot 3: PD Distribution (PSI visual)
bins = np.linspace(0, 1, 11)
axes[2].hist(pd_train, bins=bins, alpha=0.5, density=True, label='Train')
axes[2].hist(pd_test,  bins=bins, alpha=0.5, density=True, label='Test')
axes[2].set_xlabel('Predicted PD')
axes[2].set_ylabel('Density')
axes[2].set_title(f'PD Distribution (PSI={psi:.4f})')
axes[2].legend()

plt.tight_layout()
plt.savefig('validation_plots.png', dpi=150, bbox_inches='tight')
print(f"\nSaved validation_plots.png")

# ── 7. Save metrics for README ──
metrics = {
    'AUC_train': auc_train, 'AUC_test': auc_test,
    'Gini_train': gini_train, 'Gini_test': gini_test,
    'KS_train': ks_train, 'KS_test': ks_test,
    'PSI': psi
}
pd.DataFrame([metrics]).to_csv('data/validation_metrics.csv', index=False)
print("Saved data/validation_metrics.csv")

