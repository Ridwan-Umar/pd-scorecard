"""
Step 5: Logistic Regression PD Model

WHY LOGISTIC REGRESSION (for interview):
- Industry standard for PD scorecards because it outputs calibrated probabilities.
- Coefficients are interpretable: a positive coefficient means higher WoE
  (more "good") reduces the predicted probability of default.
- Regulators (Basel II/III) prefer transparent, explainable models.
- WoE-transformed inputs make it a purely linear model with no need for
  feature engineering or interaction terms.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ── Load WoE-transformed data from Step 4 ──
train_woe = pd.read_csv('data/train_woe.csv')
test_woe  = pd.read_csv('data/test_woe.csv')

X_train = train_woe.drop(columns=['default'])
y_train = train_woe['default']
X_test  = test_woe.drop(columns=['default'])
y_test  = test_woe['default']

feature_names = X_train.columns.tolist()
print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} WoE features")

# ── Fit Logistic Regression ──
# solver='lbfgs': standard quasi-Newton optimizer, works well here.
# max_iter=1000: ensure convergence (default 100 can be too few).
# C=1.0: regularization strength (1/lambda). Default value, no tuning needed
#         for a small dataset like this.
model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# ── Model Coefficients ──
# Each coefficient shows how much a 1-unit increase in WoE changes
# the log-odds of default. Negative = less likely to default.
coef_df = pd.DataFrame({
    'Feature': [f.replace('_woe', '') for f in feature_names],
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient')

print(f"\nIntercept: {model.intercept_[0]:.4f}")
print(f"\nCoefficients (sorted):")
print(coef_df.to_string(index=False))

# ── Predict Probabilities ──
# model.predict_proba returns [P(non-default), P(default)] for each row.
# We take column index 1 = P(default) = our PD estimate.
train_pd = model.predict_proba(X_train)[:, 1]
test_pd  = model.predict_proba(X_test)[:, 1]

print(f"\n-- Train PD distribution --")
print(f"  Mean PD:   {train_pd.mean():.4f}")
print(f"  Median PD: {np.median(train_pd):.4f}")
print(f"  Min PD:    {train_pd.min():.4f}")
print(f"  Max PD:    {train_pd.max():.4f}")

print(f"\n-- Test PD distribution --")
print(f"  Mean PD:   {test_pd.mean():.4f}")
print(f"  Median PD: {np.median(test_pd):.4f}")
print(f"  Min PD:    {test_pd.min():.4f}")
print(f"  Max PD:    {test_pd.max():.4f}")

# ── Save predictions for validation in Step 6 ──
pd.DataFrame({
    'actual': y_train.values,
    'predicted_pd': train_pd
}).to_csv('data/train_predictions.csv', index=False)

pd.DataFrame({
    'actual': y_test.values,
    'predicted_pd': test_pd
}).to_csv('data/test_predictions.csv', index=False)

# Save coefficients
coef_df.to_csv('data/coefficients.csv', index=False)

print(f"\nSaved data/train_predictions.csv, data/test_predictions.csv, data/coefficients.csv")

