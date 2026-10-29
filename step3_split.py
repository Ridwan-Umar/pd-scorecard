"""
Step 3: Stratified Train/Test Split
- 70/30 split with fixed seed for reproducibility
- Stratified on 'default' so both sets keep the 30% default rate
- No out-of-time split because the dataset has no date column
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned data from Step 2
df = pd.read_csv('data/german_clean.csv')

# ── Stratified split ──
# stratify=y ensures train and test have the same default rate (~30%).
# random_state=42 makes the split reproducible every time you run it.
X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,      # 70% train, 30% test
    stratify=y,           # preserve default rate in both sets
    random_state=42       # fixed seed for reproducibility
)

# ── Verify ──
print(f"Train set: {X_train.shape[0]} rows, default rate = {y_train.mean():.1%}")
print(f"Test  set: {X_test.shape[0]} rows, default rate = {y_test.mean():.1%}")
print(f"Overall:   {len(df)} rows, default rate = {y.mean():.1%}")
print(f"\nTrain default counts:\n{y_train.value_counts().sort_index()}")
print(f"\nTest default counts:\n{y_test.value_counts().sort_index()}")

# Save splits for next steps
train = pd.concat([X_train, y_train], axis=1)
test  = pd.concat([X_test, y_test], axis=1)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print(f"\nSaved data/train.csv ({len(train)} rows) and data/test.csv ({len(test)} rows)")
