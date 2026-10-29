"""
Step 7: Load data into SQLite and run credit risk SQL queries.

This script:
1. Loads the cleaned German Credit data into an SQLite database.
2. Runs 4 SQL queries that demonstrate GROUP BY, CTEs, and window functions.
3. Prints results so we can verify them.
"""
import pandas as pd
import sqlite3
import os

# ── 1. Load cleaned data into SQLite ──
df = pd.read_csv('data/german_clean.csv')

# Also load predictions so we can query PD scores
test_pred = pd.read_csv('data/test_predictions.csv')
test_raw  = pd.read_csv('data/test.csv')
test_raw['predicted_pd'] = test_pred['predicted_pd']

db_path = 'data/german_credit.db'
# Remove old DB if it exists
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)

# Write full dataset and test set with predictions
df.to_sql('credit_data', conn, index=False, if_exists='replace')
test_raw.to_sql('test_scored', conn, index=False, if_exists='replace')

print(f"Loaded {len(df)} rows into 'credit_data' table")
print(f"Loaded {len(test_raw)} rows into 'test_scored' table (with predicted PDs)")

# ── 2. Run the SQL queries from the .sql file ──
with open('sql/credit_risk_queries.sql', 'r') as f:
    sql_content = f.read()

# Split on the query separator comments
queries = []
current_query_lines = []
current_title = ""

for line in sql_content.split('\n'):
    if line.startswith('-- QUERY'):
        if current_query_lines and current_title:
            queries.append((current_title, '\n'.join(current_query_lines)))
            current_query_lines = []
        elif not current_title:
            current_query_lines = []
        current_title = line
    elif line.strip() == '' or line.startswith('--'):
        if line.startswith('--') and not line.startswith('-- QUERY'):
            current_query_lines.append(line)
    else:
        current_query_lines.append(line)

if current_query_lines and current_title:
    queries.append((current_title, '\n'.join(current_query_lines)))

# Execute each query and print results
for title, query in queries:
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    # Remove comment lines for execution
    exec_query = '\n'.join(
        line for line in query.split('\n') if not line.strip().startswith('--')
    )
    try:
        result = pd.read_sql_query(exec_query, conn)
        print(result.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
        print(f"Query was:\n{exec_query}")

conn.close()
print(f"\nDatabase saved at {db_path}")

