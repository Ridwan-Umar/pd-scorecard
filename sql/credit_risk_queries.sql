-- ============================================================
-- Credit Risk SQL Analysis -- German Credit Dataset
-- Database: SQLite (german_credit.db)
-- Tables:
--   credit_data  : 1000 rows, all features + default flag
--   test_scored  : 300 rows, test set with predicted PD
-- ============================================================

-- QUERY 1: Default rate by checking account status (GROUP BY)
-- Shows how checking account balance correlates with default risk.
-- A11 = negative balance, A12 = 0-200 DM, A13 = 200+ DM, A14 = no account.
SELECT
    status_checking,
    COUNT(*)                              AS total_customers,
    SUM("default")                        AS num_defaults,
    ROUND(AVG("default") * 100, 1)        AS default_rate_pct
FROM credit_data
GROUP BY status_checking
ORDER BY default_rate_pct DESC;

-- QUERY 2: Default rate by loan purpose and credit amount band (GROUP BY + CASE)
-- Segments loans into size buckets and shows which purposes are riskiest.
SELECT
    purpose,
    CASE
        WHEN credit_amount < 1500  THEN '1_low (<1500)'
        WHEN credit_amount < 4000  THEN '2_mid (1500-4000)'
        ELSE                            '3_high (4000+)'
    END                                   AS amount_band,
    COUNT(*)                              AS total,
    SUM("default")                        AS defaults,
    ROUND(AVG("default") * 100, 1)        AS default_rate_pct
FROM credit_data
GROUP BY purpose, amount_band
HAVING COUNT(*) >= 10
ORDER BY default_rate_pct DESC;

-- QUERY 3: Risk tier distribution using CTE
-- Buckets test-set predicted PDs into risk tiers and compares
-- predicted vs actual default rates per tier.
WITH risk_tiers AS (
    SELECT
        *,
        CASE
            WHEN predicted_pd < 0.15 THEN '1_Low Risk (<15%)'
            WHEN predicted_pd < 0.30 THEN '2_Medium Risk (15-30%)'
            WHEN predicted_pd < 0.50 THEN '3_High Risk (30-50%)'
            ELSE                          '4_Very High (50%+)'
        END AS risk_tier
    FROM test_scored
)
SELECT
    risk_tier,
    COUNT(*)                              AS num_customers,
    ROUND(AVG(predicted_pd) * 100, 1)     AS avg_predicted_pd_pct,
    ROUND(AVG("default") * 100, 1)        AS actual_default_rate_pct,
    SUM(credit_amount)                    AS total_exposure
FROM risk_tiers
GROUP BY risk_tier
ORDER BY risk_tier;

-- QUERY 4: Rank customers by PD within each checking account segment (Window Function)
-- Uses RANK() to find the top-3 highest-risk customers per segment,
-- and NTILE(4) to assign risk quartiles within each segment.
-- Window aliases can't be used in WHERE, so we wrap in a subquery.
SELECT *
FROM (
    SELECT
        status_checking,
        age,
        credit_amount,
        ROUND(predicted_pd, 4)                AS predicted_pd,
        RANK() OVER (
            PARTITION BY status_checking
            ORDER BY predicted_pd DESC
        )                                     AS risk_rank_in_segment,
        NTILE(4) OVER (
            PARTITION BY status_checking
            ORDER BY predicted_pd
        )                                     AS risk_quartile
    FROM test_scored
) ranked
WHERE risk_rank_in_segment <= 3
ORDER BY status_checking, risk_rank_in_segment;

