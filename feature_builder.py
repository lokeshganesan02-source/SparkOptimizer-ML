# feature_builder.py  — UPDATED for banking dataset
# Extracts ML features from SQL query + dataset path
# Save as: C:\Aravindh\feature_builder.py  (REPLACE old version)

import os
import re

def build_feature_vector(sql: str, dataset_path: str) -> dict:
    sql_upper = sql.upper()

    # ── SQL features ─────────────────────────────────────────────
    num_joins    = sql_upper.count("JOIN")
    num_aggs     = sum(sql_upper.count(fn) for fn in
                       ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "STDDEV(", "COUNTDISTINCT"])
    has_group_by = int("GROUP BY" in sql_upper)
    has_order_by = int("ORDER BY" in sql_upper)
    has_window   = int("OVER(" in sql_upper or "OVER (" in sql_upper)
    has_filter   = int("WHERE" in sql_upper or "FILTER" in sql_upper)
    has_distinct = int("DISTINCT" in sql_upper)
    has_fraud    = int("FRAUD" in sql_upper or "IS_FRAUD" in sql_upper)

    # ── Data features ─────────────────────────────────────────────
    try:
        size_bytes = os.path.getsize(dataset_path)
    except FileNotFoundError:
        size_bytes = 0

    input_size_mb  = round(size_bytes / (1024 * 1024), 2)
    input_size_gb  = round(input_size_mb / 1024, 4)

    # Estimated partitions based on Spark default 128MB per partition
    estimated_partitions = max(1, int(input_size_mb / 128))

    # ── Engineered features ───────────────────────────────────────
    # join_intensity: joins are expensive on large data
    join_intensity = round(num_joins * input_size_gb, 4)

    # agg_intensity: aggregations scale with data size
    agg_intensity  = round(num_aggs * input_size_gb, 4)

    # shuffle_risk_score: proxy for how much data movement
    shuffle_risk_score = (
        num_joins +
        num_aggs +
        has_group_by +
        has_window +
        has_order_by +
        has_filter
    )

    # fraud_complexity: banking-specific — fraud queries touch
    # more partitions due to skewed fraud distribution
    fraud_complexity = int(has_fraud) * (1 + num_joins)

    return {
        # SQL features
        "num_joins":           num_joins,
        "num_aggs":            num_aggs,
        "has_group_by":        has_group_by,
        "has_order_by":        has_order_by,
        "has_window":          has_window,
        "has_filter":          has_filter,
        "has_distinct":        has_distinct,
        "has_fraud":           has_fraud,

        # Data features
        "input_size_mb":        input_size_mb,
        "input_size_gb":        input_size_gb,
        "estimated_partitions": estimated_partitions,

        # Engineered
        "join_intensity":       join_intensity,
        "agg_intensity":        agg_intensity,
        "shuffle_risk_score":   shuffle_risk_score,
        "fraud_complexity":     fraud_complexity,
    }


if __name__ == "__main__":
    # Quick test
    sql  = "SELECT merchant_id, SUM(amount), AVG(amount) FROM transactions GROUP BY merchant_id"
    path = "C:/Aravindh/data/txn_5m.csv"
    feat = build_feature_vector(sql, path)
    print("Feature vector:")
    for k, v in feat.items():
        print(f"  {k:<25} {v}")