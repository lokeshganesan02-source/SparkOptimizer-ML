import re

def profile_sql(sql_text: str) -> dict:
    """
    Extract basic structural features from SQL query text.
    All features must be determinable BEFORE execution.
    """

    sql_lower = sql_text.lower()

    features = {}

    # Count JOINs
    features["num_joins"] = len(re.findall(r"\bjoin\b", sql_lower))

    # Count Aggregations
    agg_keywords = ["sum(", "count(", "avg(", "min(", "max("]
    features["num_aggs"] = sum(sql_lower.count(k) for k in agg_keywords)

    # Detect GROUP BY
    features["has_group_by"] = 1 if "group by" in sql_lower else 0

    # Detect ORDER BY
    features["has_order_by"] = 1 if "order by" in sql_lower else 0

    # Detect Window Functions
    features["has_window"] = 1 if " over(" in sql_lower else 0

    # Detect WHERE clause
    features["has_filter"] = 1 if "where " in sql_lower else 0

    # Detect DISTINCT
    features["has_distinct"] = 1 if "distinct" in sql_lower else 0

    return features


# ---------------------------------------------------
# Test block (can remove later)
# ---------------------------------------------------
if __name__ == "__main__":

    sample_sql = """
        SELECT d.Category, COUNT(*) as cnt
        FROM employees e
        JOIN department d ON e.Department = d.Department
        WHERE e.salary > 50000
        GROUP BY d.Category
        ORDER BY cnt DESC
    """

    result = profile_sql(sample_sql)
    print("SQL Profile:")
    for k, v in result.items():
        print(f"{k}: {v}")