import mysql.connector
import pandas as pd

# =========================
# MySQL connection config
# =========================
DB_CONFIG = {
    "host": "localhost",
    "port": "3306",
    "user": "root",
    "password": "root@1234",
    "database": "trodelvy_db"
}

# =========================
# SQL query
# =========================
SQL_QUERY = """
SELECT parent_account_type, SUM(qty_sold_pu) as total_sales FROM trodelvy_sales GROUP BY parent_account_type;
"""

def run_mysql_query(query: str) -> pd.DataFrame:
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)

        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)

        return df

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    result_df = run_mysql_query(SQL_QUERY)
    print(type(result_df))

    print("Query Result:")
    print(result_df)

    # Optional: save output
    # result_df.to_csv("query_result.csv", index=False)
