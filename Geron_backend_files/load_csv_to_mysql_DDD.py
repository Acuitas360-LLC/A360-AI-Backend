import pandas as pd
import mysql.connector
import math

# =========================
# MySQL Connection
# =========================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root@1234",
    database="rytelo_db",
    auth_plugin="mysql_native_password"
)

cursor = conn.cursor()

# =========================
# Read CSV
# =========================
csv_path = "rytelo_DDD.csv"
df = pd.read_csv(csv_path, dtype=str, low_memory=False)
zip_cols = ["campus_zip", "parent_zip"]

zip_cols = ["campus_zip", "parent_zip"]

for col in zip_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)  # remove .0
        .str.strip()
    )
# =========================
# Date Conversion
# =========================
df['date'] = pd.to_datetime(df['date'], errors='coerce')

df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# =========================
# Value Cleaner
# =========================
def clean_value(val):

    if val is None:
        return None

    if isinstance(val, float) and math.isnan(val):
        return None

    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("", "nan", "null", "none"):
            return None
        return val.strip()

    return val


# =========================
# Column Order (Must Match Table)
# =========================
df = df[[
    "campus_id",
    "campus_account_name",
    "date",
    "rytelo_total_mg",
    "reblozyl_total_mg",
    "rytelo_total_sls",
    "reblozyl_total_sls",
    "campus_zip",
    "campus_address",
    "campus_city",
    "campus_state",
    "campus_tier",
    "campus_account_type",
    "campus_calls",
    "campus_territory_id",
    "campus_territory",
    "campus_region_id",
    "campus_region",
    "parent_id",
    "parent_account_name",
    "parent_address",
    "parent_city",
    "parent_state",
    "parent_zip",
    "month_year",
    "quarter_year"
]]

# =========================
# Clean Rows
# =========================
data = []
for row in df.itertuples(index=False, name=None):
    cleaned_row = tuple(clean_value(v) for v in row)
    data.append(cleaned_row)


# =========================
# Insert Query
# =========================
insert_query = """
REPLACE INTO data_DDD (
    campus_id,
    campus_account_name,
    date,
    rytelo_total_mg,
    reblozyl_total_mg,
    rytelo_total_sls,
    reblozyl_total_sls,
    campus_zip,
    campus_address,
    campus_city,
    campus_state,
    campus_tier,
    campus_account_type,
    campus_calls,
    campus_territory_id,
    campus_territory,
    campus_region_id,
    campus_region,
    parent_id,
    parent_account_name,
    parent_address,
    parent_city,
    parent_state,
    parent_zip,
    month_year,
    quarter_year
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,%s, %s
)
"""

# =========================
# Insert Data
# =========================
cursor.executemany(insert_query, data)

conn.commit()

print(f"✅ Inserted {cursor.rowcount} rows successfully")

cursor.close()
conn.close()