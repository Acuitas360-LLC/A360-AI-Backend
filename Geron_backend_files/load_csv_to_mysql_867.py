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
csv_path = "GE_BuyandBill.csv"
df = pd.read_csv(csv_path)
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
print(df[['date','week_end_date']].head(10))
#df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['week_end_date'] = pd.to_datetime(df['week_end_date'], format='%d-%m-%Y', errors='coerce')

df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df['week_end_date'] = df['week_end_date'].dt.strftime('%Y-%m-%d')

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
    "rytelo_total_sls",
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
    "parent_address",
    "parent_city",
    "parent_state",
    "parent_zip",
    "parent_account_name",
    "week_end_date",
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

print(df[['date','week_end_date']].head(10))
print(df[df['date'].isna()].head())
print(df["campus_zip"].head(20))
print(df["parent_zip"].head(20))
# =========================
# Insert Query
# =========================
insert_query = """
REPLACE INTO data_867 (
    campus_id,
    campus_account_name,
    date,
    rytelo_total_mg,
    rytelo_total_sls,
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
    parent_address,
    parent_city,
    parent_state,
    parent_zip,
    parent_account_name,
    week_end_date,
    month_year,
    quarter_year
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
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