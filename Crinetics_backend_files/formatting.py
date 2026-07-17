import pandas as pd

# Read Excel file
df = pd.read_csv(
    "rytelo_DDD_masked.csv",
    dtype={
        "child_id": str,
        "parent_id": str
    }, low_memory=False
)

# Make column names lowercase
df.columns = df.columns.str.lower()

# Convert to datetime
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# Week End Date (Friday)
#df["week_end_date"] = df["date"] + pd.to_timedelta((4 - df["date"].dt.weekday) % 7, unit="D")

# Month Year
df["month_year"] = df["date"].dt.strftime("%y-%b")

# Quarter Year
df["quarter_year"] = "Q" + df["date"].dt.quarter.astype(str) + "-" + df["date"].dt.strftime("%y")

# Convert week_end_date to required format
#df["week_end_date"] = df["week_end_date"].dt.strftime("%d-%m-%Y")

# Save CSV with explicit quoting
df.to_csv("rytelo_DDD.csv", index=False, encoding="utf-8")

print(df.head())