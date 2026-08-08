
import pandas as pd

# Read Excel file
df = pd.read_excel(
    "mtor_alerts.xlsx"
)

# Make column names lowercase
df.columns = df.columns.str.lower()

# Convert to datetime
df["therapy_start_date"] = pd.to_datetime(df["therapy_start_date"], format="%Y-%m-%d")

# Week End Date (Friday)
df["week_end_date"] = df["therapy_start_date"] + pd.to_timedelta((4 - df["therapy_start_date"].dt.weekday) % 7, unit="D")

# ✅ Month Year → YYYY-MM
df["month_year"] = df["therapy_start_date"].dt.strftime("%Y-%m")

# ✅ Year and Quarter as nullable integers (avoids the float ".0" issue)
year = df["therapy_start_date"].dt.year.astype("Int64")
quarter = df["therapy_start_date"].dt.quarter.astype("Int64")

# ✅ Quarter Year → YYYY-QX
df["quarter_year"] = year.astype(str) + "-Q" + quarter.astype(str)

df["year"]=year


# Save CSV
df.to_csv("mtor_alerts_final.csv", index=False, encoding="utf-8")

print(df.head())