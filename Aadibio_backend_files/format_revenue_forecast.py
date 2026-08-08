import pandas as pd

# Read Excel file
df = pd.read_csv(
    "revenue_forecast_new_2.csv"
)

# Make column names lowercase
df.columns = df.columns.str.lower()

# Convert to datetime
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

# Week End Date (Friday)
df["week_end_date"] = df["date"] + pd.to_timedelta((4 - df["date"].dt.weekday) % 7, unit="D")

# ✅ Month Year → YYYY-MM
df["month_year"] = df["date"].dt.strftime("%Y-%m")

# ✅ Year and Quarter as nullable integers (avoids the float ".0" issue)
year = df["date"].dt.year.astype("Int64")
quarter = df["date"].dt.quarter.astype("Int64")

# ✅ Quarter Year → YYYY-QX
df["quarter_year"] = year.astype(str) + "-Q" + quarter.astype(str)

df["year"]=year
# -------------------------------
# ✅ Business Day Logic
# -------------------------------

# Step 1: Define holidays (IMPORTANT: same format as df["date"])
holiday_list = [
    "2026-01-01",
    "2026-05-25",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
    "2025-01-01",
    "2025-05-26",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    "2024-01-01",
    "2024-05-27",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    "2023-01-02",
    "2023-05-29",
    "2023-07-04",
    "2023-09-04",
    "2023-11-23",
    "2023-12-25"
    # Add more as needed
]

holiday_list = pd.to_datetime(holiday_list)

# Step 2: Weekend check (Saturday=5, Sunday=6)
is_weekend = df["date"].dt.weekday >= 5

# Step 3: Holiday check
is_holiday = df["date"].isin(holiday_list)

# Step 4: Combine logic
df["is_business_day"] = (~(is_weekend | is_holiday)).astype(int)

df["net_sales_forecast"]=df["net_sales_forecast"]*1000000
df["gross_sales_forecast"]=df["gross_sales_forecast"]*1000000

# Save CSV
df.to_csv("revenue_forecast_new_final.csv", index=False, encoding="utf-8")

print(df.head())