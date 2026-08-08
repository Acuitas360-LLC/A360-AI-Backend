import pandas as pd

# Read Excel file
df = pd.read_excel(
    "data_867_pap_new.xlsx"
)

# Make column names lowercase
df.columns = df.columns.str.lower()

# Convert to datetime
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d")

# Week End Date (Friday)
df["week_end_date"] = df["transaction_date"] + pd.to_timedelta((4 - df["transaction_date"].dt.weekday) % 7, unit="D")

# ✅ Month Year → YYYY-MM
df["month_year"] = df["transaction_date"].dt.strftime("%Y-%m")

# ✅ Year and Quarter as nullable integers (avoids the float ".0" issue)
year = df["transaction_date"].dt.year.astype("Int64")
quarter = df["transaction_date"].dt.quarter.astype("Int64")

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
is_weekend = df["transaction_date"].dt.weekday >= 5

# Step 3: Holiday check
is_holiday = df["transaction_date"].isin(holiday_list)

# Step 4: Combine logic
df["is_business_day"] = (~(is_weekend | is_holiday)).astype(int)

# Save CSV
df.to_csv("data_867_pap_new_final.csv", index=False, encoding="utf-8")

print(df.head())