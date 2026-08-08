import pandas as pd

# Read Excel file
df = pd.read_excel(
    "revenue_new.xlsx"
)

# Make column names lowercase
df.columns = df.columns.str.lower()

# Convert to datetime
df["shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d")

# Week End Date (Friday)
df["week_end_date"] = df["shipment_date"] + pd.to_timedelta((4 - df["shipment_date"].dt.weekday) % 7, unit="D")

# ✅ Month Year → YYYY-MM
df["month_year"] = df["shipment_date"].dt.strftime("%Y-%m")

# ✅ Year and Quarter as nullable integers (avoids the float ".0" issue)
year = df["shipment_date"].dt.year.astype("Int64")
quarter = df["shipment_date"].dt.quarter.astype("Int64")

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
is_weekend = df["shipment_date"].dt.weekday >= 5

# Step 3: Holiday check
is_holiday = df["shipment_date"].isin(holiday_list)

# Step 4: Combine logic
df["is_business_day"] = (~(is_weekend | is_holiday)).astype(int)


# ---- 2. Base unit price table (applies to everyone by default) ----
base_price_data = {
    "2022-01": 6785.0, "2022-02": 6785.0, "2022-03": 6785.0, "2022-04": 6785.0,
    "2022-05": 6785.0, "2022-06": 6785.0, "2022-07": 6785.0, "2022-08": 6785.0,
    "2022-09": 6785.0, "2022-10": 6785.0, "2022-11": 6785.0, "2022-12": 6785.0,

    "2023-01": 6982.3,  "2023-02": 6988.55, "2023-03": 6988.55, "2023-04": 6988.55,
    "2023-05": 6988.55, "2023-06": 6988.55, "2023-07": 7093.38, "2023-08": 7093.38,
    "2023-09": 7093.38, "2023-10": 7093.38, "2023-11": 7093.38, "2023-12": 7093.38,

    "2024-01": 7412.58, "2024-02": 7412.58, "2024-03": 7412.58, "2024-04": 7412.58,
    "2024-05": 7412.58, "2024-06": 7412.58, "2024-07": 7634.96, "2024-08": 7634.96,
    "2024-09": 7634.96, "2024-10": 7634.96, "2024-11": 7634.96, "2024-12": 7634.96,

    "2025-01": 7978.5,  "2025-02": 7978.5,  "2025-03": 7978.5,  "2025-04": 7978.5,
    "2025-05": 7978.5,  "2025-06": 7978.5,  "2025-07": 8203.1,  "2025-08": 8217.9,
    "2025-09": 8217.9,  "2025-10": 8217.9,  "2025-11": 8217.9,  "2025-12": 8217.9,

    "2026-01": 8587.7,  "2026-02": 8587.7,  "2026-03": 8587.7,  "2026-04": 8587.7,
    "2026-05": 8587.7,  "2026-06": 8587.7,  "2026-07": 8845.33, "2026-08": 8845.33,
    "2026-09": 8845.33, "2026-10": 8845.33,
}

# ---- 3. PANTHERRX-specific price table ----
pantherrx_price_data = {
    "2022-02": 6649.3, "2022-03": 6649.3, "2022-04": 6649.3, "2022-05": 6649.3,
    "2022-06": 6649.3, "2022-07": 6649.3, "2022-08": 6649.3, "2022-09": 6649.3,
    "2022-10": 6649.3, "2022-11": 6649.3, "2022-12": 6649.3,

    "2023-01": 6982.3, "2023-02": 6848.8, "2023-03": 6848.8, "2023-04": 6848.8,
    "2023-05": 6848.8, "2023-06": 6649.3, "2023-07": 6649.3, "2023-08": 6848.8,
    "2023-09": 6649.3, "2023-10": 6848.8, "2023-11": 6848.8, "2023-12": 6848.8,
}

# ---- 4. Discount table (month_year -> decimal discount) ----
discount_data = {
    "2022-02": 0.15,  "2022-03": 0.15,  "2022-04": 0.15,  "2022-05": 0.15,
    "2022-06": 0.15,  "2022-07": 0.15,  "2022-08": 0.15,  "2022-09": 0.15,
    "2022-10": 0.15,  "2022-11": 0.157, "2022-12": 0.069,

    "2023-01": 0.15,  "2023-02": 0.15,  "2023-03": 0.15,  "2023-04": 0.15,
    "2023-05": 0.15,  "2023-06": 0.15,  "2023-07": 0.15,  "2023-08": 0.15,
    "2023-09": 0.22,  "2023-10": 0.15,  "2023-11": 0.15,  "2023-12": 0.21,

    "2024-01": 0.165, "2024-02": 0.165, "2024-03": 0.173, "2024-04": 0.165,
    "2024-05": 0.165, "2024-06": 0.413, "2024-07": 0.17,  "2024-08": 0.17,
    "2024-09": 0.24,  "2024-10": 0.165, "2024-11": 0.165, "2024-12": 0.25,

    "2025-01": 0.205, "2025-02": 0.205, "2025-03": 0.279, "2025-04": 0.204,
    "2025-05": 0.205, "2025-06": 0.208, "2025-07": 0.207, "2025-08": 0.206,
    "2025-09": 0.177, "2025-10": 0.205, "2025-11": 0.205, "2025-12": 0.157,

    "2026-01": 0.20,  "2026-02": 0.209, "2026-03": 0.192, "2026-04": 0.209,
    "2026-05": 0.209, "2026-06": 0.209, "2026-07": 0.209, "2026-08": 0.209,
    "2026-09": 0.209, "2026-10": 0.209, "2026-11": 0.209, "2026-12": 0.209,
}

discount_date_exceptions = {
    pd.Timestamp("2024-01-01"): 0.15,
    pd.Timestamp("2025-03-27"): 0.165,
}
# ---- 4. Row-by-row logic applying the priority rules ----
def get_unit_price(row):
    # Rule 1: hard override for shipment_date = 2024-01-01
    if pd.notna(row["shipment_date"]) and row["shipment_date"] == pd.Timestamp("2024-01-01"):
        return 7093.38

    if pd.notna(row["shipment_date"]) and row["shipment_date"] == pd.Timestamp("2026-01-07"):
            return 8587.7

    # Rule 2: PANTHERRX-specific table, with fallback to base table
    if str(row.get("wholesaler_name", "")).strip().upper() == "PANTHERRX":
        if row["month_year"] in pantherrx_price_data:
            return pantherrx_price_data[row["month_year"]]
        else:
            return base_price_data.get(row["month_year"], None)

    # Rule 3: default base table
    return base_price_data.get(row["month_year"], None)

def get_discount(row):
    if pd.notna(row["shipment_date"]) and row["shipment_date"] in discount_date_exceptions:
        return discount_date_exceptions[row["shipment_date"]]

    return discount_data.get(row["month_year"], None)

df["unit_price"] = df.apply(get_unit_price, axis=1)
df["discount"] = df.apply(get_discount, axis=1)
df['wholesaler_name'] = df['wholesaler_name'].replace(
    {'ASD': 'CENCORA', 'ONCOLOGY SUPPLY': 'CENCORA'}
)

# ---- 5. Check for any unmatched rows ----
missing = df[df["unit_price"].isna()][["month_year", "wholesaler_name", "shipment_date"]]
missing_discount = df[df["discount"].isna()][["month_year", "shipment_date"]]

if not missing.empty:
    print("⚠️ Rows with no matching unit_price:")
    print(missing)

if not missing_discount.empty:
    print("⚠️ Rows with no matching discount:")
    print(missing_discount)

df["gross_sales"]=df["qty_sold"]*df["unit_price"]
df["net_sales"]=df["gross_sales"]-df["discount"]*df["gross_sales"]
# ---- 6. Save output ----
output_file = "revenue_new_final.csv"
df.to_csv(output_file, index=False)
print(f"Done. Saved to {output_file}")
