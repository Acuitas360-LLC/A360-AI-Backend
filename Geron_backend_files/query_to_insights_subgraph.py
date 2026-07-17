from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
import pandas as pd
import json
import os
from datetime import datetime, UTC
from typing import TypedDict, Literal, Optional, List
import re
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import mysql.connector
import warnings
warnings.filterwarnings("ignore")

TRACE_FILE = "agent_trace.json"

DB_CONFIG = {
    "host": "localhost",
    "port": "3306",
    "user": "root",
    "password": "root@1234",
    "database": "rytelo_db"
}

model=ChatOpenAI(model='gpt-5.2')
model_1=ChatOpenAI(model='gpt-5-mini')
model_2=ChatOpenAI(model='gpt-5')
# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

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

def append_agent_trace(
    file_path: str,
    question: str,
    agent_trace: list
):
    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Create a new run entry
    run_entry = {
        "run_id": datetime.now(UTC).isoformat() + "Z",
        "question": question,
        "trace": agent_trace
    }

    # Append
    data.append(run_entry)

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def current_quarter():
    month = datetime.today().month
    year = datetime.today().year % 100
    q = (month - 1) // 3 + 1
    return f"Q{q}-{year:02d}"

def current_month():
    today = datetime.today()
    year = today.year % 100           # last 2 digits
    month = today.strftime("%b")      # Jan, Feb, Mar...
    return f"{year:02d}-{month}"

CURRENT_MONTH=current_month()
CURRENT_QUARTER=current_quarter()

class ReviewDecision(TypedDict):
    source: Literal["sql_reviewer", "human"]
    decision: Literal["PASS", "REJECT"]
    reason: Optional[str]

def parse_review_output(text: str, source: str) -> ReviewDecision:
    raw = text.strip()

    # Normalize whitespace
    raw = re.sub(r"\s+", " ", raw)

    upper = raw.upper()

    if upper.startswith("PASS"):
        return {
            "source": source,
            "decision": "PASS",
            "reason": None
        }

    if upper.startswith("REJECT"):
        # Remove leading "REJECT" + optional punctuation
        reason = re.sub(
            r"^REJECT[\s,:;-]*",
            "",
            raw,
            flags=re.IGNORECASE
        ).strip()

        return {
            "source": source,
            "decision": "REJECT",
            "reason": reason if reason else None
        }

    # Safety fallback (treat as reject)
    return {
        "source": source,
        "decision": "REJECT",
        "reason": raw
    }




class AgentState(TypedDict):
    # inputs
    question: str

    # agent outputs
    query_decomposer_output: str | None
    sql_generator_output: str | None
    sql_reviewer_output: str | None
    human_reviewer_output: str | None
    active_review: Optional[ReviewDecision]
    sql_executor_output: Optional[dict]
    # control
    last_output: str

    # observability
    trace: list[dict]
    run_id: str

def log_trace(state, agent, event_type, text):
    state["trace"].append({
        "agent": agent,
        "event_type": event_type,
        "text": text
    })

def query_decomposer_node(state: AgentState):
    review = state["active_review"]
    user_input=state["question"]
    # print("Review from Either Human or SQL Reviewer from Query Decomposer")
    # print(review)
    if review and (review["decision"] == "REJECT"):
        #print("I am inside Active Review from Query Decomposer")

        prompt=f"""You are a Query Decomposer agent.

    Your responsibility is to analyze a natural-language user question and convert it into a structured, deterministic JSON specification that describes HOW a SQL query should be constructed by a downstream SQL Generator.

    You must NOT generate SQL.
    You must NOT generate pseudo-SQL.
    You must describe intent, logic, filters, aggregations, grouping, ordering, subqueries, and validation rules in structured JSON.

    The SQL Generator will rely entirely on your JSON output.

    ────────────────────────
    INPUT
    ────────────────────────
    You will receive:
    1. A natural-language user question
    2. The table schema and allowed column values
    3. Optional feedback from SQL Reviewer or Human
                                        
    USER QUERY
    ────────────────────────
    {user_input}
    ────────────────────────

    Previous decomposition:
        {state['query_decomposer_output']}

    Rejection source: {review['source']}
    Reason: {review['reason']}

    Revise the decomposition to address the feedback.

    ────────────────────────
    STRICT RULES (MANDATORY)
    ────────────────────────
    - Output MUST be valid JSON only
    - Do NOT output explanations or markdown
    - Do NOT output SQL or pseudo-SQL
    - Use ONLY the provided table and columns
    - Do NOT invent columns, tables, or values
    - Be explicit and deterministic
    - Every filter, aggregation, and grouping must be stated
    - If feedback is provided, revise ONLY the affected parts
    - Preserve correct logic from previous decompositions
    - If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)

    ────────────────────────
    Metric & Output Handling Rules (Must Always Be Enforced):
    ────────────────────────
    If time periods being compared are not of equal length, normalize all metrics to a weekly average before performing comparisons.
    If the user does not specify a time period, default to the most recent 52 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    All growth metrics must be expressed in percentage (%) format.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    Always append “%” to all growth and contribution values in the output.
    All campus entities roll up to their respective parent entities.
    If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
    For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. For all other time-based calculations, use date, with month_year and quarter_year as derived references when needed.
    For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    If the question asks for campus-level information, use only campus entities (do not aggregate or substitute with parent entities).
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
    Column Selection Rule: Use rytelo_total_sls and reblozyl_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on Rytelo, use rytelo_total_mg.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where rytelo_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    campus accounts are also referred to as child.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
    Whenever any time period is involved (including but not limited to weekly averages), the output must explicitly include the time period boundaries, i.e., the start date and end date (e.g., week_start_date and week_end_date). (VERY IMPORTANT)
    The dataset contains the following time fields derived from the transaction date: **date**, **week_end_date** (week ending Friday), **month_year**, and **quarter_year**.

When the user refers to **current, recent, last, or previous** month, quarter, or year, first determine the most recent available date using:

max_week_end_date = MAX(week_end_date)

The **current or recent period** is the period that contains max_week_end_date.

---

CALENDAR PERIOD BOUNDARIES

Time period boundaries must always be determined using the **calendar definition of the period**, not from the dataset.

Do not use MIN(date) or MAX(date) from the dataset to determine period_start or period_end.

Use calendar logic:

Month start = first day of the month
Month end = last day of the month

Quarter start = first day of the quarter
Quarter end = last day of the quarter

Year start = January 1
Year end = December 31

Dataset dates must **never define the start or end of a calendar period**.

---

PERIOD COMPLETENESS

A period is considered **complete only if the dataset contains data up to the calendar end of that period**.

Month is complete if:

max_week_end_date >= month_end_date

Quarter is complete if:

max_week_end_date >= quarter_end_date

Year is complete if:

max_week_end_date >= year_end_date

If:

max_week_end_date < calendar_period_end

then the period must be treated as **incomplete**.

Never determine completeness using the **number of weeks present in the data**.

---

WEEK DEFINITION

Weeks are defined using **week_end_date** and span:

Saturday (week_end_date − 6 days) → Friday (week_end_date)

---

WEEK INCLUSION RULE

When weekly normalization is required, include a week in a period **only if at least 3 days of that week fall within the target period**.

Determine overlap between:

Week window: [week_end_date − 6 , week_end_date]
Target period: [period_start , period_end]

Include the week **only if overlap ≥ 3 days**.

The ≥3-day rule determines **which weeks belong to a period**, but it must **never determine whether totals or weekly averages are used**.

---

AGGREGATION ANCHORING RULES

If the requested period is **complete**, aggregation must anchor directly to the calendar dimension:

Complete month → aggregate using **month_year**
Complete quarter → aggregate using **quarter_year**
Complete year or half-year → aggregate directly at the period level

In these cases compute:

SUM(rytelo_total_mg)

Do **not apply week inclusion logic** when the period is complete.

---

PERIOD COMPARISON RULE

When comparing periods (MoM, QoQ, YoY, half-year comparisons):

First determine whether **each period is complete**.

Then determine whether **normalization is required**.

Normalization rule:

If **both periods are complete**
→ compare **total sales for both periods**

If **at least one period is incomplete**
→ compute **weekly averages for both periods**

Weekly average formula:

weekly_average_sales = total_sales_in_period / number_of_weeks_included

---

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total sales vs Total sales
Weekly average vs Weekly average

Never compare **weekly averages for one period with total sales for the other period**.

---

WEEK NORMALIZATION USAGE

The ≥3-day week inclusion rule must **only be applied when weekly normalization is required**.

Differences in the number of weeks caused by the ≥3-day rule must **never trigger weekly normalization**.

Weekly averages should be used **only when at least one of the compared periods is incomplete**.

---

CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

1. Identify the time periods requested by the user.
2. Determine calendar boundaries for those periods.
3. Determine whether each period is complete using max_week_end_date.
4. Determine whether normalization is required.
5. If both periods are complete → aggregate totals directly at the period level.
6. If normalization is required → apply the ≥3-day week inclusion rule and compute weekly averages.
7. Perform the comparison.

---

Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.


────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_867 (Anchor this table for rytelo sales related information)

Columns, meanings, Original Values (Only those values exist in the table) 
and example values (values other than that can exist in the table) 


- date (DATE)
→ Actual shipment date of the product (transaction date)
→ Example values:
'2025-01-29', '2025-07-22', '2025-08-20', '2025-08-07', '2024-08-06'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00


- campus_zip (VARCHAR)
→ ZIP code of the campus account location
→ Example values:
'55433', '60050', '01702', '21740', '60612'


- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30008155', '30043877', '30006184', '30039854'


- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE MCHENRY HOSPITAL',
'CHARLES RIVER MEDICAL ASSOCIATES',
'ANTIETAM ONCOLOGY AND HEMATOLOGY GROUP PC',
'RUSH UNIVERSITY MEDICAL CENTER'


- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'11850 BLACKFOOT ST NW',
'4201 MEDICAL CENTER DR',
'233 W CENTRAL ST'


- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'COON RAPIDS', 'MCHENRY', 'NATICK', 'HAGERSTOWN', 'CHICAGO'


- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'MN', 'IL', 'MA', 'MD', 'ND'


- campus_tier (VARCHAR)
→ Tier classification of the campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'


- campus_account_type (VARCHAR)
→ Account archetype indicating whether the campus account belongs to an academic or community setting
→ Original values:
'ACADEMIC', 'COMMUNITY'


- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3


- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0405', 'FF0505', 'FF0101', 'FF0202', 'FF0201'


- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'Minneapolis, MN',
'Chicago, IL',
'Boston, MA',
'Baltimore, MD',
'South Jersey, NJ'


- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF04', 'FF05', 'FF01', 'FF02', 'FF03'


- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'Central', 'Great Lakes', 'North East', 'Mid Atlantic', 'South East'


- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30028542', '30014847', '30009531'


- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE',
'CHARLES RIVER MEDICAL ASSOCIATES',
'RUSH SYSTEM FOR HEALTH',
'HEART OF AMERICA MEDICAL CENTER'


- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'2550 UNIVERSITY AVE W',
'251 E HURON ST',
'233 W CENTRAL ST'


- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'SAINT PAUL', 'CHICAGO', 'NATICK'


- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'MN', 'IL', 'MA'


- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'55114', '60611', '01702'


- week_end_date (DATE)
→ Week ending date associated with the transaction date
→ Example values:
'31-01-2025', '25-07-2025', '22-08-2025', '08-08-2025', '09-08-2024'


- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Jan', '25-Jul', '25-Aug', '24-Aug', '26-Feb'


- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q1-25', 'Q3-25', 'Q3-24', 'Q1-26', 'Q4-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_DDD (Anchor this table for Competitor Level Information)

Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table):


- date (DATE)
→ Actual shipment date of the product
→ Example values:
'2025-08-31', '2025-09-30', '2025-01-31', '2024-09-30', '2024-08-31'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- reblozyl_total_mg (INT) (Competitor)
→ Total milligrams of Reblozyl shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- reblozyl_total_sls (DECIMAL)
→ Total sales value of reblozyl for the transaction
→ Represents the monetary value associated with the shipped reblozyl quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- campus_zip (VARCHAR)
→ ZIP code of the campus account
→ Example values:
'92660', '92123', '60451', '11354', '1301'

- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30026581', '30005403', '30012146', '30000194', '30035056'

- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MEMORIALCARE CANCER INSTITUTE',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'UNIVERSITY OF CHICAGO CANCER CENTER AT SILVER CROSS',
'CHUNG YUNHEE OFFICE',
'BAYSTATE FRANKLIN MEDICAL CENTER'

- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'1441 AVOCADO AVE',
'7930 FROST ST',
'1850 SILVER CROSS BLVD',
'14218 38TH AVE',
'164 HIGH ST'

- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'NEWPORT BEACH',
'SAN DIEGO',
'NEW LENOX',
'FLUSHING',
'GREENFIELD'

- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'CA', 'IL', 'NY', 'MA', 'FL'

- campus_tier (VARCHAR)
→ Tier classification of the campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'

- campus_account_type (VARCHAR)
→ Account archetype of the campus account
→ Original values:
'ACADEMIC', 'COMMUNITY'

- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3, 4

- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0605', 'FF0505', 'FF0105', 'FF0101', 'FF0304'

- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'San Diego, CA',
'Chicago, IL',
'Long Island, NY',
'Boston, MA',
'Miami, FL'

- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF06', 'FF05', 'FF01', 'FF03', 'FF04'

- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'West', 'Great Lakes', 'North East', 'South East', 'Central'

- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30031537.0', '30054837.0', '30031634.0', '30031368.0', '30046707.0'

- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MEMORIALCARE HEALTH SYSTEM',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'THE UNIVERSITY OF CHICAGO MEDICINE',
'BAYSTATE HEALTH',
'HEMATOLOGY ONCOLOGY ASSOCIATES OF THE PALM BEACHES'

- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'17360 BROOKHURST ST',
'7930 FROST ST',
'5841 S MARYLAND AVE',
'280 CHESTNUT ST',
'3450 LANTANA RD'

- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'FOUNTAIN VALLEY', 'SAN DIEGO', 'CHICAGO', 'SPRINGFIELD', 'LAKE WORTH'

- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'CA', 'IL', 'MA', 'FL', 'MS'

- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'92708', '92123', '60637', '1199', '33462'

- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Aug', '25-Sep', '25-Jan', '24-Sep', '24-Aug'

- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q3-25', 'Q1-25', 'Q3-24', 'Q4-24', 'Q2-25'


────────────────────────
TABLE INFORMATION
────────────────────────
Table name: targets

Important Notes:
- The table contains exactly one row.
- It acts like a dictionary / configuration table.
- Each column represents a target KPI value used for benchmarking or evaluation.
- The single row stores the global target values for the system.

- account_call_per_day (DECIMAL)
→ Target number of account calls to be made per day
→ Example values:
1.75, 2.00, 3.50

- hcp_touches (DECIMAL)
→ Target number of healthcare professional (HCP) interactions or touches
→ Example values:
2.0, 3.0, 4.5

- percentage_reach_tier_1_2_per_quarter (DECIMAL)
→ Target percentage of Tier 1 and Tier 2 accounts that should be reached in a quarter
→ Example values:
85, 90, 95

- percentage_reach_all_tier_per_quarter (DECIMAL)
→ Target percentage of all account tiers that should be reached in a quarter
→ Example values:
70, 75, 80

- frequency_tier_1_2_per_quarter (DECIMAL)
→ Target number of interactions with Tier 1 and Tier 2 accounts per quarter
→ Example values:
1, 1.5, 2

- frequency_all_tier_per_quarter (DECIMAL)
→ Target number of interactions with all account tiers per quarter
→ Example values:
1, 1.2, 1.5

- increase_depth_per_account (DECIMAL)
→ Target increase in engagement depth per account
→ Example values:
0.2, 0.3, 0.5

- average_number_of_active_patients (DECIMAL)
→ Target average number of active patients per account
→ Example values:
8, 10, 12

    ────────────────────────
    DATE & TIME LOGIC RULES
    ────────────────────────
    - If the user asks for "latest", "most recent", or "max date":
    → Explicitly require a subquery to compute MAX(date_column)
    → Never use system date
    - Rolling windows (e.g. last 13 weeks):
    → Must be calculated relative to the maximum date in the data
    - Quarters and months must align with quarter_year and month_year columns

    ────────────────────────
    REQUIRED JSON STRUCTURE
    ────────────────────────
    Your output MUST follow this structure:

    {{
    "intent_summary": string,
    "tables": [string],
    "filters": [
        {{
        "column": string,
        "operator": string,
        "value": string | number | "derived:max_date" | "derived:rolling_window"
        }}
    ],
    "aggregations": [
        {{
        "metric_name": string,
        "function": "SUM" | "COUNT" | "AVG",
        "column": string,
        "group_level": "none" | "column_name"
        }}
    ],
    "subqueries": [
        {{
        "name": string,
        "purpose": string,
        "logic": string
        }}
    ],
    "group_by": [string],
    "order_by": [
        {{
        "column": string,
        "direction": "ASC" | "DESC"
        }}
    ],
    "limit": number | null,
    "final_output": {{
        "columns": [string],
        "row_granularity": "single_row" | "per_group"
    }},
    "validation_rules": [string]
    }}

    ────────────────────────
    FINAL FULL EXAMPLE
    ────────────────────────

    {{
    "intent_summary": "Calculate total sales for the last 13 weeks based on the most recent date available in the dataset.",
    "tables": ["data_867"],
    "filters": [
        {{
        "column": "week_end_date",
        "operator": ">=",
        "value": "derived:rolling_window_13_weeks_from_max_date"
        }}
    ],
    "aggregations": [
        {{
        "metric_name": "total_sales",
        "function": "SUM",
        "column": "rytelo_total_mg",
        "group_level": "none"
        }}
    ],
    "subqueries": [
        {{
        "name": "max_date_cte",
        "purpose": "Identify the most recent week_end_date in the dataset",
        "logic": "Compute MAX(week_end_date) from data_867"
        }}
    ],
    "group_by": [],
    "order_by": [],
    "limit": null,
    "final_output": {{
        "columns": ["total_sales"],
        "row_granularity": "single_row"
    }},
    "validation_rules": [
        "Rolling window must be relative to MAX week_end_date",
        "Do not use system date",
        "Apply rolling window after max date is derived"
    ]
    }}

    ────────────────────────
    FINAL REMINDER
    ────────────────────────
    - Output ONLY valid JSON
    - Follow the required structure exactly
    - Do NOT output SQL, markdown, or explanations
    """

    else :

        prompt=f"""You are a Query Decomposer agent.

    Your responsibility is to analyze a natural-language user question and convert it into a structured, deterministic JSON specification that describes HOW a SQL query should be constructed by a downstream SQL Generator.

    You must NOT generate SQL.
    You must NOT generate pseudo-SQL.
    You must describe intent, logic, filters, aggregations, grouping, ordering, subqueries, and validation rules in structured JSON.

    The SQL Generator will rely entirely on your JSON output.

    ────────────────────────
    INPUT
    ────────────────────────
    You will receive:
    1. A natural-language user question
    2. The table schema and allowed column values
    3. Optional feedback from SQL Reviewer or Human
                                        
    USER QUERY
    ────────────────────────
    {user_input}
    ────────────────────────

    ────────────────────────
    STRICT RULES (MANDATORY)
    ────────────────────────
    - Output MUST be valid JSON only
    - Do NOT output explanations or markdown
    - Do NOT output SQL or pseudo-SQL
    - Use ONLY the provided table and columns
    - Do NOT invent columns, tables, or values
    - Be explicit and deterministic
    - Every filter, aggregation, and grouping must be stated
    - If feedback is provided, revise ONLY the affected parts
    - Preserve correct logic from previous decompositions

    ────────────────────────
    Metric & Output Handling Rules (Must Always Be Enforced):
    ────────────────────────
    If time periods being compared are not of equal length, normalize all metrics to a weekly average before performing comparisons.
    If the user does not specify a time period, default to the most recent 52 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    All growth metrics must be expressed in percentage (%) format.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    Always append “%” to all growth and contribution values in the output.
    All campus entities roll up to their respective parent entities.
    If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
    For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. For all other time-based calculations, use date, with month_year and quarter_year as derived references when needed.
    For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    If the question asks for campus-level information, use only campus entities (do not aggregate or substitute with parent entities).
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
    Column Selection Rule: Use rytelo_total_sls and reblozyl_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on Rytelo, use rytelo_total_mg.
    If the user does not specify a time window, default to 52 weeks.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where rytelo_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    campus accounts are also referred to as child.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
The dataset contains the following time fields derived from the transaction date: **date**, **week_end_date** (week ending Friday), **month_year**, and **quarter_year**.
Whenever any time period is involved (including but not limited to weekly averages), the output must explicitly include the time period boundaries, i.e., the start date and end date (e.g., week_start_date and week_end_date). (VERY IMPORTANT)
When the user refers to **current, recent, last, or previous** month, quarter, or year, first determine the most recent available date using:

max_week_end_date = MAX(week_end_date)

The **current or recent period** is the period that contains max_week_end_date.

---

CALENDAR PERIOD BOUNDARIES

Time period boundaries must always be determined using the **calendar definition of the period**, not from the dataset.

Do not use MIN(date) or MAX(date) from the dataset to determine period_start or period_end.

Use calendar logic:

Month start = first day of the month
Month end = last day of the month

Quarter start = first day of the quarter
Quarter end = last day of the quarter

Year start = January 1
Year end = December 31

Dataset dates must **never define the start or end of a calendar period**.

---

PERIOD COMPLETENESS

A period is considered **complete only if the dataset contains data up to the calendar end of that period**.

Month is complete if:

max_week_end_date >= month_end_date

Quarter is complete if:

max_week_end_date >= quarter_end_date

Year is complete if:

max_week_end_date >= year_end_date

If:

max_week_end_date < calendar_period_end

then the period must be treated as **incomplete**.

Never determine completeness using the **number of weeks present in the data**.

---

WEEK DEFINITION

Weeks are defined using **week_end_date** and span:

Saturday (week_end_date − 6 days) → Friday (week_end_date)

---

WEEK INCLUSION RULE

When weekly normalization is required, include a week in a period **only if at least 3 days of that week fall within the target period**.

Determine overlap between:

Week window: [week_end_date − 6 , week_end_date]
Target period: [period_start , period_end]

Include the week **only if overlap ≥ 3 days**.

The ≥3-day rule determines **which weeks belong to a period**, but it must **never determine whether totals or weekly averages are used**.

---

AGGREGATION ANCHORING RULES

If the requested period is **complete**, aggregation must anchor directly to the calendar dimension:

Complete month → aggregate using **month_year**
Complete quarter → aggregate using **quarter_year**
Complete year or half-year → aggregate directly at the period level

In these cases compute:

SUM(rytelo_total_mg)

Do **not apply week inclusion logic** when the period is complete.

---

PERIOD COMPARISON RULE

When comparing periods (MoM, QoQ, YoY, half-year comparisons):

First determine whether **each period is complete**.

Then determine whether **normalization is required**.

Normalization rule:

If **both periods are complete**
→ compare **total sales for both periods**

If **at least one period is incomplete**
→ compute **weekly averages for both periods**

Weekly average formula:

weekly_average_sales = total_sales_in_period / number_of_weeks_included

---

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total sales vs Total sales
Weekly average vs Weekly average

Never compare **weekly averages for one period with total sales for the other period**.

---

WEEK NORMALIZATION USAGE

The ≥3-day week inclusion rule must **only be applied when weekly normalization is required**.

Differences in the number of weeks caused by the ≥3-day rule must **never trigger weekly normalization**.

Weekly averages should be used **only when at least one of the compared periods is incomplete**.

---

CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

1. Identify the time periods requested by the user.
2. Determine calendar boundaries for those periods.
3. Determine whether each period is complete using max_week_end_date.
4. Determine whether normalization is required.
5. If both periods are complete → aggregate totals directly at the period level.
6. If normalization is required → apply the ≥3-day week inclusion rule and compute weekly averages.
7. Perform the comparison.

---

Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.


────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_867 (Anchor this table for sales related information)

Columns, meanings, Original Values (Only those values exist in the table) 
and example values (values other than that can exist in the table) 


- date (DATE)
→ Actual shipment date of the product (transaction date)
→ Example values:
'2025-01-29', '2025-07-22', '2025-08-20', '2025-08-07', '2024-08-06'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00


- campus_zip (VARCHAR)
→ ZIP code of the campus account location
→ Example values:
'55433', '60050', '01702', '21740', '60612'


- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30008155', '30043877', '30006184', '30039854'


- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE MCHENRY HOSPITAL',
'CHARLES RIVER MEDICAL ASSOCIATES',
'ANTIETAM ONCOLOGY AND HEMATOLOGY GROUP PC',
'RUSH UNIVERSITY MEDICAL CENTER'


- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'11850 BLACKFOOT ST NW',
'4201 MEDICAL CENTER DR',
'233 W CENTRAL ST'


- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'COON RAPIDS', 'MCHENRY', 'NATICK', 'HAGERSTOWN', 'CHICAGO'


- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'MN', 'IL', 'MA', 'MD', 'ND'


- campus_tier (VARCHAR)
→ Tier classification of the campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'


- campus_account_type (VARCHAR)
→ Account archetype indicating whether the campus account belongs to an academic or community setting
→ Original values:
'ACADEMIC', 'COMMUNITY'


- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3


- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0405', 'FF0505', 'FF0101', 'FF0202', 'FF0201'


- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'Minneapolis, MN',
'Chicago, IL',
'Boston, MA',
'Baltimore, MD',
'South Jersey, NJ'


- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF04', 'FF05', 'FF01', 'FF02', 'FF03'


- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'Central', 'Great Lakes', 'North East', 'Mid Atlantic', 'South East'


- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30028542', '30014847', '30009531'


- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE',
'CHARLES RIVER MEDICAL ASSOCIATES',
'RUSH SYSTEM FOR HEALTH',
'HEART OF AMERICA MEDICAL CENTER'


- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'2550 UNIVERSITY AVE W',
'251 E HURON ST',
'233 W CENTRAL ST'


- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'SAINT PAUL', 'CHICAGO', 'NATICK'


- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'MN', 'IL', 'MA'


- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'55114', '60611', '01702'


- week_end_date (DATE)
→ Week ending date associated with the transaction date
→ Example values:
'31-01-2025', '25-07-2025', '22-08-2025', '08-08-2025', '09-08-2024'


- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Jan', '25-Jul', '25-Aug', '24-Aug', '26-Feb'


- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q1-25', 'Q3-25', 'Q3-24', 'Q1-26', 'Q4-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_DDD (Anchor this table for Competitor Level Information)

Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):


- date (DATE)
→ Actual shipment date of the product
→ Example values:
'2025-08-31', '2025-09-30', '2025-01-31', '2024-09-30', '2024-08-31'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- reblozyl_total_mg (INT) (Competitor)
→ Total milligrams of Reblozyl shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- reblozyl_total_sls (DECIMAL)
→ Total sales value of reblozyl for the transaction
→ Represents the monetary value associated with the shipped reblozyl quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- campus_zip (VARCHAR)
→ ZIP code of the campus account
→ Example values:
'92660', '92123', '60451', '11354', '1301'

- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30026581', '30005403', '30012146', '30000194', '30035056'

- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MEMORIALCARE CANCER INSTITUTE',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'UNIVERSITY OF CHICAGO CANCER CENTER AT SILVER CROSS',
'CHUNG YUNHEE OFFICE',
'BAYSTATE FRANKLIN MEDICAL CENTER'

- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'1441 AVOCADO AVE',
'7930 FROST ST',
'1850 SILVER CROSS BLVD',
'14218 38TH AVE',
'164 HIGH ST'

- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'NEWPORT BEACH',
'SAN DIEGO',
'NEW LENOX',
'FLUSHING',
'GREENFIELD'

- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'CA', 'IL', 'NY', 'MA', 'FL'

- campus_tier (VARCHAR)
→ Tier classification of the campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'

- campus_account_type (VARCHAR)
→ Account archetype of the campus account
→ Original values:
'ACADEMIC', 'COMMUNITY'

- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3, 4

- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0605', 'FF0505', 'FF0105', 'FF0101', 'FF0304'

- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'San Diego, CA',
'Chicago, IL',
'Long Island, NY',
'Boston, MA',
'Miami, FL'

- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF06', 'FF05', 'FF01', 'FF03', 'FF04'

- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'West', 'Great Lakes', 'North East', 'South East', 'Central'

- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30031537', '30054837', '30031634', '30031368', '30046707'

- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MEMORIALCARE HEALTH SYSTEM',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'THE UNIVERSITY OF CHICAGO MEDICINE',
'BAYSTATE HEALTH',
'HEMATOLOGY ONCOLOGY ASSOCIATES OF THE PALM BEACHES'

- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'17360 BROOKHURST ST',
'7930 FROST ST',
'5841 S MARYLAND AVE',
'280 CHESTNUT ST',
'3450 LANTANA RD'

- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'FOUNTAIN VALLEY', 'SAN DIEGO', 'CHICAGO', 'SPRINGFIELD', 'LAKE WORTH'

- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'CA', 'IL', 'MA', 'FL', 'MS'

- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'92708.0', '92123', '60637', '1199', '33462'

- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Aug', '25-Sep', '25-Jan', '24-Sep', '24-Aug'

- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q3-25', 'Q1-25', 'Q3-24', 'Q4-24', 'Q2-25'


────────────────────────
TABLE INFORMATION
────────────────────────
Table name: targets

Important Notes:
- The table contains exactly one row.
- It acts like a dictionary / configuration table.
- Each column represents a target KPI value used for benchmarking or evaluation.
- The single row stores the global target values for the system.

- account_call_per_day (DECIMAL)
→ Target number of account calls to be made per day
→ Example values:
1.75, 2.00, 3.50

- hcp_touches (DECIMAL)
→ Target number of healthcare professional (HCP) interactions or touches
→ Example values:
2.0, 3.0, 4.5

- percentage_reach_tier_1_2_per_quarter (DECIMAL)
→ Target percentage of Tier 1 and Tier 2 accounts that should be reached in a quarter
→ Example values:
85, 90, 95

- percentage_reach_all_tier_per_quarter (DECIMAL)
→ Target percentage of all account tiers that should be reached in a quarter
→ Example values:
70, 75, 80

- frequency_tier_1_2_per_quarter (DECIMAL)
→ Target number of interactions with Tier 1 and Tier 2 accounts per quarter
→ Example values:
1, 1.5, 2

- frequency_all_tier_per_quarter (DECIMAL)
→ Target number of interactions with all account tiers per quarter
→ Example values:
1, 1.2, 1.5

- increase_depth_per_account (DECIMAL)
→ Target increase in engagement depth per account
→ Example values:
0.2, 0.3, 0.5

- average_number_of_active_patients (DECIMAL)
→ Target average number of active patients per account
→ Example values:
8, 10, 12

    ────────────────────────
    DATE & TIME LOGIC RULES
    ────────────────────────
    - If the user asks for "latest", "most recent", or "max date":
    → Explicitly require a subquery to compute MAX(date_column)
    → Never use system date
    - Rolling windows (e.g. last 13 weeks):
    → Must be calculated relative to the maximum date in the data
    - Quarters and months must align with quarter_year and month_year columns

    ────────────────────────
    REQUIRED JSON STRUCTURE
    ────────────────────────
    Your output MUST follow this structure:

    {{
    "intent_summary": string,
    "tables": [string],
    "filters": [
        {{
        "column": string,
        "operator": string,
        "value": string | number | "derived:max_date" | "derived:rolling_window"
        }}
    ],
    "aggregations": [
        {{
        "metric_name": string,
        "function": "SUM" | "COUNT" | "AVG",
        "column": string,
        "group_level": "none" | "column_name"
        }}
    ],
    "subqueries": [
        {{
        "name": string,
        "purpose": string,
        "logic": string
        }}
    ],
    "group_by": [string],
    "order_by": [
        {{
        "column": string,
        "direction": "ASC" | "DESC"
        }}
    ],
    "limit": number | null,
    "final_output": {{
        "columns": [string],
        "row_granularity": "single_row" | "per_group"
    }},
    "validation_rules": [string]
    }}

    ────────────────────────
    FINAL FULL EXAMPLE
    ────────────────────────

    {{
    "intent_summary": "Calculate total sales for the last 13 weeks based on the most recent date available in the dataset.",
    "tables": ["data_867"],
    "filters": [
        {{
        "column": "week_end_date",
        "operator": ">=",
        "value": "derived:rolling_window_13_weeks_from_max_date"
        }}
    ],
    "aggregations": [
        {{
        "metric_name": "total_sales",
        "function": "SUM",
        "column": "rytelo_total_mg",
        "group_level": "none"
        }}
    ],
    "subqueries": [
        {{
        "name": "max_date_cte",
        "purpose": "Identify the most recent week_end_date in the dataset",
        "logic": "Compute MAX(week_end_date) from data_867"
        }}
    ],
    "group_by": [],
    "order_by": [],
    "limit": null,
    "final_output": {{
        "columns": ["total_sales"],
        "row_granularity": "single_row"
    }},
    "validation_rules": [
        "Rolling window must be relative to MAX week_end_date",
        "Do not use system date",
        "Apply rolling window after max date is derived"
    ]
    }}

    ────────────────────────
    FINAL REMINDER
    ────────────────────────
    - Output ONLY valid JSON
    - Follow the required structure exactly
    - Do NOT output SQL, markdown, or explanations
    """
    result=model.invoke(prompt).content
    print("Query Decomposer Output")
    print(result)
    print("-"*100)
    state['query_decomposer_output']=result
    state["last_output"] = result
    state["active_review"] = None
    log_trace(state, "query_decomposer", "TextMessage", result)   

    return state

def sql_generator_node(state):
    user_input=state["question"]
    query_decomposer_output=state['query_decomposer_output']
    prompt = f"""You are an expert MySQL SQL Generator.

Your responsibility is to generate a valid MySQL SELECT query based STRICTLY on the structured JSON produced by the Query Decomposer.

You do NOT receive a natural-language question directly.
You MUST rely entirely on the Query Decomposer output.

────────────────────────
INPUTS YOU WILL RECEIVE
────────────────────────
1. Query Decomposer JSON (authoritative source of logic)
2. Table schema with column descriptions and example values
3. Optional FEEDBACK from a SQL Reviewer or Human

The Query Decomposer JSON defines:
- Intent
- Tables to use
- Filters and operators
- Aggregations and metrics
- Grouping logic
- Ordering and limits
- Subqueries (e.g., MAX date, rolling windows)
- Validation constraints

You must translate this JSON into executable MySQL SQL.

────────────────────────
STRICT RULES (MANDATORY)
────────────────────────
- Generate ONLY SELECT queries
- NEVER use DELETE, UPDATE, INSERT, DROP, ALTER, or TRUNCATE
- Use ONLY tables and columns explicitly present in the schema
- Use valid MySQL syntax
- Do NOT hallucinate columns, tables, or joins
- Do NOT add logic not present in the Query Decomposer JSON
- Do NOT explain the query
- Do NOT output markdown or commentary
- Output ONLY the SQL query
- When ONLY_FULL_GROUP_BY mode is enabled, ensure all non-aggregated columns in an aggregated query are either included in GROUP BY or wrapped inside an aggregate function (e.g., MAX(), MIN()).
- If using CROSS JOIN with a single-row CTE and mixing aggregated and non-aggregated fields, always wrap single-row fields inside an aggregate function to maintain MySQL compatibility.
- Never rely on implicit single-row behavior of CTEs; explicitly aggregate them to avoid strict SQL mode failures.
- Ensure all computed division denominators use NULLIF(column, 0) to prevent division-by-zero errors.
- All percentage outputs must be formatted using ROUND() and appended with '%' using CONCAT().
- If strict SQL mode may cause failure, prioritize query structure that is compliant without disabling SQL modes.
- If the user does not explicitly specify campus or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
- Follow structured logic: identify columns → filter → group → aggregate → sort/rank.
- Combine related calculations (totals, rankings, comparisons) into one cohesive query.
- Keep queries readable using clear aliases and section comments.
- Return only relevant, well-labeled results that fully answer the question.
- Always verify unique column values before applying filters to avoid incorrect conditions.

────────────────────────
Metric & Output Handling Rules (Must Always Be Enforced):
────────────────────────
If time periods being compared are not of equal length, normalize all metrics to a weekly average before performing comparisons.
If the user does not specify a time period, default to the most recent 52 weeks of available data.
If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
All growth metrics must be expressed in percentage (%) format.
LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
Always append “%” to all growth and contribution values in the outpSut.
While calculating Parent Account and campus BC Potential Segment metrics, merge “Others” and “Unknown” into “Low” before computation.
All campus entities roll up to their respective parent entities.
If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. For all other time-based calculations, use date, with month_year and quarter_year as derived references when needed.
For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
If the question asks for campus-level information, use only campus entities (do not aggregate or substitute with parent entities).
Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
Whenever the query references “nation,” compute the national-level metrics and include them in the output.
All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
Column Selection Rule: Use rytelo_total_sls and reblozyl_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on Rytelo, use rytelo_total_mg.
If the user does not specify a time window, default to 52 weeks.
If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where rytelo_total_mg > 0.
If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
Campus accounts are also referred to as child.
Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
If asked about sales by default give national sales don't group by campus_id or campus_account name.
Whenever any time period is involved (including but not limited to weekly averages), the output must explicitly include the time period boundaries, i.e., the start date and end date (e.g., week_start_date and week_end_date). (VERY IMPORTANT)
The dataset contains the following time fields derived from the transaction date: **date**, **week_end_date** (week ending Friday), **month_year**, and **quarter_year**.

When the user refers to **current, recent, last, or previous** month, quarter, or year, first determine the most recent available date using:

max_week_end_date = MAX(week_end_date)

The **current or recent period** is the period that contains max_week_end_date.

---

CALENDAR PERIOD BOUNDARIES

Time period boundaries must always be determined using the **calendar definition of the period**, not from the dataset.

Do not use MIN(date) or MAX(date) from the dataset to determine period_start or period_end.

Use calendar logic:

Month start = first day of the month
Month end = last day of the month

Quarter start = first day of the quarter
Quarter end = last day of the quarter

Year start = January 1
Year end = December 31

Dataset dates must **never define the start or end of a calendar period**.

---

PERIOD COMPLETENESS

A period is considered **complete only if the dataset contains data up to the calendar end of that period**.

Month is complete if:

max_week_end_date >= month_end_date

Quarter is complete if:

max_week_end_date >= quarter_end_date

Year is complete if:

max_week_end_date >= year_end_date

If:

max_week_end_date < calendar_period_end

then the period must be treated as **incomplete**.

Never determine completeness using the **number of weeks present in the data**.

---

WEEK DEFINITION

Weeks are defined using **week_end_date** and span:

Saturday (week_end_date − 6 days) → Friday (week_end_date)

---

WEEK INCLUSION RULE

When weekly normalization is required, include a week in a period **only if at least 3 days of that week fall within the target period**.

Determine overlap between:

Week window: [week_end_date − 6 , week_end_date]
Target period: [period_start , period_end]

Include the week **only if overlap ≥ 3 days**.

The ≥3-day rule determines **which weeks belong to a period**, but it must **never determine whether totals or weekly averages are used**.

---

AGGREGATION ANCHORING RULES

If the requested period is **complete**, aggregation must anchor directly to the calendar dimension:

Complete month → aggregate using **month_year**
Complete quarter → aggregate using **quarter_year**
Complete year or half-year → aggregate directly at the period level

In these cases compute:

SUM(rytelo_total_mg)

Do **not apply week inclusion logic** when the period is complete.

---

PERIOD COMPARISON RULE

When comparing periods (MoM, QoQ, YoY, half-year comparisons):

First determine whether **each period is complete**.

Then determine whether **normalization is required**.

Normalization rule:

If **both periods are complete**
→ compare **total sales for both periods**

If **at least one period is incomplete**
→ compute **weekly averages for both periods**

Weekly average formula:

weekly_average_sales = total_sales_in_period / number_of_weeks_included

---

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total sales vs Total sales
Weekly average vs Weekly average

Never compare **weekly averages for one period with total sales for the other period**.

---

WEEK NORMALIZATION USAGE

The ≥3-day week inclusion rule must **only be applied when weekly normalization is required**.

Differences in the number of weeks caused by the ≥3-day rule must **never trigger weekly normalization**.

Weekly averages should be used **only when at least one of the compared periods is incomplete**.

---

CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

1. Identify the time periods requested by the user.
2. Determine calendar boundaries for those periods.
3. Determine whether each period is complete using max_week_end_date.
4. Determine whether normalization is required.
5. If both periods are complete → aggregate totals directly at the period level.
6. If normalization is required → apply the ≥3-day week inclusion rule and compute weekly averages.
7. Perform the comparison.

---

Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.

────────────────────────
LOGIC TRANSLATION RULES
────────────────────────
- Every filter in the JSON MUST appear in the WHERE clause
- Every aggregation MUST appear exactly as defined
- group_by fields MUST be applied exactly as specified
- order_by MUST be applied only if present
- limit MUST be applied only if present
- Subqueries defined in the JSON MUST be implemented as CTEs or inline subqueries
- "derived:max_date" MUST be implemented using a MAX(date_column) subquery
- Rolling windows MUST be calculated relative to the derived max date, never system date
- Never infer dates using CURRENT_DATE unless explicitly instructed

────────────────────────
FEEDBACK HANDLING
────────────────────────
If FEEDBACK is provided:
- Fix ONLY the issues explicitly mentioned
- Do NOT introduce new logic
- Do NOT remove correct logic
- Preserve the structure implied by the Query Decomposer

────────────────────────
FINAL OUTPUT REQUIREMENT
────────────────────────
Output ONLY the final MySQL SELECT query.
No explanations.
No comments.
No additional text.

────────────────────────
    QUERY DECOMPOSITION
────────────────────────
{query_decomposer_output}

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_867

Columns, meanings, Original Values (Only those values exist in the table) 
and example values (values other than that can exist in the table) 


- date (DATE)
→ Actual shipment date of the product (transaction date)
→ Example values:
'2025-01-29', '2025-07-22', '2025-08-20', '2025-08-07', '2024-08-06'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00


- campus_zip (VARCHAR)
→ ZIP code of the campus account location
→ Example values:
'55433', '60050', '01702', '21740', '60612'


- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30008155', '30043877', '30006184', '30039854'


- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE MCHENRY HOSPITAL',
'CHARLES RIVER MEDICAL ASSOCIATES',
'ANTIETAM ONCOLOGY AND HEMATOLOGY GROUP PC',
'RUSH UNIVERSITY MEDICAL CENTER'


- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'11850 BLACKFOOT ST NW',
'4201 MEDICAL CENTER DR',
'233 W CENTRAL ST'


- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'COON RAPIDS', 'MCHENRY', 'NATICK', 'HAGERSTOWN', 'CHICAGO'


- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'MN', 'IL', 'MA', 'MD', 'ND'


- campus_tier (VARCHAR)
→ Tier classification of the campus campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'


- campus_account_type (VARCHAR)
→ Account archetype indicating whether the campus account belongs to an academic or community setting
→ Original values:
'ACADEMIC', 'COMMUNITY'


- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3


- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0405', 'FF0505', 'FF0101', 'FF0202', 'FF0201'


- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'Minneapolis, MN',
'Chicago, IL',
'Boston, MA',
'Baltimore, MD',
'South Jersey, NJ'


- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF04', 'FF05', 'FF01', 'FF02', 'FF03'


- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'Central', 'Great Lakes', 'North East', 'Mid Atlantic', 'South East'


- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30028542', '30014847', '30009531'


- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MINNESOTA ONCOLOGY',
'NORTHWESTERN MEDICINE',
'CHARLES RIVER MEDICAL ASSOCIATES',
'RUSH SYSTEM FOR HEALTH',
'HEART OF AMERICA MEDICAL CENTER'


- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'2550 UNIVERSITY AVE W',
'251 E HURON ST',
'233 W CENTRAL ST'


- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'SAINT PAUL', 'CHICAGO', 'NATICK'


- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'MN', 'IL', 'MA'


- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'55114', '60611', '01702'


- week_end_date (DATE)
→ Week ending date associated with the transaction date
→ Example values:
'31-01-2025', '25-07-2025', '22-08-2025', '08-08-2025', '09-08-2024'


- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Jan', '25-Jul', '25-Aug', '24-Aug', '26-Feb'


- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q1-25', 'Q3-25', 'Q3-24', 'Q1-26', 'Q4-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_DDD (Anchor this table for Competitor Level Information)

Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):


- date (DATE)
→ Actual shipment date of the product
→ Example values:
'2025-08-31', '2025-09-30', '2025-01-31', '2024-09-30', '2024-08-31'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- reblozyl_total_mg (INT) (Competitor)
→ Total milligrams of Reblozyl shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- reblozyl_total_sls (DECIMAL)
→ Total sales value of reblozyl for the transaction
→ Represents the monetary value associated with the shipped reblozyl quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- campus_zip (VARCHAR)
→ ZIP code of the campus account
→ Example values:
'92660', '92123', '60451', '11354', '1301'

- campus_id (VARCHAR)
→ Unique identifier of the campus account
→ Example values:
'30026581', '30005403', '30012146', '30000194', '30035056'

- campus_account_name (VARCHAR)
→ campus account or treatment center name
→ Example values:
'MEMORIALCARE CANCER INSTITUTE',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'UNIVERSITY OF CHICAGO CANCER CENTER AT SILVER CROSS',
'CHUNG YUNHEE OFFICE',
'BAYSTATE FRANKLIN MEDICAL CENTER'

- campus_address (VARCHAR)
→ Address of the campus account
→ Example values:
'1441 AVOCADO AVE',
'7930 FROST ST',
'1850 SILVER CROSS BLVD',
'14218 38TH AVE',
'164 HIGH ST'

- campus_city (VARCHAR)
→ City where the campus account is located
→ Example values:
'NEWPORT BEACH',
'SAN DIEGO',
'NEW LENOX',
'FLUSHING',
'GREENFIELD'

- campus_state (VARCHAR)
→ State where the campus account is located
→ Example values:
'CA', 'IL', 'NY', 'MA', 'FL'

- campus_tier (VARCHAR)
→ Tier classification of the campus campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'

- campus_account_type (VARCHAR)
→ Account archetype of the campus account
→ Original values:
'ACADEMIC', 'COMMUNITY'

- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3, 4

- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code
→ Example values:
'FF0605', 'FF0505', 'FF0105', 'FF0101', 'FF0304'

- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id
→ Example values:
'San Diego, CA',
'Chicago, IL',
'Long Island, NY',
'Boston, MA',
'Miami, FL'

- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory
→ Example values:
'FF06', 'FF05', 'FF01', 'FF03', 'FF04'

- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Original values:
'West', 'Great Lakes', 'North East', 'South East', 'Central'

- parent_id (VARCHAR)
→ Unique identifier of the parent account
→ Example values:
'30031537', '30054837', '30031634', '30031368', '30046707'

- parent_account_name (VARCHAR)
→ Parent account or health system name
→ Example values:
'MEMORIALCARE HEALTH SYSTEM',
'SAN DIEGO ONCOLOGY MEDICAL CLINIC',
'THE UNIVERSITY OF CHICAGO MEDICINE',
'BAYSTATE HEALTH',
'HEMATOLOGY ONCOLOGY ASSOCIATES OF THE PALM BEACHES'

- parent_address (VARCHAR)
→ Address of the parent account
→ Example values:
'17360 BROOKHURST ST',
'7930 FROST ST',
'5841 S MARYLAND AVE',
'280 CHESTNUT ST',
'3450 LANTANA RD'

- parent_city (VARCHAR)
→ City of the parent account
→ Example values:
'FOUNTAIN VALLEY', 'SAN DIEGO', 'CHICAGO', 'SPRINGFIELD', 'LAKE WORTH'

- parent_state (VARCHAR)
→ State of the parent account
→ Example values:
'CA', 'IL', 'MA', 'FL', 'MS'

- parent_zip (VARCHAR)
→ ZIP code of the parent account
→ Example values:
'92708', '92123', '60637', '1199', '33462'

- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Aug', '25-Sep', '25-Jan', '24-Sep', '24-Aug'

- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q3-25', 'Q1-25', 'Q3-24', 'Q4-24', 'Q2-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: targets

Important Notes:
- The table contains exactly one row.
- It acts like a dictionary / configuration table.
- Each column represents a target KPI value used for benchmarking or evaluation.
- The single row stores the global target values for the system.

- account_call_per_day (DECIMAL)
→ Target number of account calls to be made per day
→ Example values:
1.75, 2.00, 3.50

- hcp_touches (DECIMAL)
→ Target number of healthcare professional (HCP) interactions or touches
→ Example values:
2.0, 3.0, 4.5

- percentage_reach_tier_1_2_per_quarter (DECIMAL)
→ Target percentage of Tier 1 and Tier 2 accounts that should be reached in a quarter
→ Example values:
85, 90, 95

- percentage_reach_all_tier_per_quarter (DECIMAL)
→ Target percentage of all account tiers that should be reached in a quarter
→ Example values:
70, 75, 80

- frequency_tier_1_2_per_quarter (DECIMAL)
→ Target number of interactions with Tier 1 and Tier 2 accounts per quarter
→ Example values:
1, 1.5, 2

- frequency_all_tier_per_quarter (DECIMAL)
→ Target number of interactions with all account tiers per quarter
→ Example values:
1, 1.2, 1.5

- increase_depth_per_account (DECIMAL)
→ Target increase in engagement depth per account
→ Example values:
0.2, 0.3, 0.5

- average_number_of_active_patients (DECIMAL)
→ Target average number of active patients per account
→ Example values:
8, 10, 12


    ────────────────────────
    EXAMPLES (FOR GUIDANCE)
    ────────────────────────

    Example 1:
    User Question:
    "Total quantity sold by region in Q4-24"

    Expected SQL Output:
    SELECT campus_region, SUM(rytelo_total_mg) AS total_qty
    FROM data_867
    WHERE quarter_year = 'Q4-24'
    GROUP BY campus_region;

    ────────────────────────

    Example 2:
    User Question:
    "Top 5 campus accounts in Central by quantity sold"

    Expected SQL Output:
    SELECT parent_name, SUM(rytelo_total_mg) AS total_qty
    FROM data_867
    WHERE campus_region = 'Central'
    GROUP BY campus_account_name
    ORDER BY total_qty DESC
    LIMIT 5;

    """

    response = model.invoke(prompt).content
    print("SQL Generator Response")
    print(response)
    print("-"*100)
    state["sql_generator_output"] = response
    state["last_output"] = response

    log_trace(state, "SQL_Generator", "TextMessage", response)
    return state

def sql_reviewer_node(state: AgentState):
    user_input=state["question"]
    generated_sql=state["sql_generator_output"]
    query_decomposition=state["query_decomposer_output"]
    human_feedback = state.get("human_reviewer_output") or None
    # print("Human Feedback from SQL Reviewer")
    # print(human_feedback)
    prompt=f"""
You are an expert SQL reviewer for MySQL.

Your role is to VALIDATE correctness, safety, and logical consistency of a generated SQL query.
You are NOT a SQL generator.
You must understand analytical intent, including rolling windows and derived dates.

────────────────────────
OUTPUT RESTRICTION (MANDATORY)
────────────────────────

You must NEVER write, regenerate, or rewrite SQL (even partially).

You must NEVER propose an alternative SQL query.

If the SQL is wrong, only state the exact issue(s) causing rejection.

────────────────────────
WHAT YOU MUST CHECK
────────────────────────

Reject the SQL ONLY if one or more of the following are true:

❌ The query uses forbidden statements:

DELETE, UPDATE, INSERT, DROP, ALTER, TRUNCATE

❌ The query references:

Tables not listed in the schema

Columns not listed in the schema

❌ The SQL contains invalid MySQL syntax

❌ The SQL logic is internally inconsistent, such as:

Aggregates used without proper GROUP BY

WHERE filters applied to aggregated columns incorrectly

GROUP BY columns that do not align with SELECT columns

❌ The query violates explicit user intent
(e.g., user asks for “latest date” but query uses CURDATE without justification)

────────────────────────
WHAT IS EXPLICITLY ALLOWED
────────────────────────

You MUST allow the following patterns if used correctly:

✔ Common Table Expressions (WITH clauses)
✔ Subqueries in SELECT / WHERE / FROM
✔ Derived-date logic using:

MAX(date_column)

DATE_SUB / DATE_ADD
✔ Rolling window calculations (e.g., last 13 weeks)
✔ Aggregations (SUM, COUNT, AVG)
✔ ORDER BY and LIMIT
✔ Aliases
✔ Nested queries
✔ Filtering using derived values


Do NOT reject a query just because it is complex.

────────────────────────
IMPORTANT CLARIFICATIONS
────────────────────────

• Example values in the schema are ILLUSTRATIVE ONLY and must NEVER be used to reject SQL.
• “Original values” listed in the schema are NOT exhaustive and must NEVER be used to reject SQL.
• Do NOT validate whether literal filter values exist in the dataset (out of scope).

For example, do NOT reject because parent_eid or parent_name is different from shown examples.

• Queries using MAX(date_column) instead of system date
are PREFERRED for “latest / most recent” questions

• Rolling windows must be evaluated relative to the data
→ Using MAX(week_end_date) is VALID and CORRECT

• Subqueries and CTEs do NOT require rejection unless syntactically invalid

• Do NOT reject a query because it is not optimal or not written in the same style as examples.
Only reject for correctness, safety, schema mismatch, syntax errors, or explicit intent mismatch.

If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
month_year and quarter_year are columns present in both data_867 and data_DDD.
────────────────────────
INPUT CONTEXT
────────────────────────

Consider the current month as: {CURRENT_MONTH}
Consider the current quarter as: {CURRENT_QUARTER}

USER QUERY
────────────────────────
{user_input}
────────────────────────

Genrated SQL (IMPORTANT)
────────────────────────
{generated_sql}
────────────────────────
Query decomposition (for reference):
{query_decomposition}

Human feedback (if any):
{human_feedback}
If human feedback is provided, treat it as a strict constraint and prioritize it during evaluation.

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_867

Columns, meanings, Original Values (Only those values exist in the table) 
and example values (values other than that can exist in the table) 


- date (DATE)
→ Actual shipment date of the product (transaction date)
→ Example values:
'2025-01-29', '2025-07-22', '2025-08-20', '2025-08-07', '2024-08-06'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00


- campus_zip (VARCHAR)
→ ZIP code of the campus account location



- campus_id (VARCHAR)
→ Unique identifier of the campus account



- campus_account_name (VARCHAR)
→ campus account or treatment center name



- campus_address (VARCHAR)
→ Address of the campus account



- campus_city (VARCHAR)
→ City where the campus account is located



- campus_state (VARCHAR)
→ State where the campus account is located



- campus_tier (VARCHAR)
→ Tier classification of the campus campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'


- campus_account_type (VARCHAR)
→ Account archetype indicating whether the campus account belongs to an academic or community setting
→ Example values:
'ACADEMIC', 'COMMUNITY'


- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3


- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code



- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id



- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory



- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Example values:
'Central', 'Great Lakes', 'North East', 'Mid Atlantic', 'South East'


- parent_id (VARCHAR)
→ Unique identifier of the parent account



- parent_account_name (VARCHAR)
→ Parent account or health system name



- parent_address (VARCHAR)
→ Address of the parent account



- parent_city (VARCHAR)
→ City of the parent account



- parent_state (VARCHAR)
→ State of the parent account



- parent_zip (VARCHAR)
→ ZIP code of the parent account



- week_end_date (DATE)
→ Week ending date associated with the transaction date
→ Example values:
'31-01-2025', '25-07-2025', '22-08-2025', '08-08-2025', '09-08-2024'


- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Jan', '25-Jul', '25-Aug', '24-Aug', '26-Feb'


- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q1-25', 'Q3-25', 'Q3-24', 'Q1-26', 'Q4-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: data_DDD (Anchor this table for Competitor Level Information)

Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):


- date (DATE)
→ Actual shipment date of the product
→ Example values:
'2025-08-31', '2025-09-30', '2025-01-31', '2024-09-30', '2024-08-31'


- rytelo_total_mg (INT)
→ Total milligrams of Rytelo shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- reblozyl_total_mg (INT) (Competitor)
→ Total milligrams of Reblozyl shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- rytelo_total_sls (DECIMAL)
→ Total sales value of Rytelo for the transaction
→ Represents the monetary value associated with the shipped Rytelo quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- reblozyl_total_sls (DECIMAL)
→ Total sales value of reblozyl for the transaction
→ Represents the monetary value associated with the shipped reblozyl quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- campus_zip (VARCHAR)
→ ZIP code of the campus account


- campus_id (VARCHAR)
→ Unique identifier of the campus account


- campus_account_name (VARCHAR)
→ campus account or treatment center name


- campus_address (VARCHAR)
→ Address of the campus account


- campus_city (VARCHAR)
→ City where the campus account is located


- campus_state (VARCHAR)
→ State where the campus account is located


- campus_tier (VARCHAR)
→ Tier classification of the campus campus
→ Example values:
'Tier 1', 'Tier 2', 'Tier 3', 'No Tier'

- campus_account_type (VARCHAR)
→ Account archetype of the campus account
→ Example values:
'ACADEMIC', 'COMMUNITY'

- campus_calls (INT)
→ Number of sales calls or interactions logged for the account
→ Example values:
0, 1, 2, 3, 4

- campus_territory_id (VARCHAR)
→ Territory identifier corresponding to the ZIP code


- campus_territory (VARCHAR)
→ Territory name corresponding to campus_territory_id


- campus_region_id (VARCHAR)
→ Region identifier corresponding to the territory


- campus_region (VARCHAR)
→ Region name corresponding to campus_region_id
→ Example values:
'West', 'Great Lakes', 'North East', 'South East', 'Central'

- parent_id (VARCHAR)
→ Unique identifier of the parent account


- parent_account_name (VARCHAR)
→ Parent account or health system name


- parent_address (VARCHAR)
→ Address of the parent account


- parent_city (VARCHAR)
→ City of the parent account


- parent_state (VARCHAR)
→ State of the parent account


- parent_zip (VARCHAR)
→ ZIP code of the parent account


- month_year (VARCHAR)
→ Month and year label corresponding to the transaction date
→ Example values:
'25-Aug', '25-Sep', '25-Jan', '24-Sep', '24-Aug'

- quarter_year (VARCHAR)
→ Quarter and year label corresponding to the transaction date
→ Example values:
'Q3-25', 'Q1-25', 'Q3-24', 'Q4-24', 'Q2-25'

────────────────────────
TABLE INFORMATION
────────────────────────
Table name: targets

Important Notes:
- The table contains exactly one row.
- It acts like a dictionary / configuration table.
- Each column represents a target KPI value used for benchmarking or evaluation.
- The single row stores the global target values for the system.

- account_call_per_day (DECIMAL)
→ Target number of account calls to be made per day
→ Example values:
1.75, 2.00, 3.50

- hcp_touches (DECIMAL)
→ Target number of healthcare professional (HCP) interactions or touches
→ Example values:
2.0, 3.0, 4.5

- percentage_reach_tier_1_2_per_quarter (DECIMAL)
→ Target percentage of Tier 1 and Tier 2 accounts that should be reached in a quarter
→ Example values:
85, 90, 95

- percentage_reach_all_tier_per_quarter (DECIMAL)
→ Target percentage of all account tiers that should be reached in a quarter
→ Example values:
70, 75, 80

- frequency_tier_1_2_per_quarter (DECIMAL)
→ Target number of interactions with Tier 1 and Tier 2 accounts per quarter
→ Example values:
1, 1.5, 2

- frequency_all_tier_per_quarter (DECIMAL)
→ Target number of interactions with all account tiers per quarter
→ Example values:
1, 1.2, 1.5

- increase_depth_per_account (DECIMAL)
→ Target increase in engagement depth per account
→ Example values:
0.2, 0.3, 0.5

- average_number_of_active_patients (DECIMAL)
→ Target average number of active patients per account
→ Example values:
8, 10, 12

────────────────────────
RESPONSE FORMAT (STRICT)
────────────────────────

Respond ONLY in this format, with no extra text:

PASS or REJECT
FEEDBACK:

If REJECT: list the exact technical or logical issues

If PASS: say exactly → "PASS, SQL is safe and valid"

Do NOT provide suggestions, rewrites, or explanations.
Do NOT output SQL.
Do NOT reject queries that correctly implement analytical intent.

"""
    response=model.invoke(prompt).content
    print("SQL Reviewer Output")
    print(response)
    
    decision = parse_review_output(response, "sql_reviewer")
    # print("SQL Reviewer Decision")
    # print(decision)
    print("-"*100)
    state["active_review"] = decision  # 🔑 anchor here

    log_trace(state, "sql_reviewer", "TextMessage", response)
    state["last_output"] = response
    state["sql_reviewer_output"]=response
    return state

def sql_executor(state: AgentState):
    sql_generator_output=state["sql_generator_output"]
    result_df = run_mysql_query(sql_generator_output)
    print("Query Result:")
    print(result_df)
    serialized_df = {
        "columns": result_df.columns.tolist(),
        "data": result_df.to_dict(orient="records")
    }
    summary = f"Query executed successfully. Rows returned: {len(result_df)}"
    log_trace(state, "sql_executor", "TextMessage", summary)
    state["sql_executor_output"] = serialized_df
    state["last_output"]=summary
    return state

def human_node(state: AgentState):
    result=interrupt({"Decision": "Reject or Accept the query, if rejected give the feedback"})
    print("Human Reviewer Output")
    print(result["feedback"])
    print("-"*100)
    if result["feedback"].startswith("R"):
        log_trace(
        state,
        agent="human_reviewer",
        event_type="TextMessage",
        text=result["feedback"]
    )
        state["last_output"]=result["feedback"]
        decision = parse_review_output(result["feedback"], source="human")
        state["active_review"] = decision
        state["human_reviewer_output"]=result["feedback"]

    else:
        state["last_output"]=result["feedback"]
        decision = parse_review_output(result["feedback"], source="human")
        state["active_review"] = None
        log_trace(state, "human_reviewer", "TextMessage", result["feedback"])
        state["human_reviewer_output"]=result["feedback"]
    # Trace for audit/debug
    return state

def terminator_node(state: AgentState):
    state["last_output"] = "TERMINATE"
    return state

def reviewer_router(state: AgentState):
    output = state["last_output"].upper()
    if "PASS" in output:
        return "sql_executor"
    return "query_decomposer"
def human_router(state: AgentState):
    output = state["last_output"].upper()

    approve_keywords = ["SUCCESS","APPROVE", "LOOKS GOOD", "TERMINATE", "YES", "OK", "GOOD", "PASS"]
    reject_keywords = ["REJECT", "CHANGE", "FIX", "MODIFY", "WRONG", "INCORRECT", "NO"]

    if any(k in output for k in approve_keywords):
        return "terminator"
    if any(k in output for k in reject_keywords):
        return "query_decomposer"

    # default safe loop
    return "query_decomposer"



builder = StateGraph(AgentState)

# Nodes
builder.add_node("query_decomposer", query_decomposer_node)
builder.add_node("sql_generator", sql_generator_node)
#builder.add_node("sql_reviewer", sql_reviewer_node)
builder.add_node("sql_executor",sql_executor)
builder.add_node("human", human_node)
builder.add_node("terminator", terminator_node)

# Entry
builder.set_entry_point("query_decomposer")

# Edges
builder.add_edge("query_decomposer", "sql_generator")
#builder.add_edge("sql_generator", "sql_reviewer")
builder.add_edge("sql_generator", "sql_executor")
# Conditional edges
# builder.add_conditional_edges(
#     "sql_reviewer",
#     reviewer_router,
#     {
#         "sql_executor": "sql_executor",
#         "query_decomposer": "query_decomposer",
#     },
# )
builder.add_edge("sql_executor", "human")
builder.add_conditional_edges(
    "human",
    human_router,
    {
        "terminator": "terminator",
        "query_decomposer": "query_decomposer",
    },
)

# Terminator → END
builder.add_edge("terminator", END)

checkpointer=MemorySaver()

graph = builder.compile(
    checkpointer=checkpointer  
)

if __name__=="__main__":
    config={"configurable":{"thread_id":"1234"}}
    user_input=input("Enter your Query: ")
    initial_state = {
    "last_output": "",
    "query_decomposer_output": None,
    "sql_generator_output": None,
    "sql_reviewer_output": None,
    "human_reviewer_output": None,
    "active_review": None,
    "sql_executor_output": None,
    "trace": [],
    "question": user_input,
    "run_id": datetime.now(UTC).isoformat() + "Z"
}
   
    result = graph.invoke(initial_state, config=config)

    while True:
        interrupts = result.get("__interrupt__", [])

        if not interrupts:
            # No interrupt → graph finished
            break

        prompt_to_human = interrupts[0].value
        print(f"HITL: {prompt_to_human}")

        decision = input("Your Decision: ")

        # Resume graph with human feedback
        result = graph.invoke(
            Command(resume={"feedback": decision}),
            config=config
        )

    # Final result after approval
    #print(result)
    append_agent_trace("agent_trace_1.json", user_input, result["trace"])



