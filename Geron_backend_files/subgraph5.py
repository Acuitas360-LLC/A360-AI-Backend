from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json
import os
from datetime import datetime, UTC
from typing import TypedDict, Literal, Optional, List
import re
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import warnings
warnings.filterwarnings("ignore")
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
import mysql.connector
import pandas as pd
import snowflake.connector

TRACE_FILE = "agent_trace.json"
# =========================
# MySQL connection config
# =========================
DB_CONFIG = {
    "host": "localhost",
    "port": "3306",
    "user": "root",
    "password": "root@1234",
    "database": "drug_db"
}

model=ChatOpenAI(model='gpt-5.2')
model_1=ChatOpenAI(model='gpt-5-mini')
model_2=ChatOpenAI(model='gpt-5-nano')
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


def run_snowflake_query(query):
    conn = snowflake.connector.connect(
        user="ahusain",
        password="Murtaza@40401059",
        account="ua60309.south-central-us.azure",
        warehouse="RELMORA_COMPUTE",
        database="RELMORA_DB",
        schema="RELMORA_SCHEMA"
    )

    cursor = conn.cursor()
    cursor.execute(query)

    # Fetch data
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    cursor.close()
    conn.close()

    return df


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

def get_recent_messages(messages: list, n: int = 10):
    return messages[-n:] if len(messages) > n else messages

def get_clean_recent_turns(messages: List, n_turns: int = 3):
    """
    Returns last n conversation turns in compact form.

    Keeps:
    - HumanMessage (full)
    - AIMessage with only:
        * SQL Query Executed
        * Result Summary

    Removes:
    - Query Results
    - Visualization code
    - Relevant Questions
    """

    turns = []
    current_turn = []

    # -------- Step 1: Build turns (reverse traversal) --------
    for msg in reversed(messages):
        current_turn.insert(0, msg)

        if isinstance(msg, HumanMessage):
            turns.insert(0, current_turn)
            current_turn = []

            if len(turns) == n_turns:
                break

    # -------- Step 2: Clean AI messages --------
    cleaned_messages = []

    for turn in turns:
        for msg in turn:

            # Keep human messages fully
            if isinstance(msg, HumanMessage):
                cleaned_messages.append(msg)

            elif isinstance(msg, AIMessage):
                content = msg.content or ""

                sql_part = ""
                summary_part = ""

                # Extract SQL
                if "SQL Query Executed:" in content:
                    sql_part = content.split("SQL Query Executed:")[-1]

                    # Stop at next section
                    for stop in ["Result Summary:", "Relevant Questions:", "Query Results:", "Visualization Code:"]:
                        if stop in sql_part:
                            sql_part = sql_part.split(stop)[0]
                            break

                    sql_part = sql_part.strip()

                # Extract Summary
                if "Result Summary:" in content:
                    summary_part = content.split("Result Summary:")[-1]

                    for stop in ["Relevant Questions:", "Query Results:", "Visualization Code:"]:
                        if stop in summary_part:
                            summary_part = summary_part.split(stop)[0]
                            break

                    summary_part = summary_part.strip()

                # Build cleaned AI message if something exists
                cleaned_content = ""

                if sql_part:
                    cleaned_content += "SQL Query Executed:\n" + sql_part + "\n\n"

                if summary_part:
                    cleaned_content += "Result Summary:\n" + summary_part

                if cleaned_content:
                    cleaned_messages.append(
                        AIMessage(content=cleaned_content.strip())
                    )

    return cleaned_messages



class AgentState(TypedDict):
    # inputs
    question: str
    messages: List[BaseMessage]
    # agent outputs
    query_decomposer_output: str | None
    sql_generator_output: str | None
    sql_reviewer_output: str | None
    human_reviewer_output: str | None
    query_decomposer_rag_examples_text: str | None
    sql_generator_rag_examples_text: str | None
    result_summary: str | None
    visualization_code:str | None
    visualization_spec:str | None
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

def build_messages(state,SYSTEM_PROMPT):
    recent_messages = get_clean_recent_turns(state["messages"])
    print("Recent Messages")
    print(recent_messages)
    print("-"*100)

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *recent_messages
    ]

def query_decomposer_node(state: AgentState):
    review = state["active_review"]
    user_input=state["question"]
    messages=state['messages']
    query_decomposer_rag_examples_text=state['query_decomposer_rag_examples_text']
    

    last_human_message = next(
    msg for msg in reversed(messages) if isinstance(msg, HumanMessage)
)
    user_input=last_human_message.content
    print("Last Human Message")
    print(user_input)
    print("-"*100)
    if review and (review["decision"] == "REJECT"):

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
    Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on relmora, use relmora_total_mg.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    campus accounts are also referred to as child.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
    When weekly averages are used (because at least one period is incomplete), the output must also include the week_end_date values of the weeks that were included in the weekly average calculation.(VERY IMPORTANT)
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

SUM(relmora_total_mg)

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
Table name: data_867 (Anchor this table for relmora sales related information)

Columns, meanings, Original Values (Only those values exist in the table) 
and example values (values other than that can exist in the table) 


- date (DATE)
→ Actual shipment date of the product (transaction date)
→ Example values:
'2025-01-29', '2025-07-22', '2025-08-20', '2025-08-07', '2024-08-06'


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- zynava_total_mg (INT) (Competitor)
→ Total milligrams of zynava shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- zynava_total_sls (DECIMAL)
→ Total sales value of zynava for the transaction
→ Represents the monetary value associated with the shipped zynava quantity
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
    "validation_rules": [string],
    "rag_alignment": {{
    "rag_provided": boolean,
    "used_examples": [string],
    "borrowed_patterns": [string],
    "differences_from_examples": [string]
         }}
    }}


   {query_decomposer_rag_examples_text}

    ────────────────────────
    FINAL REMINDER
    ────────────────────────
    - Output ONLY valid JSON
    - Follow the required structure exactly
    - Do NOT output SQL, markdown, or explanations
    """

    else :

        prompt=f"""You are a Query Decomposer agent.

Your responsibility is to analyze conversational natural-language input and convert it into a structured, deterministic JSON specification that describes HOW a SQL query should be constructed by a downstream SQL Generator.

You must NOT generate SQL.
You must NOT generate pseudo-SQL.
You must describe intent, logic, filters, aggregations, grouping, ordering, subqueries, and validation rules in structured JSON.

The SQL Generator will rely entirely on your JSON output.

────────────────────────
INPUT HANDLING
────────────────────────
You will receive recent conversation context and the latest Human message.

IMPORTANT:
- The latest Human message is the PRIMARY source of intent.
- Prior context is for reference only and must be used to:
  - Preserve correct previously established logic
  - Resolve references (e.g., “same as before”, “change this”)
- If there is any conflict, the latest Human message overrides prior intent.
- If the latest message requests a modification, apply ONLY the requested changes.
- If the latest message restates the request, treat it as a full replacement.

────────────────────────
INPUT
────────────────────────
USER QUERY (LATEST HUMAN MESSAGE)
{user_input}

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
    Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on relmora, use relmora_total_mg.
    If the user does not specify a time window, default to 52 weeks.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    campus accounts are also referred to as child.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
The dataset contains the following time fields derived from the transaction date: **date**, **week_end_date** (week ending Friday), **month_year**, and **quarter_year**.
When weekly averages are used (because at least one period is incomplete), the output must also include the week_end_date values of the weeks that were included in the weekly average calculation.(VERY IMPORTANT)
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

SUM(relmora_total_mg)

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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- zynava_total_mg (INT) (Competitor)
→ Total milligrams of zynava shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- zynava_total_sls (DECIMAL)
→ Total sales value of zynava for the transaction
→ Represents the monetary value associated with the shipped zynava quantity
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
    "validation_rules": [string],
    "rag_alignment": {{
    "rag_provided": boolean,
    "used_examples": [string],
    "borrowed_patterns": [string],
    "differences_from_examples": [string]
         }}
    }}
  
   {query_decomposer_rag_examples_text}

    ────────────────────────
    FINAL REMINDER
    ────────────────────────
    - Output ONLY valid JSON
    - Follow the required structure exactly
    - Do NOT output SQL, markdown, or explanations
    """
    final_prompt = build_messages(state, prompt)
    # print("Final Prompt")
    # print(final_prompt)
    print("-"*100)
    result=model.invoke(final_prompt).content
    print("Query Decomposer Output")
    print("-"*100)
    print(result)
    # state['query_decomposer_output']=result
    # state["last_output"] = result
    # state["active_review"] = None
    log_trace(state, "query_decomposer", "TextMessage", result)   

    return {
        "query_decomposer_output":result,
        "last_output":result,
        "active_review":None
    }

def sql_generator_node(state):
    user_input=state["question"]
    query_decomposer_output=state['query_decomposer_output']
    sql_generator_rag_examples_text=state['sql_generator_rag_examples_text']
    prompt = f"""
You are an expert Snowflake SQL Generator.

Your responsibility is to generate a valid Snowflake SELECT query based STRICTLY on the structured JSON produced by the Query Decomposer.

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

You must translate this JSON into executable Snowflake SQL.

────────────────────────
STRICT RULES (MANDATORY)
────────────────────────
- Generate ONLY SELECT queries
- NEVER use DELETE, UPDATE, INSERT, DROP, ALTER, or TRUNCATE
- Use ONLY tables and columns explicitly present in the schema
- Use valid Snowflake SQL syntax
- Do NOT hallucinate columns, tables, or joins
- Do NOT add logic not present in the Query Decomposer JSON
- Do NOT explain the query
- Do NOT output markdown or commentary
- Output ONLY the SQL query

- All non-aggregated columns in SELECT must be explicitly included in GROUP BY

- Ensure all computed division denominators use NULLIF(column, 0) to prevent division-by-zero errors

- All percentage outputs must use ROUND() and be formatted using CONCAT(value, '%')

- If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level (VERY IMPORTANT)

- Follow structured logic: identify columns → filter → group → aggregate → sort/rank

- Combine related calculations into one cohesive query

- Keep queries readable using clear aliases

- Return only relevant, well-labeled results

────────────────────────
SNOWFLAKE-SPECIFIC RULES (MANDATORY)
────────────────────────
- Use DATEADD() for all date arithmetic
  Example: DATEADD(WEEK, -12, date_column)

- NEVER use DATE_SUB or INTERVAL

- NEVER use backticks (`); use double quotes "alias" when needed

- Use CAST(... AS INTEGER) or ::INTEGER instead of SIGNED

- Use CASE WHEN instead of IF()

- Use CONCAT() for string concatenation

- Avoid MySQL-specific functions

- Use CURRENT_DATE instead of CURDATE()

- Ensure type safety in numeric operations

- Avoid implicit casting

- Ensure CROSS JOIN does not introduce unintended duplication

- Keep date window logic consistent and explicit

STRICT SQL RULES:
1. Every column in SELECT that is NOT inside an aggregate function MUST be present in the GROUP BY clause.
2. NEVER include columns in SELECT that are not grouped or aggregated.
3. When using aliases (e.g., W.column), ensure the same alias is used consistently in SELECT and GROUP BY.
4. Do NOT use implicit grouping — Snowflake requires explicit GROUP BY.
5. If a column is constant (e.g., from a CTE), still include it in GROUP BY if selected.
6. Prefer explicit GROUP BY column names over positional indexes.

AGGREGATION RULES:
7. If aggregation is used (COUNT, SUM, AVG, etc.), verify ALL non-aggregated fields are grouped.
8. Avoid mixing aggregated and non-aggregated columns incorrectly.

VALIDATION BEFORE OUTPUT:
9. Double-check that the query will not produce:
   - "not a valid group by expression"
   - "column not in group by"
   - ambiguous column errors

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
Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share or competitor comparisons. For queries focused solely on relmora, use relmora_total_mg.
If the user does not specify a time window, default to 52 weeks.
If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
Campus accounts are also referred to as child.
Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
If asked about sales by default give national sales don't group by campus_id or campus_account name.
When weekly averages are used (because at least one period is incomplete), the output must also include the week_end_date values of the weeks that were included in the weekly average calculation. (VERY IMPORTANT)
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

SUM(relmora_total_mg)

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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- zynava_total_mg (INT) (Competitor)
→ Total milligrams of zynava shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- zynava_total_sls (DECIMAL)
→ Total sales value of zynava for the transaction
→ Represents the monetary value associated with the shipped zynava quantity
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

    {sql_generator_rag_examples_text}

    """

    response = model.invoke(prompt).content
    print("SQL Generator Response")
    print(response)
    # state["sql_generator_output"] = response
    # state["last_output"] = response

    log_trace(state, "SQL_Generator", "TextMessage", response)
    return {
        "sql_generator_output":response,
        "last_output":response
    }

def sql_reviewer_node(state: AgentState):
    user_input=state["question"]
    generated_sql=state["sql_generator_output"]
    query_decomposition=state["query_decomposer_output"]
    human_feedback = state.get("human_reviewer_output") or None
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

If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)

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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 846, 564, 611, 47

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
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


- relmora_total_mg (INT)
→ Total milligrams of relmora shipped in the transaction
→ Example values:
0, 1034, 376, 658, 1692


- zynava_total_mg (INT) (Competitor)
→ Total milligrams of zynava shipped in the transaction
→ Example values:
0, 275, 350, 475, 425

- relmora_total_sls (DECIMAL)
→ Total sales value of relmora for the transaction
→ Represents the monetary value associated with the shipped relmora quantity
→ Example values:
0.00, 62457.48, 57252.72, 15270.78, 266868.00

- zynava_total_sls (DECIMAL)
→ Total sales value of zynava for the transaction
→ Represents the monetary value associated with the shipped zynava quantity
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
    print("-"*100)
    print(response)
    print("-"*100)
    decision = parse_review_output(response, "sql_reviewer")
    print("SQL Reviewer Decision")
    print(decision)
    #state["active_review"] = decision  # 🔑 anchor here

    log_trace(state, "sql_reviewer", "TextMessage", response)
    #state["last_output"] = response
    return {
        "sql_reviewer_output":response,
        "active_review":decision,
        "last_output":response
    }

def human_node(state: AgentState):
    result=interrupt({"Decision": "Reject or Accept the query, if rejected give the feedback"})
    if result["feedback"].startswith("R"):
        log_trace(
        state,
        agent="human_reviewer",
        event_type="TextMessage",
        text=result["feedback"]
    )
        #state["last_output"]=result["feedback"]
        decision = parse_review_output(result["feedback"], source="human")
        #state["active_review"] = decision
        return {
            "human_reviewer_output":result["feedback"],
            "last_output":result["feedback"],
            "active_review":decision
        }

    else:
        state["last_output"]=result["feedback"]
        decision = parse_review_output(result["feedback"], source="human")
        state["active_review"] = None
        log_trace(state, "human_reviewer", "TextMessage", result["feedback"])
    # Trace for audit/debug
    return {
        "human_reviewer_output":result["feedback"],
        "last_output":result["feedback"],
        "active_review":None
    }

def terminator_node(state: AgentState):
    state["last_output"] = "TERMINATE"
    append_agent_trace(
        file_path="agent_trace_2.json",
        question=state["question"],
        agent_trace=state["trace"]
    )
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

def sql_executor(state: AgentState):
    sql_generator_output=state["sql_generator_output"]
    result_df = run_snowflake_query(sql_generator_output)
    print("Query Result:")
    print(result_df)
    serialized_df = {
        "columns": result_df.columns.tolist(),
        "data": result_df.to_dict(orient="records")
    }
    summary = f"Query executed successfully. Rows returned: {len(result_df)}"
    
    return {
    "sql_executor_output": serialized_df,
    "last_output":summary
    }

def summarizer_node(state: AgentState):
    query_decomposer_output=state["query_decomposer_output"]
    sql_generator_output=state["sql_generator_output"]
    sql_executor_output=state["sql_executor_output"]
    prompt=f"""
        You are a Business Analyst summarizing analytical query results for a non-technical stakeholder.

        Your task is to explain what the results mean in business terms, focusing on impact, trends, and implications rather than technical implementation details.

        You are provided with the following information:

        Query Decomposer Output:
        {query_decomposer_output}

        SQL Generator Output (final SQL that was executed):
        {sql_generator_output}

        SQL Executor Output (summarized results, not raw data):
        {sql_executor_output}

        Your responsibilities:
        - Explain the business question that was addressed, based on how the query was interpreted.
        - Describe at a high level what data was analyzed and the scope of the analysis (time period, entities, key filters), without discussing SQL syntax.
        - Summarize the results in terms of business outcomes, key figures, and observable trends.
        - Highlight what stands out in the results and why it matters from a business perspective.
        - If the result set is empty, clearly state that no relevant activity or records were found for the given criteria.
        - If numerical values are present, focus on the most decision-relevant numbers rather than listing all values.
        - Avoid speculation or recommendations unless they are directly supported by the results.
        - Do NOT restate raw SQL or technical execution details.
        - Do NOT mention internal agents, system steps, prompts, or implementation details.

        Tone and style:
        - Business-focused, clear, and confident
        - Written as if explaining findings in a meeting or report
        - Plain English, minimal jargon
        - No emojis
        - No references to yourself or the system

        Output format:
        - A concise business summary (3–6 sentences)
        - Emphasize meaning and implications over mechanics


    """
    response=model.invoke(prompt).content
    
    return {
        "result_summary":response,
        "last_output":response
    }

def visualization_node(state: AgentState):
    query_decomposer_output=state["query_decomposer_output"]
    
    sql_executor_output=state["sql_executor_output"]
    #result_summary=state["result_summary"]
    prompt=f"""
    You are a Visualization Agent.

    Your goal is to create a useful Plotly visualization whenever possible.
    You should generate a chart in nearly all situations.
    Return NO_VISUALIZATION only in true edge cases.

    Inputs

    Query Decomposer Output:
    {query_decomposer_output}

    SQL Executor Output:
    {sql_executor_output}

    Assume the SQL output will be reconstructed into a Pandas DataFrame named df.

    Responsibilities

    Choose the most meaningful chart for understanding patterns, comparisons, trends, rankings, or distributions

    Align the chart with the analytical intent

    Use only columns present in the data

    Always try to visualize even small tables (2–5 rows)

    Growth Rule (VERY IMPORTANT)

    If any growth metric exists (growth, percent change, WoW/MoM/QoQ/YoY, increase/decrease), the chart must include both the base metric and the growth metric in the same visualization.

    Rules

    Use Plotly only (plotly.express or plotly.graph_objects)

    Output only valid Python code that defines fig

    No Streamlit, no explanations, no comments, no markdown

    Do not invent or infer missing columns

    If using secondary_y=True, the figure must be created using make_subplots with secondary_y enabled. Do not use secondary_y with go.Figure(). Never mix manual yaxis='y2' assignment with secondary_y. Ensure one consistent axis strategy.
    
    PLOTLY FORMATTING RULES (MANDATORY)

    - NEVER use Python `%` string formatting inside `hovertemplate`
    - ALWAYS use f-strings for dynamic values
    - Preserve Plotly placeholders like `%{{x}}`, `%{{y}}` exactly
    - When using f-strings, escape Plotly placeholders with double curly braces:
    Example: f"Region=%{{x}}<br>Value=%{{y}}"
    - Do NOT mix `%` formatting and Plotly placeholders in the same string

    Output

    Return either:

    Python code defining fig

    NO_VISUALIZATION (only when visualization is impossible)

"""
    response=model.invoke(prompt).content
    print("Visualization Code")
    print("-"*100)
    print(response)
    log_trace(state, "visualization_node", "TextMessage", response)
    return {
        "visualization_code":response
    }


def visualization_spec_node(state: AgentState):
    query_decomposer_output = state["query_decomposer_output"]
    sql_executor_output = state["sql_executor_output"]
    visualization_code = state.get("visualization_code")

    prompt = f"""
    You are a Visualization Spec Agent for a Next.js frontend.

        Return only a strict JSON object that helps frontend chart rendering.
        Do not return markdown. Do not return explanations.

        Inputs

        Query Decomposer Output:
        {query_decomposer_output}

        SQL Executor Output:
        {sql_executor_output}

        Plotly Visualization Code (if available):
        {visualization_code}

        Rules
        - Use only fields that exist in SQL Executor Output columns.
        - Never invent columns.
        - Prefer a single-series chart unless multi-series is required by user intent.
        - If growth-related field exists, include base and growth in same chart spec.
        - If Plotly code is provided and uses secondary_y, multiple traces, or annotations, reflect that complexity in the spec.
        - If chart is not useful, set chart_required to false.

        Required output JSON schema:
        {{
            "chart_required": true,
            "reason_code": null,
            "chart_intent": "trend|comparison|composition|distribution|relationship",
            "complexity": "single_series|multi_series|combo_dual_axis",
            "x": {{
                "field": "",
                "type": "date|category|numeric",
                "sort": "asc|desc|none"
            }},
            "series": [
                {{
                    "name": "",
                    "field": "",
                    "mark": "line|area|bar|scatter",
                    "axis": "left|right",
                    "stack": false
                }}
            ],
            "group_by_field": null,
            "transforms": {{
                "top_n": null,
                "others_bucket": false,
                "normalize": "none|percent"
            }},
            "interaction": {{
                "tooltip_fields": [],
                "show_points": true
            }},
            "formatting": {{
                "x_date_format": "YYYY-MM-DD",
                "y_format": "number|currency|percent",
                "tick_rotation": -35,
                "x_title": "",
                "y_left_title": "",
                "y_right_title": "",
                "legend_position": "top|bottom"
            }},
            "annotations": [
                {{
                    "type": "hline",
                    "axis": "left|right",
                    "value": 0,
                    "label": ""
                }}
            ],
            "renderer_hints": {{
                "preferred_renderer": "recharts|echarts",
                "recharts_type": "AreaChart|LineChart|BarChart|ComposedChart",
                "echarts_type": "line|bar|scatter|mixed"
            }},
            "validation": {{
                "used_columns": [],
                "missing_columns": [],
                "notes": []
            }}
        }}

        If chart is impossible:
        - set chart_required=false
        - set reason_code to one of: insufficient_numeric_data, single_scalar_result, empty_result, unsupported_shape
        - series should be empty array
        """

    response = model.invoke(prompt).content
    print("Visualization Spec")
    print("-" * 100)
    print(response)
    log_trace(state, "visualization_spec_node", "TextMessage", response)
    return {
        "visualization_spec": response,
    }



def build_graph(checkpointer=None):
    """
    Builds and returns a compiled LangGraph graph.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("query_decomposer", query_decomposer_node)
    builder.add_node("sql_generator", sql_generator_node)
    #builder.add_node("sql_reviewer", sql_reviewer_node)
    builder.add_node("sql_executor",sql_executor)
    builder.add_node("summarizer_node",summarizer_node)
    builder.add_node("visualization_node",visualization_node)
    #builder.add_node("human", human_node)
    builder.add_node("terminator", terminator_node)

    # Entry
    builder.set_entry_point("query_decomposer")

    # Edges
    builder.add_edge("query_decomposer", "sql_generator")
    #builder.add_edge("sql_generator", "sql_reviewer")
    builder.add_edge("sql_generator", "sql_executor")
    # Conditional edges
#     builder.add_conditional_edges(
#     "sql_reviewer",
#     reviewer_router,
#     {
#         "sql_executor": "sql_executor",
#         "query_decomposer": "query_decomposer",
#     },
# )

    # builder.add_conditional_edges(
    #     "human",
    #     human_router,
    #     {
    #         "terminator": "terminator",
    #         "query_decomposer": "query_decomposer",
    #     },
    # )

    # END
    #builder.add_edge("sql_reviewer","sql_executor")
    builder.add_edge("sql_executor","summarizer_node")
    builder.add_edge("sql_executor","visualization_node")
    builder.add_edge("summarizer_node","terminator")
    builder.add_edge("visualization_node","terminator")
    builder.add_edge("terminator", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    return graph

if __name__=="__main__":
    config={"configurable":{"thread_id":"12345"}}
    user_input=input("Enter your Query: ")
    initial_state = {
    "last_output": "",
    "query_decomposer_output": None,
    "sql_generator_output": None,
    "sql_reviewer_output": None,
    "human_reviewer_output": None,
    "visualization_spec": None,
    "active_review": None,
    "trace": [],
    "question": user_input,
    "run_id": datetime.now(UTC).isoformat() + "Z"
}
    graph=build_graph()
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
    print(result)
    append_agent_trace("agent_trace_2.json", user_input, result["trace"])



