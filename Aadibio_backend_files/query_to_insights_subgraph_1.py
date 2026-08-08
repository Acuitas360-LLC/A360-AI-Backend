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
import warnings
import snowflake.connector
warnings.filterwarnings("ignore")

TRACE_FILE = "agent_trace.json"

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": os.getenv("MYSQL_PORT", "3306"),
    "user": os.getenv("MYSQL_USER", ""),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "")
}

model=ChatOpenAI(model='gpt-5.4')
model_1=ChatOpenAI(model='gpt-5.3-codex')
model_2=ChatOpenAI(model='gpt-5')
# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

# def run_mysql_query(query: str) -> pd.DataFrame:
#     conn = None
#     try:
#         conn = mysql.connector.connect(**DB_CONFIG)

#         cursor = conn.cursor(dictionary=True)
#         cursor.execute(query)

#         rows = cursor.fetchall()
#         df = pd.DataFrame(rows)

#         return df

#     finally:
#         if conn:
#             conn.close()

def run_snowflake_query(query):
    required_snowflake_env = [
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]
    missing_env = [key for key in required_snowflake_env if not os.getenv(key)]
    if missing_env:
        raise ValueError(
            f"Missing Snowflake environment variables: {', '.join(missing_env)}"
        )

    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
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
    - If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)

   ────────────────────────
    Metric & Output Handling Rules (Must Always Be Enforced):
    ────────────────────────
    data_867 Rules:
        For data_867: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867 is the commercial demand table
         
    


    data_867_pap Rules:
        For data_867_pap: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867_pap contains records for both PAP and Commercial transactions in a single table, differentiated by the transaction_type column:
            transaction_type = 'PAP' → record belongs to PAP
            transaction_type = 'COM' → record belongs to Commercial

    forecast Rules:
        The budget column in data_867_forecast represents the monthly demand forecast value for that particular month_year — i.e., it is the forecasted/budgeted demand quantity, not an actual/realized value.
        MTD Forecast = (budget value for the current month_year / number_of_business_days_month for the current month_year) × business_days_elapsed_so_far for the current month_year. Report the answer to the nearest integer
        QTD/YTD Forecast/Budget = SUM(budget) for all month_year rows with quarter_year/year matching the current period and month_year < current month_year, plus the MTD Forecast for the current month_year. Report the answer to the nearest integer

    Revenue Table Rules:
        The revenue table contains shipment-level data. Consisting of net sales and gross sales.
        For any query related to revenue, this table must be treated as the anchor/source of truth. All revenue responses should be derived from this table rather than any other source.
        Revenue and net sales are synonyms

    revenue_forecast Rules:
        Anchor this table for revenue/ sales forecast
        The net_sales_forecast and gross_sales_forecast columns in revenue_forecast represent the daily demand forecast value for that particular date — i.e., they are the forecasted/budgeted values for net_sales and gross_sales respectively, not actual/realized values.

    parent_marketing_target Rules:
        "Target accounts" and "Top 75 accounts" queries anchor to the parent marketing target table.
        Account classification and bucket are synonyms — always treat them as the same field in parent_marketing_target table.

    mtor_alerts Rules:
        Route mTOR Patient Therapy Alert queries — patient-level alert counts by region, therapy initiation dates, unique therapies administered, and ongoing therapy account counts — to the mtor_alerts table.
        MTOR alerts table queries default to YTD (January 1 of the current year through the current date).
        Any MTOR alerts query defaults to YTD: Jan 1 of the CURRENT calendar year through the latest available date.

    Cross Table Rules (data_867 + data_867_pap):
        To calculate demand, always anchor to the qty_sold column and filter records where valid_orders = 1. Compute total demand as SUM(qty_sold) from data_867 plus SUM(qty_sold) from data_867_pap.
        If a query does not explicitly specify whether to calculate Commercial demand only or Commercial + PAP demand, then by default, only Commercial demand should be shown (i.e., filter using transaction_type = 'COM' for data_pap and complete data of data_867 for that timeperiod).
        If the query explicitly asks for PAP (or "Commercial + PAP"), then both transaction_type = 'COM' and transaction_type = 'PAP' records should be included from data_867_pap — i.e., do not filter by transaction_type at all, or use transaction_type IN ('COM', 'PAP')
        Breadth = # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Depth = demand ÷ # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Always use these exact formulas.
        Every account-level query is Commercial + PAP by default. Never scope to Commercial only unless the user explicitly asks for it.
        **Ordering account:**
            A child account that has at least one record in the window where BOTH
            conditions hold on the SAME record:
            - valid_order = 1
            - qty_sold > 0

        Weekly Dormant Account Count
            Activity table = UNION (distinct child_id, week_end_date) of data_867 and data_867_pap, both filtered to valid_order = 1 AND qty_sold > 0.
        
            Analysis weeks = the 8 most recent week_end_date values found in the activity table (MAX(week_end_date) down to 7 weeks prior). Do not use calendar/today's date.
        
            For each analysis week W:
        
            Universe: only child_ids with at least one activity row where week_end_date <= W.
            Dormant flag: a child_id in the universe is dormant if it has zero activity rows with week_end_date BETWEEN (W - 7 weeks) AND W (inclusive 8-week window, current week included).
            Output: week_start_date = W - 6 days, week_end_date = W, weekly_dormant_accounts = COUNT(DISTINCT dormant child_id).
        
            Order output by week_end_date ascending.
        
            Dormant Addition Trend: dormant_additions(W) = weekly_dormant_accounts(W) - weekly_dormant_accounts(W - 1 week).
        

        Top 25 accounts = top 25 parent accounts ranked by total demand vials (commercial + PAP) from Jan 1, 2025 through the current date, sorted descending by demand vials.
        Reactivated Account Rules:

            QUALIFYING ORDER: a week counts as "ordered" only if it contains a row with
            valid_order = 1 AND qty_sold > 0. Apply this same test to every leg of the rule.

            GRAIN: child_account_id x week. Never roll up to parent account.

            For evaluation week W, a child account is REACTIVATED only if ALL THREE hold:
            1. PRIOR ACTIVITY — at least one qualifying order strictly before week W-8.
            2. DORMANCY — zero qualifying orders in weeks W-8 through W-1.
            3. RETURN — at least one qualifying order in W.

            Span is exactly 9 weeks (8 dormant + W), plus a lifetime lookback used only for
            condition 1. Never use any other window or trigger length.

            NEW ACCOUNTS ARE NEVER REACTIVATED. An account whose first-ever qualifying order
            falls in W fails condition 1 and is classified NEW. Test via account-level
            first_qualifying_order_date < start_of_week(W-8). Never infer prior activity from
            signup, creation, or onboarding dates — existence is not activity.

            WEEKLY TREND: for every week W, re-apply the test against W's OWN trailing 8
            weeks; the window rolls with W and is never anchored to a fixed date.
            Output: week_start_date | week_end_date | reactivated_accounts = COUNT(DISTINCT child_account_id)
            - One account counts once per week, but may reactivate again in a later week.
            - A reactivation in W makes that account ineligible in W+1..W+8 by construction.
                Correct behaviour — do not adjust.
            - Every account ordering in W is exactly one of NEW / REACTIVATED / CONTINUING;
                the three must sum to distinct orderers in W. If not, flag it.
        
            List of REACTIVATED ACCOUNTS — TRAILING N WEEKS
                
                QUALIFYING ORDER: a (child_account_id, week) counts as ordered only if a row has
                valid_order = 1 AND qty_sold > 0. Apply this test to every leg. Never use signup,
                creation, or onboarding dates — existence is not activity.
                
                GRAIN: child_account_id x week. Never roll up to parent.
                
                Evaluate each of the N most recent completed weeks INDEPENDENTLY, then UNION ALL.
                Never treat the N weeks as one block; the 8-week lookback rolls with each W.
                
                For week W, an account is REACTIVATED only if ALL THREE hold:
                1. first_qualifying_order_date < start_of_week(W-8)   [strict <]
                2. zero qualifying orders in W-8 .. W-1  (all 8 weeks)
                3. >= 1 qualifying order in W
                Span is exactly 9 weeks regardless of N. N changes how many weeks are evaluated,
                never the window length.
                
                NEW IS NEVER REACTIVATED: first qualifying order inside W fails leg 1 → NEW, excluded.
                
                If history doesn't cover 9 weeks before the oldest evaluated week, stop and say so —
                never emit partial-window results.
                
                OUTPUT: week_start_date | week_end_date | child_id | child_name  unioned across all N weeks.
                An account may recur in a later week; a reactivation in W blocks W+1..W+8 by
                construction — correct, do not adjust.
                
                CHECK: per week, NEW + REACTIVATED + CONTINUING = distinct accounts ordering in W.
                If not, flag it.
    
    Cross Table Rules (data_867 + data_867_pap + forecast):    
        Attainment / Achievement / Against Budget is calculated as: Attainment (%) = (Actual Demand / Budget) × 100

    Cross Table Rules (revenue + revenue_forecast):
        balance_to_go = net_sales_forecast/gross_sales_forecast(revenue_forecast Table) - net_sales/gross_sales(revenue table) — always compute it this way, and never label a value "balance to go" unless it follows this exact formula.
 
    Default Rules:
        Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics with total Volume demand., where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level) (VERY IMPORTANT).
        All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, or segmented data.
        If the user does not explicitly specify a total demand denominator, assume overall national demand as the default denominator.
        For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
        All child entities roll up to their respective parent entities.
        If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
        Always accompany any growth metric or percentage value with the corresponding absolute volume demand value.
        Whenever the query references “nation,” compute the national-level metrics and include them in the output.
        Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
        Our Product is fyarro.
        For every time period in the output, explicitly display the corresponding number of business days
        Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
        Always perform aggregations using ID fields (e.g., child_id, parent_id) for accuracy, and include the corresponding names in the final output.
        All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
        If asked about demand by default give national demand don't group by parent_id or parent_name.
        Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.
        When displaying the daily average metric, always round and format the value to exactly 1 decimal place.
        Every demand metric must be named demand_vials_<direction><period> (e.g., demand_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every budget metric must be named budget_vials_<direction><period> (e.g., budget_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every revenue metric must be named _$_<direction><period> (e.g., gross_sales_$_r4w for recent 4 weeks) — never display revenue without this exact prefix and suffix format.
        Default to the child account for all account-level queries. Exception: for any Top 25, Top 75, or account tier/segment query, always anchor to the parent account instead — never the child.
        Whenever information is displayed at the account level, only display Commercial + PAP metrics. Don't default to Commercial
        Whenever a query is at the account level, never display daily average metrics — always display total (aggregate) metrics instead, unless the user explicitly instructs otherwise.
        All revenue and sales values must be reported to one decimal place (e.g., $12345.6). No exceptions — round, do not truncate.
        For any trend-level question, default to the most recent 52 weeks of data for account level,  demand and revenue/sales, unless a different time period is explicitly specified.
        Any query involving sales, revenue, net sales, or gross sales must anchor to the Revenue table as the single source of truth — no other table should be used for these metrics.
        Don't display/calculate the number of Business Days for revenue/sales related queries
        Sales/revenue trend data must always be shown broken down by month.
        An ordering account is the one whose valid_order=1 and qty_sold>0
        Whenever displaying actual sales metrics, always include the corresponding ex-factory vials quantity alongside them.
        
        A reactivated account is defined as a child account that was previously dormant (no order in the last 8 weeks) and has now placed an order, moving it out of dormant status.
        If a result is a decimal number, round it to one decimal place before presenting it (e.g., 3.14159 → 3.1).
        Longitudinal trend = pivoted view with accounts as rows and the actual week_end_date values as column headers (real dates, never generic labels like "Week 1"). Each cell shows that account's demand vials (commercial + PAP) for that week. Account level follows query context. Default window is the most recent 26 weeks, anchored to MAX(week_end_date) from the data itself. Build the pivot dynamically so column headers are the real dates — never static positional labels.
    
    Time Rules:

    If the user does not specify a time period, default to the most recent 8 weeks of available data.
    The current time period for revenue forecast must always be determined using the revenue table, not the revenue_forecast table.
    When output is at a weekly grain, suppress all daily average/daily-derived metrics unless the user explicitly requests a daily breakdown alongside the weekly view.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total demand and Daily Average demand, where Daily Average = Total demand / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    All output metrics must include the time window in their label (e.g., demand_4w, demand_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_demand, monthly_demand, quarterly_demand, yearly_demand) and should not include an additional time window prefix or suffix.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 8 Weeks (R8W) vs Prior 8 Weeks (P8W).
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    Always determine the latest time period using transaction_date, Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
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

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total demand vs Total demand
Daily average vs Daily average

Never compare **daily averages for one period with total demand for the other period**.

---


CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

For comparisons:
1. Identify requested time periods.
2. Determine calendar boundaries.
3. Check completeness using max_week_end_date.
 - Make a decision based on period completness:
CASE 
  WHEN pc.is_recent_period_complete = 1 
   AND pc.is_previous_period_complete = 1
  THEN total_growth
  ELSE NULL
END AS total_growth (VERY IMPORTANT)
4. If both periods are complete → aggregate totals at period level and display total growth also calculate daily average at period level and display daily average growth.
5. If ANY period is incomplete -> you MUST NOT generate, compute, or include total growth or national total growth in the output schema itself. These fields must be completely omitted (not NULL, not blank). ONLY include daily average growth and national daily average growth. This rule strictly overrides all other instructions, including any rule that says to always display total metrics (MANDATORY).
Metric Visibility Rule (MANDATORY):
- If BOTH periods are complete → display BOTH total volume growth and daily average growth (including national metrics).
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume demand and total National demand.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume demand and total National demand.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---


TABLE SCHEMA:

Table: data_867 — transaction-level demand dataset
- customer (VARCHAR): distributor/wholesaler name at order level (e.g., McKesson Plasma MPB, AmerisourceBergen)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- customer_name (VARCHAR): ship-to/site account name as recorded on the order (short form)
- street (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- unit_price (DECIMAL): price per unit
- extended_price (DECIMAL): total order line value (unit_price × qty, pre-return)
- qty_sold (INT): units sold 
- qty_return (INT): units returned
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- speciality_distributor (VARCHAR): specialty distributor name fulfilling the order (CENCORA, MSH, MPB)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic, Pharmacy)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- qty_sold_extended (INT): net qty sold after adjustments/aggregation
- qty_return_extended (INT): net qty returned after adjustments/aggregation
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: data_867_pap — transaction-level demand dataset
- transaction_type (VARCHAR): Transaction Type Pap or Commercial (PAP, COM)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- qty_sold (INT): units sold 
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: forecast — monthly budget/forecast dataset
- transaction_type (VARCHAR): Transaction Type scope for the forecast COM->Commercial, COM+PAP -> Commercial with PAP (COM, COM+PAP)
- month_year (VARCHAR): month label (e.g., 2026-01)
- budget (INT): forecasted/budgeted demand (units) for the month
- number_of_business_days_month (INT): count of business days in that month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue -  Net sales and Gross Sales at shipment date level
- ex_factory_vials (INT): units sold in that month
- shipment_date (DATE): Date at which the drug was shiped (YYYY-MM-DD)
- valid_order (INT): valid order flag (0,1)
- wholesaler_name (VARCHAR): Name of the wholesaler 
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales (DECIMAL): actual net sales at shipment level
- gross_sales (DECIMAL): actual net sales at shipment level
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue_forecast — daily budget/forecast for net sales and gross sales dataset
- date (DATE): date in the format (YYYY-MM-DD)
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales_forecast (DECIMAL): forecasted/budgeted net sales (dollars) for the month
- gross_sales_forecast (DECIMAL): forecasted/budgeted gross sales (dollars) for the month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: parent_marketing_target - Parent Target Accounts Details
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- account_classification (VARCHAR): Classified into 3 categories (Deepen Base, Sustain Accounts, Unlock Growth)

Table: mtor_alerts - HCP-to-account mapping table linking mTOR therapy prescribers (NPI, specialty, therapy start date) to their practice location and corresponding child/parent account hierarchy.
- mtor_therapy (VARCHAR): mtor therapy name
- therapy_start_date (DATE): start date of the mTOR therapy
- hcp_name (VARCHAR): name of the hcp
- npi (INTEGER): HCP NPI Value
- hcp_speciality (VARCHAR): Speciality of the HCP
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- address (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- child_id (VARCHAR): unique site/child account identifier
- child_name (VARCHAR): full/legal name of the site (child) account
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
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
    "intent_summary": "Calculate total demand for the last 13 weeks based on the most recent date available in the dataset.",
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
        "metric_name": "total_demand",
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
        "columns": ["total_demand"],
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
    - Use ONLY the provided table and columns
    - Do NOT invent columns, tables, or values
    - Be explicit and deterministic
    - Every filter, aggregation, and grouping must be stated
    - If feedback is provided, revise ONLY the affected parts
    - Preserve correct logic from previous decompositions

   ────────────────────────
    Metric & Output Handling Rules (Must Always Be Enforced):
    ────────────────────────
    data_867 Rules:
        For data_867: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867 is the commercial demand table
         
    


    data_867_pap Rules:
        For data_867_pap: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867_pap contains records for both PAP and Commercial transactions in a single table, differentiated by the transaction_type column:
            transaction_type = 'PAP' → record belongs to PAP
            transaction_type = 'COM' → record belongs to Commercial
        If the query explicitly asks for PAP (or "Commercial + PAP"), then both transaction_type = 'COM' and transaction_type = 'PAP' records should be included from data_867_pap — i.e., do not filter by transaction_type at all, or use transaction_type IN ('COM', 'PAP')
    
    forecast Rules:
        The budget column in data_867_forecast represents the monthly demand forecast value for that particular month_year — i.e., it is the forecasted/budgeted demand quantity, not an actual/realized value.
        MTD Forecast = (budget value for the current month_year / number_of_business_days_month for the current month_year) × business_days_elapsed_so_far for the current month_year. Report the answer to the nearest integer
        QTD/YTD Forecast/Budget = SUM(budget) for all month_year rows with quarter_year/year matching the current period and month_year < current month_year, plus the MTD Forecast for the current month_year. Report the answer to the nearest integer

    Revenue Table Rules:
        The revenue table contains shipment-level data. Consisting of net sales and gross sales.
        For any query related to revenue, this table must be treated as the anchor/source of truth. All revenue responses should be derived from this table rather than any other source.
        Revenue and net sales are synonyms
    
    revenue_forecast Rules:
        Anchor this table for revenue/ sales forecast 
        The net_sales_forecast and gross_sales_forecast columns in revenue_forecast represent the daily demand forecast value for that particular date — i.e., they are the forecasted/budgeted values for net_sales and gross_sales respectively, not actual/realized values.

    parent_marketing_target Rules:
            "Target accounts" and "Top 75 accounts" queries anchor to the parent marketing target table.
            Account classification and bucket are synonyms — always treat them as the same field in parent_marketing_target table.
    
    mtor_alerts Rules:
        Route mTOR Patient Therapy Alert queries — patient-level alert counts by region, therapy initiation dates, unique therapies administered, and ongoing therapy account counts — to the mtor_alerts table.
        MTOR alerts table queries default to YTD (January 1 of the current year through the current date).
        Any MTOR alerts query defaults to YTD: Jan 1 of the CURRENT calendar year through the latest available date.

    Cross Table Rules (data_867 + data_867_pap):
        To calculate demand, always anchor to the qty_sold column and filter records where valid_orders = 1. Compute total demand as SUM(qty_sold) from data_867 plus SUM(qty_sold) from data_867_pap.
        If a query does not explicitly specify whether to calculate Commercial demand only or Commercial + PAP demand, then by default, only Commercial demand should be shown (i.e., filter using transaction_type = 'COM' for data_pap and complete data of data_867 for that timeperiod).
        If the query explicitly asks for PAP (or "Commercial + PAP"), then both transaction_type = 'COM' and transaction_type = 'PAP' records should be included from data_867_pap — i.e., do not filter by transaction_type at all, or use transaction_type IN ('COM', 'PAP')
        Breadth = # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Depth = demand ÷ # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Always use these exact formulas.
        Every account-level query is Commercial + PAP by default. Never scope to Commercial only unless the user explicitly asks for it.
        **Ordering account:**
            A child account that has at least one record in the window where BOTH
            conditions hold on the SAME record:
            - valid_order = 1
            - qty_sold > 0

            
        Weekly Dormant Account Count
            Activity table = UNION (distinct child_id, week_end_date) of data_867 and data_867_pap, both filtered to valid_order = 1 AND qty_sold > 0.
                
            Analysis weeks = the 8 most recent week_end_date values found in the activity table (MAX(week_end_date) down to 7 weeks prior). Do not use calendar/today's date.
                
            For each analysis week W:
                
            Universe: only child_ids with at least one activity row where week_end_date <= W.
            Dormant flag: a child_id in the universe is dormant if it has zero activity rows with week_end_date BETWEEN (W - 7 weeks) AND W (inclusive 8-week window, current week included).
            Output: week_start_date = W - 6 days, week_end_date = W, weekly_dormant_accounts = COUNT(DISTINCT dormant child_id).
                
            Order output by week_end_date ascending.
                
            Dormant Addition Trend: dormant_additions(W) = weekly_dormant_accounts(W) - weekly_dormant_accounts(W - 1 week).
        

        Top 25 accounts = top 25 parent accounts ranked by total demand vials (commercial + PAP) from Jan 1, 2025 through the current date, sorted descending by demand vials.
        Reactivated Account Rules:

            QUALIFYING ORDER: a week counts as "ordered" only if it contains a row with
            valid_order = 1 AND qty_sold > 0. Apply this same test to every leg of the rule.

            GRAIN: child_account_id x week. Never roll up to parent account.

            For evaluation week W, a child account is REACTIVATED only if ALL THREE hold:
            1. PRIOR ACTIVITY — at least one qualifying order strictly before week W-8.
            2. DORMANCY — zero qualifying orders in weeks W-8 through W-1.
            3. RETURN — at least one qualifying order in W.

            Span is exactly 9 weeks (8 dormant + W), plus a lifetime lookback used only for
            condition 1. Never use any other window or trigger length.

            NEW ACCOUNTS ARE NEVER REACTIVATED. An account whose first-ever qualifying order
            falls in W fails condition 1 and is classified NEW. Test via account-level
            first_qualifying_order_date < start_of_week(W-8). Never infer prior activity from
            signup, creation, or onboarding dates — existence is not activity.

            WEEKLY TREND: for every week W, re-apply the test against W's OWN trailing 8
            weeks; the window rolls with W and is never anchored to a fixed date.
            Output: week_start_date | week_end_date | reactivated_accounts = COUNT(DISTINCT child_account_id)
            - One account counts once per week, but may reactivate again in a later week.
            - A reactivation in W makes that account ineligible in W+1..W+8 by construction.
                Correct behaviour — do not adjust.
            - Every account ordering in W is exactly one of NEW / REACTIVATED / CONTINUING;
                the three must sum to distinct orderers in W. If not, flag it.
                
        List of REACTIVATED ACCOUNTS — TRAILING N WEEKS
        
                    QUALIFYING ORDER: a (child_account_id, week) counts as ordered only if a row has
                    valid_order = 1 AND qty_sold > 0. Apply this test to every leg. Never use signup,
                    creation, or onboarding dates — existence is not activity.
        
                    GRAIN: child_account_id x week. Never roll up to parent.
        
                    Evaluate each of the N most recent completed weeks INDEPENDENTLY, then UNION ALL.
                    Never treat the N weeks as one block; the 8-week lookback rolls with each W.
        
                    For week W, an account is REACTIVATED only if ALL THREE hold:
                    1. first_qualifying_order_date < start_of_week(W-8)   [strict <]
                    2. zero qualifying orders in W-8 .. W-1  (all 8 weeks)
                    3. >= 1 qualifying order in W
                    Span is exactly 9 weeks regardless of N. N changes how many weeks are evaluated,
                    never the window length.
        
                    NEW IS NEVER REACTIVATED: first qualifying order inside W fails leg 1 → NEW, excluded.
        
                    If history doesn't cover 9 weeks before the oldest evaluated week, stop and say so —
                    never emit partial-window results.
        
                    OUTPUT: week_start_date | week_end_date | child_id | child_name  unioned across all N weeks.
                    An account may recur in a later week; a reactivation in W blocks W+1..W+8 by
                    construction — correct, do not adjust.
        
                    CHECK: per week, NEW + REACTIVATED + CONTINUING = distinct accounts ordering in W.
                    If not, flag it.

    Cross Table Rules (data_867 + data_867_pap + forecast):    
        Attainment / Achievement / Against Budget is calculated as: Attainment (%) = (Actual Demand / Budget) × 100

    Cross Table Rules (revenue + revenue_forecast):
        balance_to_go = net_sales_forecast/gross_sales_forecast(revenue_forecast Table) - net_sales/gross_sales(revenue table) — always compute it this way, and never label a value "balance to go" unless it follows this exact formula.
    

    Default Rules:
        Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics with total Volume demand., where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level) (VERY IMPORTANT).
        All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, or segmented data.
        If the user does not explicitly specify a total demand denominator, assume overall national demand as the default denominator.
        For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
        All child entities roll up to their respective parent entities.
        If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
        Always accompany any growth metric or percentage value with the corresponding absolute volume demand value.
        Whenever the query references “nation,” compute the national-level metrics and include them in the output.
        Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
        Our Product is fyarro.
        For every time period in the output, explicitly display the corresponding number of business days
        Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
        Always perform aggregations using ID fields (e.g., child_id, parent_id) for accuracy, and include the corresponding names in the final output.
        All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
        If asked about demand by default give national demand don't group by parent_id or parent_name.
        Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.
        When displaying the daily average metric, always round and format the value to exactly 1 decimal place.
        Every demand metric must be named demand_vials_<direction><period> (e.g., demand_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every budget metric must be named budget_vials_<direction><period> (e.g., budget_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every revenue metric must be named _$_<direction><period> (e.g., gross_sales_$_r4w for recent 4 weeks) — never display revenue without this exact prefix and suffix format.
        Default to the child account for all account-level queries. Exception: for any Top 25, Top 75, or account tier/segment query, always anchor to the parent account instead — never the child.
        Whenever information is displayed at the account level (parent and child) including depth and breadth, only display Commercial + PAP metrics. Don't default to Commercial
        Whenever information is displayed at the account level (parent and child) including depth and breadth, only display Commercial + PAP metrics. Don't default to Commercial
        All revenue and sales values must be reported to one decimal place (e.g., $12345.6). No exceptions — round, do not truncate.
        For any trend-level question, default to the most recent 52 weeks of data for account level,  demand and revenue/sales, unless a different time period is explicitly specified.
        Any query involving sales, revenue, net sales, or gross sales must anchor to the Revenue table as the single source of truth — no other table should be used for these metrics.
        Don't display/calculate the number of Business Days for revenue/sales related queries
        Sales/revenue trend data must always be shown broken down by month.
        Whenever displaying actual sales metrics, always include the corresponding ex-factory vials quantity alongside them.
        An ordering account is the one whose valid_order=1 and qty_sold>0
        If a result is a decimal number, round it to one decimal place before presenting it (e.g., 3.14159 → 3.1).
        Longitudinal trend = pivoted view with accounts as rows and the actual week_end_date values as column headers (real dates, never generic labels like "Week 1"). Each cell shows that account's demand vials (commercial + PAP) for that week. Account level follows query context. Default window is the most recent 26 weeks, anchored to MAX(week_end_date) from the data itself. Build the pivot dynamically so column headers are the real dates — never static positional labels.
    
    Time Rules:

    If the user does not specify a time period, default to the most recent 8 weeks of available data.
    The current time period for revenue forecast must always be determined using the revenue table, not the revenue_forecast table.
    When output is at a weekly grain, suppress all daily average/daily-derived metrics unless the user explicitly requests a daily breakdown alongside the weekly view.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total demand and Daily Average demand, where Daily Average = Total demand / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    All output metrics must include the time window in their label (e.g., demand_4w, demand_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_demand, monthly_demand, quarterly_demand, yearly_demand) and should not include an additional time window prefix or suffix.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 8 Weeks (R8W) vs Prior 8 Weeks (P8W).
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    Always determine the latest time period using transaction_date, Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
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

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total demand vs Total demand
Daily average vs Daily average

Never compare **daily averages for one period with total demand for the other period**.

---


CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

For comparisons:
1. Identify requested time periods.
2. Determine calendar boundaries.
3. Check completeness using max_week_end_date.
 - Make a decision based on period completness:
CASE 
  WHEN pc.is_recent_period_complete = 1 
   AND pc.is_previous_period_complete = 1
  THEN total_growth
  ELSE NULL
END AS total_growth (VERY IMPORTANT)
4. If both periods are complete → aggregate totals at period level and display total growth also calculate daily average at period level and display daily average growth.
5. If ANY period is incomplete -> you MUST NOT generate, compute, or include total growth or national total growth in the output schema itself. These fields must be completely omitted (not NULL, not blank). ONLY include daily average growth and national daily average growth. This rule strictly overrides all other instructions, including any rule that says to always display total metrics (MANDATORY).
Metric Visibility Rule (MANDATORY):
- If BOTH periods are complete → display BOTH total volume growth and daily average growth (including national metrics).
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume demand and total National demand.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume demand and total National demand.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---


TABLE SCHEMA:

Table: data_867 — transaction-level demand dataset
- customer (VARCHAR): distributor/wholesaler name at order level (e.g., McKesson Plasma MPB, AmerisourceBergen)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- customer_name (VARCHAR): ship-to/site account name as recorded on the order (short form)
- street (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- unit_price (DECIMAL): price per unit
- extended_price (DECIMAL): total order line value (unit_price × qty, pre-return)
- qty_sold (INT): units sold 
- qty_return (INT): units returned
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- speciality_distributor (VARCHAR): specialty distributor name fulfilling the order (CENCORA, MSH, MPB)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic, Pharmacy)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- qty_sold_extended (INT): net qty sold after adjustments/aggregation
- qty_return_extended (INT): net qty returned after adjustments/aggregation
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: data_867_pap — transaction-level demand dataset
- transaction_type (VARCHAR): Transaction Type Pap or Commercial (PAP, COM)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- qty_sold (INT): units sold 
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: forecast — monthly budget/forecast dataset
- transaction_type (VARCHAR): Transaction Type scope for the forecast COM->Commercial, COM+PAP -> Commercial with PAP (COM, COM+PAP)
- month_year (VARCHAR): month label (e.g., 2026-01)
- budget (INT): forecasted/budgeted demand (units) for the month
- number_of_business_days_month (INT): count of business days in that month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue -  Net sales and Gross Sales at shipment date level
- ex_factory_vials (INT): units sold in that month
- shipment_date (DATE): Date at which the drug was shiped (YYYY-MM-DD)
- valid_order (INT): valid order flag (0,1)
- wholesaler_name (VARCHAR): Name of the wholesaler 
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales (DECIMAL): actual net sales at shipment level
- gross_sales (DECIMAL): actual net sales at shipment level
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue_forecast — dailyy budget/forecast for net sales and gross sales dataset
- date (DATE): date in the format (YYYY-MM-DD)
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales_forecast (DECIMAL): forecasted/budgeted net sales (dollars) for the month
- gross_sales_forecast (DECIMAL): forecasted/budgeted gross sales (dollars) for the month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: parent_marketing_target - Parent Target Accounts Details
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- account_classification (VARCHAR): Classified into 3 categories (Deepen Base, Sustain Accounts, Unlock Growth)

Table: mtor_alerts - HCP-to-account mapping table linking mTOR therapy prescribers (NPI, specialty, therapy start date) to their practice location and corresponding child/parent account hierarchy.
- mtor_therapy (VARCHAR): mtor therapy name
- therapy_start_date (DATE): start date of the mTOR therapy
- hcp_name (VARCHAR): name of the hcp
- npi (INTEGER): HCP NPI Value
- hcp_speciality (VARCHAR): Speciality of the HCP
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- address (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- child_id (VARCHAR): unique site/child account identifier
- child_name (VARCHAR): full/legal name of the site (child) account
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
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
    "intent_summary": "Calculate total demand for the last 13 weeks based on the most recent date available in the dataset.",
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
        "metric_name": "total_demand",
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
        "columns": ["total_demand"],
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
    data_867 Rules:
        For data_867: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867 is the commercial demand table
         
    


    data_867_pap Rules:
        For data_867_pap: The table contains week_end_date, month_year, quarter_year and year. Use week_end_date for weekly calculations.
        data_867_pap contains records for both PAP and Commercial transactions in a single table, differentiated by the transaction_type column:
            transaction_type = 'PAP' → record belongs to PAP
            transaction_type = 'COM' → record belongs to Commercial
        If the query explicitly asks for PAP (or "Commercial + PAP"), then both transaction_type = 'COM' and transaction_type = 'PAP' records should be included from data_867_pap — i.e., do not filter by transaction_type at all, or use transaction_type IN ('COM', 'PAP')
    
    forecast Rules:
        The budget column in data_867_forecast represents the monthly demand forecast value for that particular month_year — i.e., it is the forecasted/budgeted demand quantity, not an actual/realized value.
        MTD Forecast / Budget = (budget value for the current month_year / number_of_business_days_month for the current month_year) × business_days_elapsed_so_far for the current month_year. Report the answer to the nearest integer
        QTD/YTD Forecast/Budget = SUM(budget) for all month_year rows with quarter_year/year matching the current period and month_year < current month_year, plus the MTD Forecast for the current month_year. Report the answer to the nearest integer
    
    Revenue Table Rules:
        The revenue table contains shipment-level data. Consisting of net sales and gross sales.
        For any query related to revenue, this table must be treated as the anchor/source of truth. All revenue responses should be derived from this table rather than any other source.
        Revenue and net sales are synonyms
        
    revenue_forecast Rules:
        Anchor this table for revenue/ sales forecast
        The net_sales_forecast and gross_sales_forecast columns in revenue_forecast represent the daily demand forecast value for that particular date — i.e., they are the forecasted/budgeted values for net_sales and gross_sales respectively, not actual/realized values.

    parent_marketing_target Rules:
        "Target accounts" and "Top 75 accounts" queries anchor to the parent marketing target table.
        Account classification and bucket are synonyms — always treat them as the same field in parent_marketing_target table.    
    
    mtor_alerts Rules:
        Route mTOR Patient Therapy Alert queries — patient-level alert counts by region, therapy initiation dates, unique therapies administered, and ongoing therapy account counts — to the mtor_alerts table.
        MTOR alerts table queries default to YTD (January 1 of the current year through the current date).
        Any MTOR alerts query defaults to YTD: Jan 1 of the CURRENT calendar year through the latest available date.

    Cross Table Rules (data_867 + data_867_pap):
        To calculate demand, always anchor to the qty_sold column and filter records where valid_orders = 1. Compute total demand as SUM(qty_sold) from data_867 plus SUM(qty_sold) from data_867_pap.
        If a query does not explicitly specify whether to calculate Commercial demand only or Commercial + PAP demand, then by default, only Commercial demand should be shown (i.e., filter using transaction_type = 'COM' for data_pap and complete data of data_867 for that timeperiod).
        If the query explicitly asks for PAP (or "Commercial + PAP"), then both transaction_type = 'COM' and transaction_type = 'PAP' records should be included from data_867_pap — i.e., do not filter by transaction_type at all, or use transaction_type IN ('COM', 'PAP')
        Breadth = # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Depth = demand ÷ # of ordering child accounts where the qty_sold > 0 and valid_order = 1. Always use these exact formulas.
        Every account-level query is Commercial + PAP by default. Never scope to Commercial only unless the user explicitly asks for it.
        An ordering account is the one whose valid_order=1 and qty_sold>0
        **Ordering account:**
            A child account that has at least one record in the window where BOTH
            conditions hold on the SAME record:
            - valid_order = 1
            - qty_sold > 0

        Weekly Dormant Account Count
            Activity table = UNION (distinct child_id, week_end_date) of data_867 and data_867_pap, both filtered to valid_order = 1 AND qty_sold > 0.
                        
            Analysis weeks = the 8 most recent week_end_date values found in the activity table (MAX(week_end_date) down to 7 weeks prior). Do not use calendar/today's date.
                        
            For each analysis week W:
                        
            Universe: only child_ids with at least one activity row where week_end_date <= W.
            Dormant flag: a child_id in the universe is dormant if it has zero activity rows with week_end_date BETWEEN (W - 7 weeks) AND W (inclusive 8-week window, current week included).
            Output: week_start_date = W - 6 days, week_end_date = W, weekly_dormant_accounts = COUNT(DISTINCT dormant child_id).
                        
            Order output by week_end_date ascending.
                        
            Dormant Addition Trend: dormant_additions(W) = weekly_dormant_accounts(W) - weekly_dormant_accounts(W - 1 week).
        
        Top 25 accounts = top 25 parent accounts ranked by total demand vials (commercial + PAP) from Jan 1, 2025 through the current date, sorted descending by demand vials.

        Reactivated Account Rules:

            QUALIFYING ORDER: a week counts as "ordered" only if it contains a row with
            valid_order = 1 AND qty_sold > 0. Apply this same test to every leg of the rule.

            GRAIN: child_account_id x week. Never roll up to parent account.

            For evaluation week W, a child account is REACTIVATED only if ALL THREE hold:
            1. PRIOR ACTIVITY — at least one qualifying order strictly before week W-8.
            2. DORMANCY — zero qualifying orders in weeks W-8 through W-1.
            3. RETURN — at least one qualifying order in W.

            Span is exactly 9 weeks (8 dormant + W), plus a lifetime lookback used only for
            condition 1. Never use any other window or trigger length.

            NEW ACCOUNTS ARE NEVER REACTIVATED. An account whose first-ever qualifying order
            falls in W fails condition 1 and is classified NEW. Test via account-level
            first_qualifying_order_date < start_of_week(W-8). Never infer prior activity from
            signup, creation, or onboarding dates — existence is not activity.

            WEEKLY TREND: for every week W, re-apply the test against W's OWN trailing 8
            weeks; the window rolls with W and is never anchored to a fixed date.
            Output: week_start_date | week_end_date | reactivated_accounts = COUNT(DISTINCT child_account_id)
            - One account counts once per week, but may reactivate again in a later week.
            - A reactivation in W makes that account ineligible in W+1..W+8 by construction.
                Correct behaviour — do not adjust.
            - Every account ordering in W is exactly one of NEW / REACTIVATED / CONTINUING;
                the three must sum to distinct orderers in W. If not, flag it.

        List of REACTIVATED ACCOUNTS — TRAILING N WEEKS

            QUALIFYING ORDER: a (child_account_id, week) counts as ordered only if a row has
            valid_order = 1 AND qty_sold > 0. Apply this test to every leg. Never use signup,
            creation, or onboarding dates — existence is not activity.

            GRAIN: child_account_id x week. Never roll up to parent.

            Evaluate each of the N most recent completed weeks INDEPENDENTLY, then UNION ALL.
            Never treat the N weeks as one block; the 8-week lookback rolls with each W.

            For week W, an account is REACTIVATED only if ALL THREE hold:
            1. first_qualifying_order_date < start_of_week(W-8)   [strict <]
            2. zero qualifying orders in W-8 .. W-1  (all 8 weeks)
            3. >= 1 qualifying order in W
            Span is exactly 9 weeks regardless of N. N changes how many weeks are evaluated,
            never the window length.

            NEW IS NEVER REACTIVATED: first qualifying order inside W fails leg 1 → NEW, excluded.

            If history doesn't cover 9 weeks before the oldest evaluated week, stop and say so —
            never emit partial-window results.

            OUTPUT: week_start_date | week_end_date | child_id | child_name  unioned across all N weeks.
            An account may recur in a later week; a reactivation in W blocks W+1..W+8 by
            construction — correct, do not adjust.

            CHECK: per week, NEW + REACTIVATED + CONTINUING = distinct accounts ordering in W.
            If not, flag it.

    Cross Table Rules (data_867 + data_867_pap + forecast):    
        Attainment / Achievement / Against Budget is calculated as: Attainment (%) = (Actual Demand / Budget) × 100

    Cross Table Rules (revenue + revenue_forecast):
        balance_to_go = net_sales_forecast/gross_sales_forecast(revenue_forecast Table for that time period) - net_sales/gross_sales(revenue table for that time period) — always compute it this way, and never label a value "balance to go" unless it follows this exact formula.
 

    Default Rules:
        Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics with total Volume demand., where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level) (VERY IMPORTANT).
        All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, or segmented data.
        If the user does not explicitly specify a total demand denominator, assume overall national demand as the default denominator.
        For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
        All child entities roll up to their respective parent entities.
        If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
        Always accompany any growth metric or percentage value with the corresponding absolute volume demand value.
        Whenever the query references “nation,” compute the national-level metrics and include them in the output.
        Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
        Our Product is fyarro.
        For every time period in the output, explicitly display the corresponding number of business days
        Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
        Always perform aggregations using ID fields (e.g., child_id, parent_id) for accuracy, and include the corresponding names in the final output.
        All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
        If asked about demand by default give national demand don't group by parent_id or parent_name.
        Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.
        When displaying the daily average metric, always round and format the value to exactly 1 decimal place.
        Every demand metric must be named demand_vials_<direction><period> (e.g., demand_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every budget metric must be named budget_vials_<direction><period> (e.g., budget_vials_r4w for recent 4 weeks) — never display demand without this exact prefix and suffix format.
        Every revenue metric must be named _$_<direction><period> (e.g., gross_sales_$_r4w for recent 4 weeks) — never display revenue without this exact prefix and suffix format.
        Default to the child account for all account-level queries. Exception: for any Top 25, Top 75, or account tier/segment query, always anchor to the parent account instead — never the child.
        Whenever information is displayed at the account level, only display Commercial + PAP metrics. DOn't default to Commercial
        Whenever information is displayed at the account level (parent and child) including depth and breadth, only display Commercial + PAP metrics. Don't default to Commercial
        All revenue and sales values must be reported to one decimal place (e.g., $12345.6). No exceptions — round, do not truncate.
        For any trend-level question, default to the most recent 52 weeks of data for demand, account level and revenue/sales, unless a different time period is explicitly specified.
        Any query involving sales, revenue, net sales, or gross sales must anchor to the Revenue table as the single source of truth — no other table should be used for these metrics.
        Don't display/calculate the number of Business Days for revenue/sales related queries
        Sales/revenue trend data must always be shown broken down by month.
        Whenever displaying actual sales metrics, always include the corresponding ex-factory vials quantity alongside them.
        A reactivated account is defined as a child account that was previously dormant (no order in the last 8 weeks) and has now placed an order, moving it out of dormant status.
        If a result is a decimal number, round it to one decimal place before presenting it (e.g., 3.14159 → 3.1).
        Longitudinal trend = pivoted view with accounts as rows and the actual week_end_date values as column headers (real dates, never generic labels like "Week 1"). Each cell shows that account's demand vials (commercial + PAP) for that week. Account level follows query context. Default window is the most recent 26 weeks, anchored to MAX(week_end_date) from the data itself. Build the pivot dynamically so column headers are the real dates — never static positional labels.
    
    Time Rules:

    If the user does not specify a time period, default to the most recent 8 weeks of available data.
    The current time period for revenue forecast must always be determined using the revenue table, not the revenue_forecast table.
    When output is at a weekly grain, suppress all daily average/daily-derived metrics unless the user explicitly requests a daily breakdown alongside the weekly view.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total demand and Daily Average demand, where Daily Average = Total demand / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    All output metrics must include the time window in their label (e.g., demand_4w, demand_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_demand, monthly_demand, quarterly_demand, yearly_demand) and should not include an additional time window prefix or suffix.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 8 Weeks (R8W) vs Prior 8 Weeks (P8W).
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    Always determine the latest time period using transaction_date, Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
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

CONSISTENT COMPARISON RULE (CRITICAL)

Both periods in a comparison must use the **same aggregation basis**.

Allowed comparisons:

Total demand vs Total demand
Daily average vs Daily average

Never compare **daily averages for one period with total demand for the other period**.

---


CALCULATION ORDER (MANDATORY)

All calculations must follow this strict order:

For comparisons:
1. Identify requested time periods.
2. Determine calendar boundaries.
3. Check completeness using max_week_end_date.
 - Make a decision based on period completness:
CASE 
  WHEN pc.is_recent_period_complete = 1 
   AND pc.is_previous_period_complete = 1
  THEN total_growth
  ELSE NULL
END AS total_growth (VERY IMPORTANT)
4. If both periods are complete → aggregate totals at period level and display total growth also calculate daily average at period level and display daily average growth.
5. If ANY period is incomplete -> you MUST NOT generate, compute, or include total growth or national total growth in the output schema itself. These fields must be completely omitted (not NULL, not blank). ONLY include daily average growth and national daily average growth. This rule strictly overrides all other instructions, including any rule that says to always display total metrics (MANDATORY).
Metric Visibility Rule (MANDATORY):
- If BOTH periods are complete → display BOTH total volume growth and daily average growth (including national metrics).
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume demand and total National demand.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume demand and total National demand.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---


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

TABLE SCHEMA:

Table: data_867 — transaction-level demand dataset
- customer (VARCHAR): distributor/wholesaler name at order level (e.g., McKesson Plasma MPB, AmerisourceBergen)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- customer_name (VARCHAR): ship-to/site account name as recorded on the order (short form)
- street (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- unit_price (DECIMAL): price per unit
- extended_price (DECIMAL): total order line value (unit_price × qty, pre-return)
- qty_sold (INT): units sold 
- qty_return (INT): units returned
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- speciality_distributor (VARCHAR): specialty distributor name fulfilling the order (CENCORA, MSH, MPB)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic, Pharmacy)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- qty_sold_extended (INT): net qty sold after adjustments/aggregation
- qty_return_extended (INT): net qty returned after adjustments/aggregation
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: data_867_pap — transaction-level demand dataset
- transaction_type (VARCHAR): Transaction Type Pap or Commercial (PAP, COM)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- qty_sold (INT): units sold 
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: forecast — monthly budget/forecast dataset
- transaction_type (VARCHAR): Transaction Type scope for the forecast COM->Commercial, COM+PAP -> Commercial with PAP (COM, COM+PAP)
- month_year (VARCHAR): month label (e.g., 2026-01)
- budget (INT): forecasted/budgeted demand (units) for the month
- number_of_business_days_month (INT): count of business days in that month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue -  Net sales and Gross Sales at shipment date level
- ex_factory_vials (INT): units sold in that month
- shipment_date (DATE): Date at which the drug was shiped (YYYY-MM-DD)
- valid_order (INT): valid order flag (0,1)
- wholesaler_name (VARCHAR): Name of the wholesaler 
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales (DECIMAL): actual net sales at shipment level
- gross_sales (DECIMAL): actual net sales at shipment level
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue_forecast — daily budget/forecast for net sales and gross sales dataset
- date (DATE): date in the format (YYYY-MM-DD)
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales_forecast (DECIMAL): forecasted/budgeted net sales (dollars) for the month
- gross_sales_forecast (DECIMAL): forecasted/budgeted gross sales (dollars) for the month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: parent_marketing_target - Parent Target Accounts Details
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- account_classification (VARCHAR): Classified into 3 categories (Deepen Base, Sustain Accounts, Unlock Growth)

Table: mtor_alerts - HCP-to-account mapping table linking mTOR therapy prescribers (NPI, specialty, therapy start date) to their practice location and corresponding child/parent account hierarchy.
- mtor_therapy (VARCHAR): mtor therapy name
- therapy_start_date (DATE): start date of the mTOR therapy
- hcp_name (VARCHAR): name of the hcp
- npi (INTEGER): HCP NPI Value
- hcp_speciality (VARCHAR): Speciality of the HCP
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- address (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- child_id (VARCHAR): unique site/child account identifier
- child_name (VARCHAR): full/legal name of the site (child) account
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
    ────────────────────────
    EXAMPLES (FOR GUIDANCE)
    ────────────────────────

    Example 1:
    User Question:
    "Total quantity sold by region in Q4-24"

    Expected SQL Output:
    SELECT child_region, SUM(rytelo_total_mg) AS total_qty
    FROM data_867
    WHERE quarter_year = 'Q4-24'
    GROUP BY child_region;

    ────────────────────────

    Example 2:
    User Question:
    "Top 5 child accounts in Central by quantity sold"

    Expected SQL Output:
    SELECT parent_name, SUM(rytelo_total_mg) AS total_qty
    FROM data_867
    WHERE child_region = 'Central'
    GROUP BY child_account_name
    ORDER BY total_qty DESC
    LIMIT 5;

    """

    response = model_1.invoke(prompt).content[0]["text"]
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
You are an expert SQL reviewer for Snowflake SQL.

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

❌ The SQL contains invalid Snowflake SQL syntax

────────────────────────
WHAT IS EXPLICITLY ALLOWED
────────────────────────

You MUST allow the following patterns if used correctly:

✔ Common Table Expressions (WITH clauses)
✔ Subqueries in SELECT / WHERE / FROM
✔ Derived-date logic using:
MAX(date_column)

✔ Date functions such as:
DATEADD, DATEDIFF

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


TABLE SCHEMA:


Table: data_867 — transaction-level demand dataset
- customer (VARCHAR): distributor/wholesaler name at order level (e.g., McKesson Plasma MPB, AmerisourceBergen)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- customer_name (VARCHAR): ship-to/site account name as recorded on the order (short form)
- street (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- unit_price (DECIMAL): price per unit
- extended_price (DECIMAL): total order line value (unit_price × qty, pre-return)
- qty_sold (INT): units sold 
- qty_return (INT): units returned
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- speciality_distributor (VARCHAR): specialty distributor name fulfilling the order (CENCORA, MSH, MPB)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic, Pharmacy)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- qty_sold_extended (INT): net qty sold after adjustments/aggregation
- qty_return_extended (INT): net qty returned after adjustments/aggregation
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: data_867_pap — transaction-level demand dataset
- transaction_type (VARCHAR): Transaction Type Pap or Commercial (PAP, COM)
- transaction_date (DATE): order transaction date (YYYY-MM-DD)
- qty_sold (INT): units sold 
- valid_order (INT): valid order flag (0,1)
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- account_type (VARCHAR): account type (Academic, Non Academic - Hospital, Non Academic - Clinic)
- child_id (VARCHAR): unique site/child account identifier
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- region (VARCHAR): demand region (e.g., Central, West, East)
- child_name (VARCHAR): full/legal name of the site (child) account
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
- is_business_day (INT): business day label (0,1)

Table: forecast — monthly budget/forecast dataset
- transaction_type (VARCHAR): Transaction Type scope for the forecast COM->Commercial, COM+PAP -> Commercial with PAP (COM, COM+PAP)
- month_year (VARCHAR): month label (e.g., 2026-01)
- budget (INT): forecasted/budgeted demand (units) for the month
- number_of_business_days_month (INT): count of business days in that month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue -  Net sales and Gross Sales at shipment date level
- qty_sold (INT): units sold in that month
- shipment_date (DATE): Date at which the drug was shiped (YYYY-MM-DD)
- valid_order (INT): valid order flag (0,1)
- short_wholesaler_name (VARCHAR): Name of the wholesaler 
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales (DECIMAL): actual net sales at shipment level
- gross_sales (DECIMAL): actual net sales at shipment level
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: revenue_forecast — monthly budget/forecast for net sales and gross sales dataset
- date (DATE): date in the format (YYYY-MM-DD)
- week_end_date (DATE): week end date ending at Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2026-01)
- net_sales_forecast (DECIMAL): forecasted/budgeted net sales (dollars) for the month
- gross_sales_forecast (DECIMAL): forecasted/budgeted gross sales (dollars) for the month
- quarter_year (VARCHAR): quarter label (e.g., 2026-Q1)
- year (INT): forecast year (e.g., 2026)

Table: mtor_alerts - HCP-to-account mapping table linking mTOR therapy prescribers (NPI, specialty, therapy start date) to their practice location and corresponding child/parent account hierarchy.
- mtor_therapy (VARCHAR): mtor therapy name
- therapy_start_date (DATE): start date of the mTOR therapy
- hcp_name (VARCHAR): name of the hcp
- npi (INTEGER): HCP NPI Value
- hcp_speciality (VARCHAR): Speciality of the HCP
- region (VARCHAR): demand region (e.g., Central, West, East)
- parent_name (VARCHAR): parent organization name
- parent_id (VARCHAR): unique parent account identifier
- address (VARCHAR): ship-to street address
- city (VARCHAR): ship-to city
- state (VARCHAR): ship-to state
- zip (VARCHAR): ship-to ZIP code
- child_id (VARCHAR): unique site/child account identifier
- child_name (VARCHAR): full/legal name of the site (child) account
- week_end_date (DATE): week ending Friday (YYYY-MM-DD)
- month_year (VARCHAR): month label (e.g., 2024-08)
- quarter_year (VARCHAR): quarter label (e.g., 2024-Q3)
- year (INT): transaction year (e.g., 2024)
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
    result_df = run_snowflake_query(sql_generator_output)
    result_df = result_df.dropna(axis=1, how='all')
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
builder.add_node("sql_reviewer", sql_reviewer_node)
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
    config={"configurable":{"thread_id":"123459"}}
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



