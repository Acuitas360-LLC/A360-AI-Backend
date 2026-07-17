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

model=ChatOpenAI(model='gpt-5.4')
model_1=ChatOpenAI(model='gpt-5.3-codex')
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

# def get_clean_recent_turns(messages: List, n_turns: int = 3):
#     """
#     Returns last n conversation turns in compact form.

#     Keeps:
#     - HumanMessage (full)
#     - AIMessage with only:
#         * SQL Query Executed
#         * Result Summary

#     Removes:
#     - Query Results
#     - Visualization code
#     - Relevant Questions
#     """

#     turns = []
#     current_turn = []

#     # -------- Step 1: Build turns (reverse traversal) --------
#     for msg in reversed(messages):
#         current_turn.insert(0, msg)

#         if isinstance(msg, HumanMessage):
#             turns.insert(0, current_turn)
#             current_turn = []

#             if len(turns) == n_turns:
#                 break

#     # -------- Step 2: Clean AI messages --------
#     cleaned_messages = []

#     for turn in turns:
#         for msg in turn:

#             # Keep human messages fully
#             if isinstance(msg, HumanMessage):
#                 cleaned_messages.append(msg)

#             elif isinstance(msg, AIMessage):
#                 content = msg.content or ""

#                 sql_part = ""
#                 summary_part = ""

#                 # Extract SQL
#                 if "SQL Query Executed:" in content:
#                     sql_part = content.split("SQL Query Executed:")[-1]

#                     # Stop at next section
#                     for stop in ["Result Summary:", "Relevant Questions:", "Query Results:", "Visualization Code:"]:
#                         if stop in sql_part:
#                             sql_part = sql_part.split(stop)[0]
#                             break

#                     sql_part = sql_part.strip()

#                 # Extract Summary
#                 if "Result Summary:" in content:
#                     summary_part = content.split("Result Summary:")[-1]

#                     for stop in ["Relevant Questions:", "Query Results:", "Visualization Code:"]:
#                         if stop in summary_part:
#                             summary_part = summary_part.split(stop)[0]
#                             break

#                     summary_part = summary_part.strip()

#                 # Build cleaned AI message if something exists
#                 cleaned_content = ""

#                 if sql_part:
#                     cleaned_content += "SQL Query Executed:\n" + sql_part + "\n\n"

#                 if summary_part:
#                     cleaned_content += "Result Summary:\n" + summary_part

#                 if cleaned_content:
#                     cleaned_messages.append(
#                         AIMessage(content=cleaned_content.strip())
#                     )

#     return cleaned_messages

from typing import List
from langchain_core.messages import HumanMessage, AIMessage


def get_clean_recent_turns(messages: List, n_turns: int = 3):
    """
    Returns last n conversation turns in compact form.

    Keeps:
    - HumanMessage (full)
    - AIMessage with only:
        * SQL Query Executed
        * Result Summary
    - SQL Results (structured, from additional_kwargs)

    Removes:
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

    # -------- Step 2: Clean messages --------
    cleaned_messages = []

    for turn in turns:
        for msg in turn:

            # ✅ 1. Keep Human messages fully
            if isinstance(msg, HumanMessage):
                cleaned_messages.append(msg)
                continue

            elif isinstance(msg, AIMessage):

                # 🔥 Safe extraction of additional_kwargs
                kwargs = getattr(msg, "additional_kwargs", {}) or {}
                print("kwargs: ",kwargs)
                msg_type = kwargs.get("type")
                print("msg_type: ",msg_type)

                # ✅ 2. Preserve SQL Results (PRIMARY: kwargs, FALLBACK: content)
                if (
                    msg_type == "sql_result"
                    or (msg.content and msg.content.lower().strip() == "sql query results")
                ):
                    cleaned_messages.append(
                        AIMessage(
                            content="SQL Query Results",
                            additional_kwargs={
                                "type": "sql_result",
                                "data": kwargs.get("data")
                            }
                        )
                    )
                    continue

                # 🚫 Skip visualization messages
                if msg_type == "visualization":
                    continue

                content = msg.content or ""

                sql_part = ""
                summary_part = ""

                # -------- Extract SQL --------
                if "SQL Query Executed:" in content:
                    sql_part = content.split("SQL Query Executed:")[-1]

                    for stop in [
                        "Result Summary:",
                        "Relevant Questions:",
                        "Query Results:",
                        "Visualization Code:"
                    ]:
                        if stop in sql_part:
                            sql_part = sql_part.split(stop)[0]
                            break

                    sql_part = sql_part.strip()

                # -------- Extract Summary --------
                if "Result Summary:" in content:
                    summary_part = content.split("Result Summary:")[-1]

                    for stop in [
                        "Relevant Questions:",
                        "Query Results:",
                        "Visualization Code:"
                    ]:
                        if stop in summary_part:
                            summary_part = summary_part.split(stop)[0]
                            break

                    summary_part = summary_part.strip()

                # -------- Build cleaned AI message --------
                cleaned_content = ""

                if sql_part:
                    cleaned_content += "SQL Query Executed:\n" + sql_part + "\n\n"

                if summary_part:
                    cleaned_content += "Result Summary:\n" + summary_part

                if cleaned_content.strip():
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
    recent_context_prompt=f"""
────────────────────────
RECENT CONVERSATION CONTEXT (CRITICAL)
────────────────────────
The following messages represent the MOST RECENT conversation history.

They contain previously computed:

* Entities (e.g., accounts, regions, products, tiers, segments)
* Filters (e.g., date ranges, time periods, conditions)
* Metrics (e.g., sales, growth, aggregates)
* SQL queries and their results

You MUST treat this as ACTIVE CONTEXT.

Rules:

1. Anchor to Recent Context:

   * Always use this context to interpret the current request.
   * Prefer the MOST RECENT relevant information.

2. Reference Resolution:

   * Terms like "those", "them", "same", "above", "previous", "that"
     MUST be resolved using this context.

3. Do NOT Recompute:

   * If entities, filters, or results already exist, reuse them.
   * Do NOT generate new values if they can be derived from context.

4. Maintain Continuity:

   * Preserve the same entities, filters, and grouping unless explicitly changed.
   * Apply only incremental changes requested by the user.

5. Trust Data Over Assumptions:

   * Prefer SQL results and explicit values over inferred logic.

"""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=recent_context_prompt),
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
    Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics with total Volume Sales., where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level) (VERY IMPORTANT).
    All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, territory, or segmented data.
    If the user does not specify a time period, default to the most recent 13 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    All campus entities roll up to their respective parent entities.
    All sales-related outputs must include the base metric (e.g., mg or sls) as a clear prefix or suffix in the field name
    If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
    For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. 
    For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total Sales and Daily Average Sales, where Daily Average = Total Sales / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
    Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share. For queries focused solely on relmora, use relmora_total_mg.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    For relmora vs zynava growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    Our Product is Relmora and Competitor product is zynava.
    For our product vs competitor growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    For every time period in the output, explicitly display the corresponding number of business days
    Always perform aggregations using ID fields (e.g., campus_id, parent_id, region_id, territory_id) for accuracy, and include the corresponding names in the final output.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
    If the user asks for any target campuses–related information (such as campus region, campus tier, territory, or any attribute belonging to target campuses), then:
        Refer to the target_campuses table for those attributes.
        Always perform a join between data_867 and target_campuses if the user query requires analysis of sales of target campuses.
        Use the appropriate common identifier (e.g., campus_id) to join the tables.
        Ensure the final result includes the requested information from target_campuses along with relevant data from data_867.
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

Total sales vs Total sales
Daily average vs Daily average

Never compare **daily averages for one period with total sales for the other period**.

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
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume sales and total National Sales.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume sales and total National Sales.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---

For data_867: Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.

data_DDD Rules (Competitor & Market Share Analysis or share in general) (VERY IMPORTANT):

- Use relmora_total_sls and zynava_total_sls for all calculations.
- date is the month-end field and the primary time anchor.
- Set the window start date to the first day of the month and the end date to the last day of the month.
- Always display metrics for relmora, zynava, and the total.
- Format all Market Share Percentage values to one decimal place in the final output.

Time Handling:
- Always derive time ranges using date.
- Use month_year or quarter_year only when explicitly requested, but still derive boundaries from date.
- If no timeframe is specified → use last 3 months.
- For “current/recent/last” → determine using MAX(date).

Growth & Comparison:
- Default growth = R3M vs P3M.
- If periods are unequal → normalize to monthly averages.

Anomaly Detection:
- For spike/drop/anomaly → use 1-month window.

Granularity:
- Data is monthly (each row = month-end snapshot).
- Do not treat as daily data.

Restrictions:
- No week_end_date → no weekly calculations or averages.
- Avoid unnecessary averaging unless required for normalization.

Key Principle:
- date is the single source of truth for all time logic.


TABLE SCHEMA:

Table: data_867 — transaction-level sales dataset (weekly + campus-level analysis)
- date (DATE): transaction date (YYYY-MM-DD)
- week_end_date (DATE): week ending Friday
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- is_business_day (INT): Business Day Label (0,1)
- relmora_total_mg (INT): relmora sales volume (mg)
- relmora_total_sls (DECIMAL): relmora sales value
- campus_id (VARCHAR): unique campus identifier
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP code
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- parent_id (VARCHAR): parent account ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP
- campus_calls (INT): number of sales interactions

Table: data_DDD — competitor + market share dataset (monthly, month-end anchored)
- date (DATE): month-end date (PRIMARY time anchor)
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- relmora_total_mg (INT): relmora sales volume (mg)
- zynava_total_mg (INT): competitor (zynava) volume
- relmora_total_sls (DECIMAL): relmora sales value
- zynava_total_sls (DECIMAL): zynava sales value
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (West, Great Lakes, North East, South East, Central)
- parent_id (VARCHAR): parent ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP

Table: target_campuses — prioritized campuses for strategic focus
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_city (VARCHAR): city
- campus_state (VARCHAR): state
- campus_zip (VARCHAR): ZIP
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- campus_territory (VARCHAR): territory
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_calls (INT): number of calls


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
CONTEXT CONTINUITY (CRITICAL)
────────────────────────
Recent messages contain previously computed:

Entities (e.g., accounts, regions, products, tiers, segments)
Filters (e.g., date ranges, time periods, conditions)
Metrics (e.g., sales, growth, aggregates)
Grouping and aggregation logic

You MUST treat them as ACTIVE CONTEXT.

Rules:

Reference Resolution:
If the user refers to:
"those", "them", "same", "above", "previous", "that"
→ Resolve using the MOST RECENT relevant context.
Entity & Filter Reuse:
NEVER regenerate entities or filters if they already exist.
ALWAYS reuse exact values from prior results when available.
Continuity Enforcement:
Maintain SAME entities
Maintain SAME filters (unless explicitly changed)
Maintain SAME grouping level and granularity
Incremental Changes Only:
If the user asks for a modification (e.g., growth, comparison, breakdown),
apply ONLY that change on top of existing context.
Do NOT recompute from scratch unless explicitly asked.
Source of Truth Priority:
Resolve context using:
(1) SQL Query Results (highest priority)
(2) Explicit values mentioned in prior messages
(3) SQL Query logic

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
    Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics with total Volume Sales. where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level)(VERY IMPORTANT).
    All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, territory, or segmented data.
    If the user does not specify a time period, default to the most recent 13 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    All campus entities roll up to their respective parent entities.
    All sales-related outputs must include the base metric (e.g., mg or sls) as a clear prefix or suffix in the field name
    If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
    For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. 
    For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total Sales and Daily Average Sales, where Daily Average = Total Sales / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
    Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share. For queries focused solely on relmora, use relmora_total_mg.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    For relmora vs zynava growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Our Product is Relmora and Competitor product is zynava.
    For our product vs competitor growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    For every time period in the output, explicitly display the corresponding number of business days
    Always perform aggregations using ID fields (e.g., campus_id, parent_id, region_id, territory_id) for accuracy, and include the corresponding names in the final output.
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
    If the user asks for any target campuses–related information (such as campus region, campus tier, territory, or any attribute belonging to target campuses), then:
        Refer to the target_campuses table for those attributes.
        Always perform a join between data_867 and target_campuses if the user query requires analysis of sales of target campuses.
        Use the appropriate common identifier (e.g., campus_id) to join the tables.
        Ensure the final result includes the requested information from target_campuses along with relevant data from data_867.
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

Total sales vs Total sales
Daily average vs Daily average

Never compare **daily averages for one period with total sales for the other period**.

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
END AS total_growth
4. If both periods are complete → aggregate totals at period level and display total growth also calculate daily average at period level and display daily average growth. 
5. If ANY period is incomplete -> you MUST NOT generate, compute, or include total growth or national total growth in the output schema itself. These fields must be completely omitted (not NULL, not blank). ONLY include daily average growth and national daily average growth. This rule strictly overrides all other instructions.
Metric Visibility Rule (MANDATORY):
- If BOTH periods are complete → display BOTH total volume growth and daily average growth (including national metrics).
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume sales and total National Sales.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume sales and total National Sales.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---

For data_867: Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.

data_DDD Rules (Competitor & Market Share Analysis or share in general) (VERY IMPORTANT):

- Use relmora_total_sls and zynava_total_sls for all calculations.
- date is the month-end field and the primary time anchor.
- Set the window start date to the first day of the month and the end date to the last day of the month.
- Always display metrics for relmora, zynava, and the total.
- Format all Market Share Percentage values to one decimal place in the final output.

Time Handling:
- Always derive time ranges using date.
- Use month_year or quarter_year only when explicitly requested, but still derive boundaries from date.
- If no timeframe is specified → use last 3 months.
- For “current/recent/last” → determine using MAX(date).

Growth & Comparison:
- Default growth = R3M vs P3M.
- If periods are unequal → normalize to monthly averages.

Anomaly Detection:
- For spike/drop/anomaly → use 1-month window.

Granularity:
- Data is monthly (each row = month-end snapshot).
- Do not treat as daily data.

Restrictions:
- No week_end_date → no weekly calculations or averages.
- Avoid unnecessary averaging unless required for normalization.

Key Principle:
- date is the single source of truth for all time logic.


TABLE SCHEMA:

Table: data_867 — transaction-level sales dataset (weekly + campus-level analysis)
- date (DATE): transaction date (YYYY-MM-DD)
- week_end_date (DATE): week ending Friday
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- is_business_day (INT): Business Day Label (0,1)
- relmora_total_mg (INT): relmora sales volume (mg)
- relmora_total_sls (DECIMAL): relmora sales value
- campus_id (VARCHAR): unique campus identifier
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP code
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- parent_id (VARCHAR): parent account ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP
- campus_calls (INT): number of sales interactions

Table: data_DDD — competitor + market share dataset (monthly, month-end anchored)
- date (DATE): month-end date (PRIMARY time anchor)
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- relmora_total_mg (INT): relmora sales volume (mg)
- zynava_total_mg (INT): competitor (zynava) volume
- relmora_total_sls (DECIMAL): relmora sales value
- zynava_total_sls (DECIMAL): zynava sales value
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (West, Great Lakes, North East, South East, Central)
- parent_id (VARCHAR): parent ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP


Table: target_campuses — prioritized campuses for strategic focus
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_city (VARCHAR): city
- campus_state (VARCHAR): state
- campus_zip (VARCHAR): ZIP
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- campus_territory (VARCHAR): territory
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_calls (INT): number of calls
        
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
All growth metrics must be expressed in percentage (%) format.
Always append “%” to all growth and contribution values in the output.
All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.

Display both period-level metrics and daily average metrics ONLY when the period is complete. If the period is incomplete, display only daily average metrics, where Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END computed at NATIONAL level)(VERY IMPORTANT).
All business day calculations MUST be performed strictly at the national level only, and must NEVER be derived from any regional, tier, territory, or segmented data.
If the user does not specify a time period, default to the most recent 13 weeks of available data.
If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
All campus entities roll up to their respective parent entities.
All sales-related outputs must include the base metric (e.g., mg or sls) as a clear prefix or suffix in the field name
If the user does not explicitly specify campus or parent level, default all queries and aggregations to the campus entity level. (VERY IMPORTANT)
For data_867: The table contains week_end_date, month_year, and quarter_year. Use week_end_date for weekly calculations. 
    For data_DDD: The date column represents the month-end shipment date and the table does not contain week_end_date. Therefore, use date (month-end) for time-based calculations, with month_year and quarter_year used when monthly or quarterly aggregation is required.
    For a specific month or quarter queries, filter using `month_year` or `quarter_year` respectively. Calculate Total Sales and Daily Average Sales, where Daily Average = Total Sales / SUM(is_business_day). Always display both metrics.
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    When the aggregation is based on a specific time granularity, the metric name should reflect it explicitly (e.g., weekly_sales, monthly_sales, quarterly_sales, yearly_sales) and should not include an additional time window prefix or suffix.
    Column Selection Rule: Use relmora_total_sls and zynava_total_sls only when the query explicitly asks about market share. For queries focused solely on relmora, use relmora_total_mg.
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where relmora_total_mg > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    For relmora vs zynava growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    Always determine the latest time period using week_end_date or date, not MAX(quarter_year) or MAX(month_year), because quarter_year and month_year are strings and may not sort chronologically. Retrieve the corresponding quarter_year or month_year from the row with the latest week_end_date or date.
    Whenever a user asks about performance, always calculate and include the growth (percentage change vs the previous comparable period)
    Our Product is Relmora and Competitor product is zynava.
    For our product vs competitor growth comparisons, use `data_867.relmora_total_mg` and `data_ddd.zynava_total_mg`. Compare recent 3 months vs previous 3 months, anchored to the latest `month_year` from `data_ddd`, and apply the same months to `data_867`.
    For every time period in the output, explicitly display the corresponding number of business days
    Always perform aggregations using ID fields (e.g., campus_id, parent_id, region_id, territory_id) for accuracy, and include the corresponding names in the final output.
    Whenever growth is calculated for any segmentation level (e.g., segment, tier, region, area, geography, account type, city, state, or territory), also calculate nation growth and add a column indicating whether the segment is performing Higher or Lower than the nation.
    All segment vs nation growth comparisons must be strictly based on Daily Average Growth (growth normalized by national business days), which serves as the single anchor metric for determining relative performance.
    If asked about sales by default give national sales don't group by campus_id or campus_account name.
    If the user asks for any target campuses–related information (such as campus region, campus tier, territory, or any attribute belonging to target campuses), then:
        Refer to the target_campuses table for those attributes.
        Always perform a join between data_867 and target_campuses if the user query requires analysis of sales of target campuses.
        Use the appropriate common identifier (e.g., campus_id) to join the tables.
        Ensure the final result includes the requested information from target_campuses along with relevant data from data_867.
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

Total sales vs Total sales
Daily average vs Daily average

Never compare **daily averages for one period with total sales for the other period**.

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
END AS total_growth
4. If both periods are complete → aggregate totals at period level and display growth on total volume also calculate daily average at period level and display daily average growth.
5. If ANY period is incomplete -> you MUST NOT generate, compute, or include total growth or national total growth in the output schema itself. These fields must be completely omitted (not NULL, not blank). ONLY include daily average growth and national daily average growth. This rule strictly overrides all other instructions.
Metric Visibility Rule (MANDATORY):
- If BOTH periods are complete → display BOTH total volume growth and daily average growth (including national metrics).
- If ANY period is incomplete → display ONLY daily average growth and national daily average growth along with total Volume sales and total National Sales.
- Total growth and national total growth MUST NOT be generated or included in the output schema when any period is incomplete but display total Volume sales and total National Sales.
6. Perform the comparison.
Daily Average = Total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END) and not Daily Average = Total / SUM(d.is_business_day) (VERY IMPORTANT)
For month/quarter queries, anchor to `month_year` and `quarter_year` respectively, and always include daily average metrics.

---

For data_867: Do not automatically restrict calculations to the **most recent completed period** unless the user explicitly requests it.

data_DDD Rules (Competitor & Market Share Analysis or share in general) (VERY IMPORTANT):

- Use relmora_total_sls and zynava_total_sls for all calculations.
- date is the month-end field and the primary time anchor.
- Set the window start date to the first day of the month and the end date to the last day of the month.
- Always display metrics for relmora, zynava, and the total.
- Format all Market Share Percentage values to one decimal place in the final output.

Time Handling:
- Always derive time ranges using date.
- Use month_year or quarter_year only when explicitly requested, but still derive boundaries from date.
- If no timeframe is specified → use last 3 months.
- For “current/recent/last” → determine using MAX(date).

Growth & Comparison:
- Default growth = R3M vs P3M.
- If periods are unequal → normalize to monthly averages.

Anomaly Detection:
- For spike/drop/anomaly → use 1-month window.

Granularity:
- Data is monthly (each row = month-end snapshot).
- Do not treat as daily data.

Restrictions:
- No week_end_date → no weekly calculations or averages.
- Avoid unnecessary averaging unless required for normalization.

Key Principle:
- date is the single source of truth for all time logic.



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

Table: data_867 — transaction-level sales dataset (weekly + campus-level analysis)
- date (DATE): transaction date (YYYY-MM-DD)
- week_end_date (DATE): week ending Friday
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- is_business_day (INT): Business Day Label (0,1)
- relmora_total_mg (INT): relmora sales volume (mg)
- relmora_total_sls (DECIMAL): relmora sales value
- campus_id (VARCHAR): unique campus identifier
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP code
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- parent_id (VARCHAR): parent account ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP
- campus_calls (INT): number of sales interactions

Table: data_DDD — competitor + market share dataset (monthly, month-end anchored)
- date (DATE): month-end date (PRIMARY time anchor)
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- relmora_total_mg (INT): relmora sales volume (mg)
- zynava_total_mg (INT): competitor (zynava) volume
- relmora_total_sls (DECIMAL): relmora sales value
- zynava_total_sls (DECIMAL): zynava sales value
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (West, Great Lakes, North East, South East, Central)
- parent_id (VARCHAR): parent ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP

Table: target_campuses — prioritized campuses for strategic focus
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_city (VARCHAR): city
- campus_state (VARCHAR): state
- campus_zip (VARCHAR): ZIP
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- campus_territory (VARCHAR): territory
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_calls (INT): number of calls

    {sql_generator_rag_examples_text}

    """

    response = model_1.invoke(prompt).content[0]["text"]
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

TABLE SCHEMA:

Table: data_867 — transaction-level sales dataset (weekly + campus-level analysis)
- date (DATE): transaction date (YYYY-MM-DD)
- week_end_date (DATE): week ending Friday
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- is_business_day (INT): Business Day Label (0,1)
- relmora_total_mg (INT): relmora sales volume (mg)
- relmora_total_sls (DECIMAL): relmora sales value
- campus_id (VARCHAR): unique campus identifier
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP code
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- parent_id (VARCHAR): parent account ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP
- campus_calls (INT): number of sales interactions

Table: data_DDD — competitor + market share dataset (monthly, month-end anchored)
- date (DATE): month-end date (PRIMARY time anchor)
- month_year (VARCHAR): month label (e.g., 2025-01)
- quarter_year (VARCHAR): quarter label (e.g., 2025-Q1)
- relmora_total_mg (INT): relmora sales volume (mg)
- zynava_total_mg (INT): competitor (zynava) volume
- relmora_total_sls (DECIMAL): relmora sales value
- zynava_total_sls (DECIMAL): zynava sales value
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_account_type (VARCHAR): account type (ACADEMIC, COMMUNITY)
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_city (VARCHAR): campus city
- campus_state (VARCHAR): campus state
- campus_zip (VARCHAR): campus ZIP
- campus_territory_id (VARCHAR): territory ID
- campus_territory (VARCHAR): territory name
- campus_region_id (VARCHAR): region ID
- campus_region (VARCHAR): region (West, Great Lakes, North East, South East, Central)
- parent_id (VARCHAR): parent ID
- parent_account_name (VARCHAR): parent account name
- parent_city (VARCHAR): parent city
- parent_state (VARCHAR): parent state
- parent_zip (VARCHAR): parent ZIP

Table: target_campuses — prioritized campuses for strategic focus
- campus_id (VARCHAR): campus ID
- campus_account_name (VARCHAR): campus name
- campus_city (VARCHAR): city
- campus_state (VARCHAR): state
- campus_zip (VARCHAR): ZIP
- campus_region (VARCHAR): region (Central, Great Lakes, North East, Mid Atlantic, South East)
- campus_territory (VARCHAR): territory
- campus_tier (VARCHAR): tier (Tier 1, Tier 2, Tier 3, No Tier)
- campus_calls (INT): number of calls
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
    result_df = result_df.dropna(axis=1, how='all')
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
        - Infer and explicitly state period completeness in the summary: presence of both total growth and daily average growth indicates complete periods, while presence of only daily average growth indicates at least one incomplete period, usually the most recent.

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

Your goal is to create a meaningful, accurate, and non-misleading Plotly visualization ONLY when the data supports it.

You MUST prioritize correctness over forcing a chart.

---

## INPUTS

Query Decomposer Output:
{query_decomposer_output}

SQL Executor Output:
{sql_executor_output}

Assume the SQL output will be reconstructed into a Pandas DataFrame named df.

---

## CORE DECISION LOGIC (MANDATORY)

Before generating a chart, you MUST:

1. Identify column types:

   * Numeric columns
   * Categorical columns
   * Datetime or ordered columns

2. Determine analytical intent:

   * Trend → requires datetime or ordered column
   * Comparison → categorical vs numeric
   * Distribution → single numeric column
   * Ranking → categorical + numeric
   * Relationship → at least two numeric columns

3. Validate if visualization is appropriate:

   * If only 1 column → NO_VISUALIZATION
   * If all columns are categorical → NO_VISUALIZATION
   * If data is too small, ambiguous, or lacks structure → NO_VISUALIZATION
   * If visualization would be misleading → NO_VISUALIZATION
   
4. Time Axis Rule: Use the dataset’s exact time granularity (week/month/quarter) for the X-axis—no transformations or mixing.
---

## ENHANCED INTENT DETECTION (ADDED)

In addition to the above, refine intent using semantic signals from the question:

* Trend Analysis:
  Keywords → "trend", "over time", "evolution", "recent", "momentum"

* Regional / Segment Comparison:
  Keywords → "across regions", "by tier", "comparison"

* Contribution / Drivers:
  Keywords → "driving", "contribution", "dependent", "share of"

* Consistency / Variability:
  Keywords → "consistent", "variability", "spread"

* Adoption / Funnel / Health:
  Keywords → "adoption", "funnel", "health", "status", "conversion"

* Market Share:
  Keywords → "market share", "gaining share", "losing share"

* Competitive Comparison:
  Keywords → multiple entities (e.g., relmora vs zynava)

* Multi-dimensional:
  Keywords → combinations like "region and tier"

---

## CHART SELECTION RULES (STRICT)

* Line Chart:
  Use ONLY if a datetime or ordered column exists

* Bar Chart:
  Use for categorical vs numeric comparisons

* Scatter Plot:
  Use ONLY if at least 2 numeric columns exist

* Histogram:
  Use for distribution of a single numeric column

* Pie Chart:
  Use ONLY if:

  * ≤ 6 categories
  * Represents part-to-whole relationship

* Flat/tabular outputs with no clear analytical mapping:
  Return NO_VISUALIZATION

* If multiple chart types are possible:
  Choose the simplest and most interpretable one

---

## ADVANCED CHART OVERRIDES (ADDED - HIGH PRIORITY)

These rules OVERRIDE basic rules when applicable:

1. Trend + Multiple Categories:
   → Use MULTI-LINE chart (color by category)

2. Contribution / Share:
   → Prefer STACKED BAR
   → If time present → STACKED AREA

3. Market Share:
   → ALWAYS convert to percentage if possible
   → Use:

   * STACKED AREA (time)
   * 100% STACKED BAR (snapshot)

4. Performance vs Target:
   → Prefer grouped bar (actual vs target)
   → If unclear → fallback to bar chart

5. Multi-dimensional (2 categorical variables):
   → Prefer HEATMAP (if dense data)
   → Else GROUPED BAR

6. Adoption / Health Categories:
   → STACKED BAR (if categorical states exist)

7. Consistency / Variability:
   → If enough data → BOX PLOT
   → Else fallback to bar/line

---

## GROWTH RULE (VERY IMPORTANT)

If any growth-related column exists (growth, %, change, WoW, MoM, QoQ, YoY):

* You MUST include BOTH:

  * Base metric (bar or line)
  * Growth metric (secondary axis)

* Use make_subplots with secondary_y=True

* DO NOT mix axis strategies:

  * If using secondary_y=True → use make_subplots ONLY
  * NEVER manually assign yaxis='y2'

---

## COLUMN USAGE RULES

* Use ONLY columns present in df

* NEVER invent or infer missing columns

* Preferred mappings:

  * x → categorical or datetime column
  * y → numeric column(s)

---

## VISUAL ENHANCEMENT RULES (ADDED)

When generating charts, apply:

* Sort categorical axes in descending order (for comparison charts)
* Highlight latest time point (for trend charts)
* Limit categories to top 10 if too many values
* Use consistent color grouping for categories
* Avoid clutter and over-plotting
* Ensure readability over aesthetics

---

## SPECIAL HANDLING RULES (ADDED)

* Growth queries:
  → Always prioritize showing trend + growth together

* Recent period queries:
  → Focus on latest available time window

* Regional queries:
  → Ensure comparisons are clearly distinguishable

* Market share queries:
  → Prefer percentage representation over absolute values

* Multi-level queries:
  → Prefer grouped or heatmap visualization

---

## PLOTLY RULES (MANDATORY)

* Use Plotly only (plotly.express or plotly.graph_objects)
* Output ONLY valid Python code defining `fig`
* No explanations, no comments, no markdown
* No Streamlit code

---

## LAYOUT / WIDTH RULE (CRITICAL)

* Plotly `width` MUST be a numeric value (e.g., 600, 800, 1000)
* NEVER use 'stretch' or 'content' inside fig.update_layout()
* NEVER use `use_container_width`
* The rendering layer (e.g., Streamlit) will handle container sizing

---

## HOVERTEMPLATE RULES (MANDATORY)

* NEVER use Python `%` string formatting
* ALWAYS use f-strings
* Preserve Plotly placeholders like `%{{x}}`, `%{{y}}`
* Escape placeholders in f-strings:
  Example: f"Region=%{{x}}<br>Value=%{{y}}"
* Do NOT mix `%` formatting with Plotly placeholders

---

## FAIL-SAFE (IMPORTANT)

Return NO_VISUALIZATION if:

* Data does not clearly map to a valid chart
* Columns are ambiguous or unsuitable
* Visualization would be confusing or misleading

---

## OUTPUT

Return either:

* Python code defining `fig`

OR

* NO_VISUALIZATION

"""
    response=model.invoke(prompt).content
    print("Visualization Code")
    print("-"*100)
    print(response)
    log_trace(state, "visualization_node", "TextMessage", response)
    return {
        "visualization_code":response
    }



def build_graph(checkpointer=None):
    """
    Builds and returns a compiled LangGraph graph.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("query_decomposer", query_decomposer_node)
    builder.add_node("sql_generator", sql_generator_node)
    builder.add_node("sql_reviewer", sql_reviewer_node)
    builder.add_node("sql_executor",sql_executor)
    builder.add_node("summarizer_node",summarizer_node)
    builder.add_node("visualization_node",visualization_node)
    #builder.add_node("human", human_node)
    builder.add_node("terminator", terminator_node)

    # Entry
    builder.set_entry_point("query_decomposer")

    # Edges
    builder.add_edge("query_decomposer", "sql_generator")
    builder.add_edge("sql_generator", "sql_reviewer")
    #builder.add_edge("sql_generator", "sql_executor")
    # Conditional edges
    builder.add_conditional_edges(
    "sql_reviewer",
    reviewer_router,
    {
        "sql_executor": "sql_executor",
        "query_decomposer": "query_decomposer",
    },
)

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



