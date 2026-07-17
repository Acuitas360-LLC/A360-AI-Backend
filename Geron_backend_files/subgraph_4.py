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

        prompt=f"""
    You are a Query Decomposer agent.

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
    If time periods being compared are not of equal length, normalize all metrics to a weekly average before performing comparisons.
    If the user does not specify a time period, default to the most recent 52 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    All growth metrics must be expressed in percentage (%) format.
    Merge “Others” into the IDN account type for both parent and child account type. (VERY IMPORTANT)
    Merge “Unknown” into the Community account type for both parent and child account type. (VERY IMPORTANT)
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    Always append “%” to all growth and contribution values in the output.
    While calculating Parent and Child Account BC Potential Segment metrics, merge “Others” and “Unknown” into “Low” before computation.
    All child entities roll up to their respective parent entities.
    If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
    For time logic, use week_end_date for all weekly calculations; for all other time-based calculations use shipt_dt (with month_year and quarter_year as its derived references when needed).
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    If the question asks for child-level information, use only child entities (do not aggregate or substitute with parent entities).
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    If the user does not specify a time window, default to 52 weeks and label the metric accordingly (e.g., sales_52w).
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where qty_sold_pu > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    ────────────────────────
    TABLE INFORMATION
    ────────────────────────
    Table name: drug_sales

    Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):

    - ship_to_poc_id (VARCHAR)
    → Unique Ship-To Point of Care identifier
    → Example values: '1375955', '1547153', '10192'

    - ship_to_poc_nm (VARCHAR)
    → Name of the Ship-To Point of Care
    → Example values:
    'NORWALK HOSP ONC PHS',
    'NORWALK GENERAL HOSPITAL',
    'SPECTRUM HEALTH BUTTERWORTH HOSPITAL'

    - shipt_dt (DATE)
    → Actual shipment date of the product
    → Example values: '2024-11-18', '2024-12-02', '2025-01-31'

    - qty_sold_pu (INT)
    → Quantity of product units sold
    → Example values: 1, 2, 3, 4, 15

    - ship_to_acnt_zip (VARCHAR)
    → ZIP code of the ship-to account
    → Example values: '90033', '2114', '6850', '77030'

    - child_eid (VARCHAR)
    → Unique identifier of the child account
    → Example values: '90033-40195037-1', '06850-40044395-1', '49503-40044298-1'

    - child_name (VARCHAR)
    → Child account or customer name
    → Example values:
    'NUVANCE HEALTH-NORWALK',
    'COREWELL HEALTH-GRAND RAPIDS',
    'COXHEALTH-SPRINGFIELD'

    - parent_eid (VARCHAR)
    → Unique identifier of the parent account
    → Example values: '40195037', '40171291', '40197104', '40256748', '40058122'

    - parent_name (VARCHAR)
    → Parent account name
    → Example values:
    'UNIVERSITY OF SOUTHERN CALIFORNIA',
    'NUVANCE HEALTH',
    'COREWELL HEALTH'

    - terr_id (VARCHAR)
    → Territory identifier of the territory corresponding to ship_to_acnt_zip. Territory is the most granular geography mapping 
    → Example values: 'TWB101', 'TEA104', 'TWH101'

    - terr_nm (VARCHAR)
    → Territory name corresponding to terr_id 
    → Example values: 'East Los Angeles - TWB101', 'Hartford - TEA104', 'Grand Rapids - TWH101'

    - area_id (VARCHAR)
    → Area identifier of the area corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Original values: 'TW0000', 'TE0000'

    - area_nm (VARCHAR)
    → Area name of the area_id
    → Original Values:
    'West', 'East', 'Unknown'

    - regn_id (VARCHAR)
    → Region identifier of the region corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Example values: 'TWB000', 'TEA000', 'TWH000

    - regn_nm (VARCHAR)
    → region name of regn_id 
    → Original values:
    'Atlanta', 'Eastern Coastal', 'Pacific Northwest', 'Southeast', 'Gulf Plains',
    'Texas', 'Great Lakes', 'Blue Grass', 'SoCAL', 'Atlantic Coastal',
    'North Central', 'New England', 'New York', 'Rocky Mountains', 'Unknown'

    - week_end_date (DATE)
    → Week ending date associated with the transaction shipt_dt 
    → Example values: '2024-11-22', '2024-12-06', '2025-01-31'

    - month_year (VARCHAR)
    → Month and year label corresponding to  transaction shipt_dt  
    → Example values: '24-Nov', '24-Dec', '25-Jan', '25-Feb', '24-Oct'

    - quarter_year (VARCHAR)
    → Quarter and year label  corresponding to  transaction shipt_dt  
    → Example values: 'Q4-24', 'Q3-24', 'Q2-24', 'Q1-25', 'Q2-25'

    - child_account_type (VARCHAR)
    → accoutn arechetype of child account informing whether child account belong to academic setting or community 
    → Original values:
    'Academic', 'Community', 'Unknown'

    - parent_account_type (VARCHAR)
    → account arechetype of parent account informing whether parent account belong to academic setting or community 
    → Original values:
    'Academic', 'IDN/Hospital', 'Community', 'Other', 'Unknown'

    - parent_bc_segment (VARCHAR)
    → Breast cancer potential segment of the parent account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - child_bc_segment (VARCHAR)
    → Breast cancer potential segment of the child account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - kad_name (VARCHAR)
    → Key Account Director name mapped to respective child account 
    → Example values: 'John Smith', 'Emily Carter'

    - kad_region (VARCHAR)
    → KAD Region assigned to Key Account Director 
    → Example values: 'North East', 'South Central'

    - kad_id (VARCHAR)
    → Unique Key Account Director ID
    → Example values: 'KAD001', 'KAD015'

    - kad_accnt_typ (VARCHAR)
    → Account type classification for KAD ACCOUNT. This is only used in perspective of KAD Team analysis 
    → Original values: 'Academic', 'Community', 'Unknown', 'IDN/Hospital'

    - rebated_flag (VARCHAR)
    → Indicates whether the transaction was corresponding to rebated KAD account 
    → Original values:
    'Rebated', 'Non-Rebated'

    - kad_flag (INT)
    → Indicates whether the account is managed under Key Account Director (KAD) structure
    → Example Values:
        1 , 2 , 3



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
    - If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)

    ────────────────────────
    Metric & Output Handling Rules (Must Always Be Enforced):
    ────────────────────────
    If time periods being compared are not of equal length, normalize all metrics to a weekly average before performing comparisons.
    If the user does not specify a time period, default to the most recent 52 weeks of available data.
    If the user does not explicitly specify a total sales denominator, assume overall national sales as the default denominator.
    For growth metrics, if the previous period value is 0 and the current period value is greater than 0, the growth must be reported as 100%.
    All growth metrics must be expressed in percentage (%) format.
    Merge “Others” into the IDN account type for both parent and child account type. (VERY IMPORTANT)
    Merge “Unknown” into the Community account type for both parent and child account type. (VERY IMPORTANT)
    LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
    Always append “%” to all growth and contribution values in the output.
    While calculating Parent and Child Account BC Potential Segment metrics, merge “Others” and “Unknown” into “Low” before computation.
    All child entities roll up to their respective parent entities.
    If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
    For time logic, use week_end_date for all weekly calculations; for all other time-based calculations use shipt_dt (with month_year and quarter_year as its derived references when needed).
    Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
    If the question asks for child-level information, use only child entities (do not aggregate or substitute with parent entities).
    Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
    Whenever the query references “nation,” compute the national-level metrics and include them in the output.
    All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
    If the user does not specify a time window, default to 52 weeks and label the metric accordingly (e.g., sales_52w).
    If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
    All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
    All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where qty_sold_pu > 0.
    If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
    ────────────────────────
    TABLE INFORMATION
    ────────────────────────
    Table name: drug_sales

    Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):

    - ship_to_poc_id (VARCHAR)
    → Unique Ship-To Point of Care identifier
    → Example values: '1375955', '1547153', '10192'

    - ship_to_poc_nm (VARCHAR)
    → Name of the Ship-To Point of Care
    → Example values:
    'NORWALK HOSP ONC PHS',
    'NORWALK GENERAL HOSPITAL',
    'SPECTRUM HEALTH BUTTERWORTH HOSPITAL'

    - shipt_dt (DATE)
    → Actual shipment date of the product
    → Example values: '2024-11-18', '2024-12-02', '2025-01-31'

    - qty_sold_pu (INT)
    → Quantity of product units sold
    → Example values: 1, 2, 3, 4, 15

    - ship_to_acnt_zip (VARCHAR)
    → ZIP code of the ship-to account
    → Example values: '90033', '2114', '6850', '77030'

    - child_eid (VARCHAR)
    → Unique identifier of the child account
    → Example values: '90033-40195037-1', '06850-40044395-1', '49503-40044298-1'

    - child_name (VARCHAR)
    → Child account or customer name
    → Example values:
    'NUVANCE HEALTH-NORWALK',
    'COREWELL HEALTH-GRAND RAPIDS',
    'COXHEALTH-SPRINGFIELD'

    - parent_eid (VARCHAR)
    → Unique identifier of the parent account
    → Example values: '40195037', '40171291', '40197104', '40256748', '40058122'

    - parent_name (VARCHAR)
    → Parent account name
    → Example values:
    'UNIVERSITY OF SOUTHERN CALIFORNIA',
    'NUVANCE HEALTH',
    'COREWELL HEALTH'

    - terr_id (VARCHAR)
    → Territory identifier of the territory corresponding to ship_to_acnt_zip. Territory is the most granular geography mapping 
    → Example values: 'TWB101', 'TEA104', 'TWH101'

    - terr_nm (VARCHAR)
    → Territory name corresponding to terr_id 
    → Example values: 'East Los Angeles - TWB101', 'Hartford - TEA104', 'Grand Rapids - TWH101'

    - area_id (VARCHAR)
    → Area identifier of the area corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Original values: 'TW0000', 'TE0000'

    - area_nm (VARCHAR)
    → Area name of the area_id 
    → Original Values:
    'West', 'East', 'Unknown'

    - regn_id (VARCHAR)
    → Region identifier of the region corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Example values: 'TWB000', 'TEA000', 'TWH000

    - regn_nm (VARCHAR)
    → region name of regn_id 
    → Original values:
    'Atlanta', 'Eastern Coastal', 'Pacific Northwest', 'Southeast', 'Gulf Plains',
    'Texas', 'Great Lakes', 'Blue Grass', 'SoCAL', 'Atlantic Coastal',
    'North Central', 'New England', 'New York', 'Rocky Mountains', 'Unknown'

    - week_end_date (DATE)
    → Week ending date associated with the transaction shipt_dt 
    → Example values: '2024-11-22', '2024-12-06', '2025-01-31'

    - month_year (VARCHAR)
    → Month and year label corresponding to  transaction shipt_dt  
    → Example values: '24-Nov', '24-Dec', '25-Jan', '25-Feb', '24-Oct'

    - quarter_year (VARCHAR)
    → Quarter and year label  corresponding to  transaction shipt_dt  
    → Example values: 'Q4-24', 'Q3-24', 'Q2-24', 'Q1-25', 'Q2-25'

    - child_account_type (VARCHAR)
    → accoutn arechetype of child account informing whether child account belong to academic setting or community 
    → Original values:
    'Academic', 'Community', 'Unknown'

    - parent_account_type (VARCHAR)
    → account arechetype of parent account informing whether parent account belong to academic setting or community 
    → Original values:
    'Academic', 'IDN/Hospital', 'Community', 'Other', 'Unknown'

    - parent_bc_segment (VARCHAR)
    → Breast cancer potential segment of the parent account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - child_bc_segment (VARCHAR)
    → Breast cancer potential segment of the child account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - kad_name (VARCHAR)
    → Key Account Director name mapped to respective child account 
    → Example values: 'John Smith', 'Emily Carter'

    - kad_region (VARCHAR)
    → KAD Region assigned to Key Account Director 
    → Example values: 'North East', 'South Central'

    - kad_id (VARCHAR)
    → Unique Key Account Director ID
    → Example values: 'KAD001', 'KAD015'

    - kad_accnt_typ (VARCHAR)
    → Account type classification for KAD ACCOUNT. This is only used in perspective of KAD Team analysis 
    → Original values: 'Academic', 'Community', 'Unknown', 'IDN/Hospital'

    - rebated_flag (VARCHAR)
    → Indicates whether the transaction was corresponding to rebated KAD account 
    → Original values:
    'Rebated', 'Non-Rebated'

    - kad_flag (INT)
    → Indicates whether the account is managed under Key Account Director (KAD) structure
    → Example Values:
        1 , 2 , 3

        
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

    You are an expert MySQL SQL Generator.

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
- If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
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
Merge “Others” into the IDN/Hospital account type for both parent and child account type. (VERY IMPORTANT)
Merge “Unknown” into the Community account type for both parent and child account type. (VERY IMPORTANT)
LTD = Launch to Date; YTD = Year to Date; MTD = Month to Date; QTD = Quarter to Date.
Always append “%” to all growth and contribution values in the outpSut.
While calculating Parent Account and Child BC Potential Segment metrics, merge “Others” and “Unknown” into “Low” before computation.
All child entities roll up to their respective parent entities.
If the user does not explicitly specify child or parent level, default all queries and aggregations to the parent entity level. (VERY IMPORTANT)
For time logic, use week_end_date for all weekly calculations; for all other time-based calculations use shipt_dt (with month_year and quarter_year as its derived references when needed).
Time windows: R13W = Recent 13 Weeks, P13W = Prior 13 Weeks, R12M (Recent 12 Months) and P12M (Prior 12 Months) must be calculated using a rolling 52-week period.
If the question asks for child-level information, use only child entities (do not aggregate or substitute with parent entities).
Always accompany any growth metric or percentage value with the corresponding absolute volume sales value.
Whenever the query references “nation,” compute the national-level metrics and include them in the output.
All output metrics must include the time window in their label (e.g., sales_4w, sales_52w, growth_12m).
If the user does not specify a time window, default to 52 weeks and label the metric accordingly (e.g., sales_52w).
If the user asks for growth without specifying a timeframe, compute growth as Recent 13 Weeks (R13W) vs Prior 13 Weeks (P13W).
All computations related to growth, decline, and weekly averages must be converted to integers (no decimals) before reporting the results.
All computations related to account ordering status (new accounts, active accounts, total accounts) must consider only records where qty_sold_pu > 0.
If the user refers to sudden behavior, spike, drop, anomaly, or similar wording, perform the analysis using a 4-week time window.
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
    Table name: drug_sales

    Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):

    - ship_to_poc_id (VARCHAR)
    → Unique Ship-To Point of Care identifier
    → Example values: '1375955', '1547153', '10192'

    - ship_to_poc_nm (VARCHAR)
    → Name of the Ship-To Point of Care
    → Example values:
    'NORWALK HOSP ONC PHS',
    'NORWALK GENERAL HOSPITAL',
    'SPECTRUM HEALTH BUTTERWORTH HOSPITAL'

    - shipt_dt (DATE)
    → Actual shipment date of the product
    → Example values: '2024-11-18', '2024-12-02', '2025-01-31'

    - qty_sold_pu (INT)
    → Quantity of product units sold
    → Example values: 1, 2, 3, 4, 15

    - ship_to_acnt_zip (VARCHAR)
    → ZIP code of the ship-to account
    → Example values: '90033', '2114', '6850', '77030'

    - child_eid (VARCHAR)
    → Unique identifier of the child account
    → Example values: '90033-40195037-1', '06850-40044395-1', '49503-40044298-1'

    - child_name (VARCHAR)
    → Child account or customer name
    → Example values:
    'NUVANCE HEALTH-NORWALK',
    'COREWELL HEALTH-GRAND RAPIDS',
    'COXHEALTH-SPRINGFIELD'

    - parent_eid (VARCHAR)
    → Unique identifier of the parent account
    → Example values: '40195037', '40171291', '40197104', '40256748', '40058122'

    - parent_name (VARCHAR)
    → Parent account name
    → Example values:
    'UNIVERSITY OF SOUTHERN CALIFORNIA',
    'NUVANCE HEALTH',
    'COREWELL HEALTH'

    - terr_id (VARCHAR)
    → Territory identifier of the territory corresponding to ship_to_acnt_zip. Territory is the most granular geography mapping 
    → Example values: 'TWB101', 'TEA104', 'TWH101'

    - terr_nm (VARCHAR)
    → Territory name corresponding to terr_id 
    → Example values: 'East Los Angeles - TWB101', 'Hartford - TEA104', 'Grand Rapids - TWH101'

    - area_id (VARCHAR)
    → Area identifier of the area corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Original values: 'TW0000', 'TE0000'

    - area_nm (VARCHAR)
    → Area name of the area_id 
    → Original Values:
    'West', 'East', 'Unknown'

    - regn_id (VARCHAR)
    → Region identifier of the region corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Example values: 'TWB000', 'TEA000', 'TWH000

    - regn_nm (VARCHAR)
    → region name of regn_id 
    → Original values:
    'Atlanta', 'Eastern Coastal', 'Pacific Northwest', 'Southeast', 'Gulf Plains',
    'Texas', 'Great Lakes', 'Blue Grass', 'SoCAL', 'Atlantic Coastal',
    'North Central', 'New England', 'New York', 'Rocky Mountains', 'Unknown'

    - week_end_date (DATE)
    → Week ending date associated with the transaction shipt_dt 
    → Example values: '2024-11-22', '2024-12-06', '2025-01-31'

    - month_year (VARCHAR)
    → Month and year label corresponding to  transaction shipt_dt  
    → Example values: '24-Nov', '24-Dec', '25-Jan', '25-Feb', '24-Oct'

    - quarter_year (VARCHAR)
    → Quarter and year label  corresponding to  transaction shipt_dt  
    → Example values: 'Q4-24', 'Q3-24', 'Q2-24', 'Q1-25', 'Q2-25'

    - child_account_type (VARCHAR)
    → accoutn arechetype of child account informing whether child account belong to academic setting or community 
    → Original values:
    'Academic', 'Community', 'Unknown'

    - parent_account_type (VARCHAR)
    → account arechetype of parent account informing whether parent account belong to academic setting or community 
    → Original values:
    'Academic', 'IDN/Hospital', 'Community', 'Other', 'Unknown'

    - parent_bc_segment (VARCHAR)
    → Breast cancer potential segment of the parent account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - child_bc_segment (VARCHAR)
    → Breast cancer potential segment of the child account
    → Original values:
    'High', 'Medium', 'Low', 'Unknown'

    - kad_name (VARCHAR)
    → Key Account Director name mapped to respective child account 
    → Example values: 'John Smith', 'Emily Carter'

    - kad_region (VARCHAR)
    → KAD Region assigned to Key Account Director 
    → Example values: 'North East', 'South Central'

    - kad_id (VARCHAR)
    → Unique Key Account Director ID
    → Example values: 'KAD001', 'KAD015'

    - kad_accnt_typ (VARCHAR)
    → Account type classification for KAD ACCOUNT. This is only used in perspective of KAD Team analysis 
    → Original values: 'Academic', 'Community', 'Unknown', 'IDN/Hospital'

    - rebated_flag (VARCHAR)
    → Indicates whether the transaction was corresponding to rebated KAD account 
    → Original values:
    'Rebated', 'Non-Rebated'

    - kad_flag (INT)
    → Indicates whether the account is managed under Key Account Director (KAD) structure
    → Example Values:
        1 , 2 , 3

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
    Table name: drug_sales

    Columns, meanings, Original Values (Only those values exist in the table) and example values (values other than that can exist in the table) (from actual data):

    - ship_to_poc_id (VARCHAR)
    → Unique Ship-To Point of Care identifier
    

    - ship_to_poc_nm (VARCHAR)
    → Name of the Ship-To Point of Care
    

    - shipt_dt (DATE)
    → Actual shipment date of the product
    → Example values: '2024-11-18', '2024-12-02', '2025-01-31'

    - qty_sold_pu (INT)
    → Quantity of product units sold
    → Example values: 1, 2, 3, 4, 15

    - ship_to_acnt_zip (VARCHAR)
    → ZIP code of the ship-to account
    

    - child_eid (VARCHAR)
    → Unique identifier of the child account
    

    - child_name (VARCHAR)
    → Child account or customer name
    

    - parent_eid (VARCHAR)
    → Unique identifier of the parent account
    

    - parent_name (VARCHAR)
    → Parent account name
   

    - terr_id (VARCHAR)
    → Territory identifier of the territory corresponding to ship_to_acnt_zip. Territory is the most granular geography mapping 
    

    - terr_nm (VARCHAR)
    → Territory name corresponding to terr_id 
   

    - area_id (VARCHAR)
    → Area identifier of the area corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    → Example values: 'TW0000', 'TE0000'

    - area_nm (VARCHAR)
    → Area name of the area_id 
    → Example Values:
    'West', 'East', 'Unknown'

    - regn_id (VARCHAR)
    → Region identifier of the region corresponding to ship_to_acnt_zip. zip codes roll up to Territory and territories to region and region to areas 
    

    - regn_nm (VARCHAR)
    → region name of regn_id 
    → Example values:
    'Atlanta', 'Eastern Coastal', 'Pacific Northwest', 'Southeast', 'Gulf Plains',
    'Texas', 'Great Lakes', 'Blue Grass', 'SoCAL', 'Atlantic Coastal',
    'North Central', 'New England', 'New York', 'Rocky Mountains', 'Unknown'

    - week_end_date (DATE)
    → Week ending date associated with the transaction shipt_dt 
    → Example values: '2024-11-22', '2024-12-06', '2025-01-31'

    - month_year (VARCHAR)
    → Month and year label corresponding to  transaction shipt_dt  
    → Example values: '24-Nov', '24-Dec', '25-Jan', '25-Feb', '24-Oct'

    - quarter_year (VARCHAR)
    → Quarter and year label  corresponding to  transaction shipt_dt  
    → Example values: 'Q4-24', 'Q3-24', 'Q2-24', 'Q1-25', 'Q2-25'

    - child_account_type (VARCHAR)
    → accoutn arechetype of child account informing whether child account belong to academic setting or community 
    → Example values:
    'Academic', 'Community', 'Unknown'

    - parent_account_type (VARCHAR)
    → account arechetype of parent account informing whether parent account belong to academic setting or community 
    → Example values:
    'Academic', 'IDN/Hospital', 'Community', 'Other', 'Unknown'

    - parent_bc_segment (VARCHAR)
    → Breast cancer potential segment of the parent account
    → Example values:
    'High', 'Medium', 'Low', 'Unknown'

    - child_bc_segment (VARCHAR)
    → Breast cancer potential segment of the child account
    → Example values:
    'High', 'Medium', 'Low', 'Unknown'

    - kad_name (VARCHAR)
    → Key Account Director name mapped to respective child account 
   

    - kad_region (VARCHAR)
    → KAD Region assigned to Key Account Director 
   

    - kad_id (VARCHAR)
    → Unique Key Account Director ID
    

    - kad_accnt_typ (VARCHAR)
    → Account type classification for KAD ACCOUNT. This is only used in perspective of KAD Team analysis 
    → Example values: 'Academic', 'Community', 'Unknown', 'IDN/Hospital'

    - rebated_flag (VARCHAR)
    → Indicates whether the transaction was corresponding to rebated KAD account 
    → Example values:
    'Rebated', 'Non-Rebated'

    - kad_flag (INT)
    → Indicates whether the account is managed under Key Account Director (KAD) structure
    → Example Values:
        1 , 2 , 3

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
    result_df = run_mysql_query(sql_generator_output)
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

    Output

    Return either:

    Python code defining fig

    NO_VISUALIZATION (only when visualization is impossible)

"""
    response=model.invoke(prompt).content
    print("Visualization Code")
    print("-"*100)
    print(response)
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
    append_agent_trace("agent_trace.json", user_input, result["trace"])



