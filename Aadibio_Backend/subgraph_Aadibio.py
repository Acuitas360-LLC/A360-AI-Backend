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
import pandas as pd
import snowflake.connector
import requests

def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


SNOWFLAKE_CONFIG = {
    "user": _env_or_default("SNOWFLAKE_USER", "ahusain"),
    "password": _env_or_default("SNOWFLAKE_PASSWORD", "Murtaza@40401059"),
    "account": _env_or_default("SNOWFLAKE_ACCOUNT", "ua60309.south-central-us.azure"),
    "warehouse": _env_or_default("SNOWFLAKE_WAREHOUSE", "AADIBIO_COMPUTE"),
    "database": _env_or_default("SNOWFLAKE_DATABASE", "AADIBIO_CAI"),
    "schema": _env_or_default("SNOWFLAKE_SCHEMA", "AADIBIO_CAI_SCHEMA"),
}


model=ChatOpenAI(model='gpt-5.4')
model_1=ChatOpenAI(model='gpt-5.3-codex')
model_2=ChatOpenAI(model='gpt-5-nano')
# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")
_mask_map   = {}   # original → masked
_demask_map = {}   # masked   → original  (inverted _mask_map)


def run_snowflake_query(query):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)

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

# def run_snowflake_query(query):
#     conn = snowflake.connector.connect(
#     user="greddy@geron.com",
#     account="hbc62861.us-east-1",
#     authenticator="externalbrowser",
#     warehouse="SF030_P_WHS_GERN_ZDH_GEN01",
#     database="P_GERN_030_ZDH_01",
#     schema="SELF_SERVE_ACUITAS_ANALYTICS"
# )

#     cursor = conn.cursor()
#     cursor.execute(query)

#     # Fetch data
#     data = cursor.fetchall()
#     columns = [col[0] for col in cursor.description]

#     # Convert to DataFrame
#     df = pd.DataFrame(data, columns=columns)

#     cursor.close()
#     conn.close()

#     return df

def load_masking_table_snowflake() -> None:
    """
    Load masking mappings directly from a Snowflake table.

    Expected columns:
    - original_value
    - masked_value
    """

    global _mask_map, _demask_map

    # Create Snowflake connection
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)

    query = f"""
        SELECT original_value, masked_value
        FROM {SNOWFLAKE_CONFIG["database"]}.{SNOWFLAKE_CONFIG["schema"]}.MASK_MAPPING
    """

    masking_df = pd.read_sql(query, conn)

    conn.close()

    # Build mask map
    for _, row in masking_df.iterrows():
        orig = row['ORIGINAL_VALUE']
        masked = row['MASKED_VALUE']

        if orig not in _mask_map:
            _mask_map[orig] = masked

    # Invert mapping
    _demask_map = {masked: orig for orig, masked in _mask_map.items()}

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

import pandas as pd

def get_descriptive_stats(df: pd.DataFrame) -> dict:
    stats = {}

    # --- Meta ---
    stats["meta"] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns))
    }

    # --- Numeric Stats ---
    numeric_df = df.select_dtypes(include=["number"])

    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()

        stats["numeric"] = {}

        for col in numeric_df.columns:
            stats["numeric"][col] = {
                "count": float(desc[col].get("count", 0)),
                "mean": float(desc[col].get("mean", 0)),
                "std": float(desc[col].get("std", 0)),
                "min": float(desc[col].get("min", 0)),
                "25%": float(desc[col].get("25%", 0)),
                "median": float(desc[col].get("50%", 0)),
                "75%": float(desc[col].get("75%", 0)),
                "max": float(desc[col].get("max", 0))
            }

    # --- Categorical Stats ---
    categorical_df = df.select_dtypes(include=["object", "category"])

    if not categorical_df.empty:
        stats["categorical"] = {}

        for col in categorical_df.columns:
            value_counts = df[col].value_counts(dropna=False)

            stats["categorical"][col] = {
                "unique": int(df[col].nunique(dropna=False)),
                "top": str(value_counts.index[0]) if not value_counts.empty else None,
                "top_count": int(value_counts.iloc[0]) if not value_counts.empty else 0
            }

    return stats

def deserialize_df(serialized_df):
    df = pd.DataFrame(
        serialized_df["data"],
        columns=serialized_df["columns"]
    )
    return df



def load_masking_file(masking_csv_path: str) -> None:
    """
    Call once at startup.
    _demask_map is derived by inverting _mask_map — single source of truth.
    """
    global _mask_map, _demask_map

    masking_df = pd.read_csv(masking_csv_path)

    for _, row in masking_df.iterrows():
        orig   = row['original_value']
        masked = row['masked_value']

        if orig not in _mask_map:
            _mask_map[orig] = masked

    # Invert _mask_map to get _demask_map
    _demask_map = {masked: orig for orig, masked in _mask_map.items()}


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


def get_all_summaries(messages: List):
    """
    Extracts only 'Result Summary' sections from all AI messages.

    Removes:
    - SQL Queries
    - Visualization code
    - Relevant Questions
    - SQL Results
    """

    cleaned_messages = []

    for msg in messages:

        if not isinstance(msg, AIMessage):
            continue

        content = msg.content or ""

        # -------- Extract Summary --------
        if "Result Summary:" in content:

            summary_part = content.split("Result Summary:")[-1]

            for stop in [
                "Relevant Questions:",
                "Query Results:",
                "Visualization Code:",
                "SQL Query Executed:"
            ]:
                if stop in summary_part:
                    summary_part = summary_part.split(stop)[0]
                    break

            summary_part = summary_part.strip()

            if summary_part:
                cleaned_messages.append(
                    AIMessage(
                        content=f"Result Summary:\n{summary_part}"
                    )
                )
    
    print("Message History")
    print(cleaned_messages)
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

def history_summarizer(meessages):
    all_summary=get_all_summaries(meessages)
    prompt=f"""
        You are a Context Compression Agent.

        Your task is to compress multiple conversation summaries into a highly dense,
        information-rich memory summary that can be used as context for future queries.

        INPUT:
        {all_summary}

        OBJECTIVE:
        - Preserve maximum business and analytical knowledge
        - Minimize token usage
        - Retain important entities, KPIs, trends, comparisons, filters, and conclusions
        - Remove repetition, filler, and verbose explanations
        - Merge overlapping insights intelligently

        IMPORTANT RULES:
        1. Preserve:
        - Metrics and KPI changes
        - Time periods
        - Regions / segments / brands
        - Comparative findings
        - Drivers and implications
        - Opportunities and risks
        - User intent patterns

        2. Remove:
        - Redundant wording
        - Explanatory filler
        - Generic transitions
        - Repeated conclusions

        3. Output Style:
        - Dense analytical summary
        - Short factual statements
        - No storytelling
        - No markdown
        - No bullets unless necessary

        4. Prioritize:
        - Recent findings
        - Repeated themes across conversations
        - Important directional changes

        5. If multiple summaries discuss the same topic:
        - Merge them into one compact insight

        OUTPUT:
        Return only the compressed memory summary.

"""
    result=model.invoke(prompt).content
    print("History Summary")
    print(result)
    return result

def get_relevant_history(state,recent_messages):

    question=state['question']

    prompt=f"""

    You are a context extraction agent. Your job is to analyze the last 3 conversation turns and extract ONLY the information relevant to answering the next user query.

    ────────────────────────────────────────────────
    STEP 0 — RELEVANCE GATE (run this first, always)
    ────────────────────────────────────────────────
    Before extracting anything, determine whether the next_query is meaningfully
    related to the conversation history.

    ═══════════════════════════════════════════════════════════
    ⚠️  ABSOLUTE RULE — NO EXCEPTIONS — READ BEFORE EVERY QUERY
    ═══════════════════════════════════════════════════════════

    FIRST, scan the query for referential language. If it contains NONE of:
    "above", "same", "those", "these", "that", "prior", "previous", "earlier",
    "mentioned", "listed", "shown", "identified", or ANY other backward-looking
    reference — STOP IMMEDIATELY. IT IS A NEW QUERY. Jump straight to the
    UNRELATED response below. Do not evaluate further.

    ONLY if referential language IS present, continue to check:
    - Does it reference entirely different entities, topics, or domains with no overlap?
    - Can it logically build on any prior turn?
    - Are there shared filters, time windows, entities, or metrics?

    → If referential language is ABSENT              — UNRELATED. Full stop.
    → If referential language is PRESENT but context — UNRELATED.
    has no overlap
    → If referential language is PRESENT and context — RELATED. Proceed to extraction.
    overlaps

    Carrying forward context without explicit referential language is a CRITICAL ERROR.

    ═══════════════════════════════════════════════════════════

    If UNRELATED, return this and STOP — do not populate any other fields:
    {{
    "query_intent": "<what the current query is asking — note it is a fresh standalone query with no relation to prior conversation>",
    "unrelated_to_history": true,
    "unrelated_reason": "<brief explanation, e.g. 'Query asks about X; prior conversation was about Y — no referential language detected'>",
    "anchored_entities": [],
    "period_context": {{}},
    "filters_applied": [],
    "metric_definitions": {{}},
    "suggested_join_keys": [],
    "exclude_from_next_query": [],
    "warnings": ["Current question is not related to the previous conversation. Context from prior turns was not applied."]
    }}

    Only proceed to full extraction if the query is RELATED or PARTIALLY RELATED.

    ────────────────────────────────────────────────
    CRITICAL RULE — ID ANCHORING
    ────────────────────────────────────────────────
    - NEVER generate, guess, or infer entity IDs (child_id, geography_id, etc.)
    - Only include IDs that appear verbatim in the SQL result data
    - If a question references "the above accounts", "those accounts", or "same accounts",
    extract the exact IDs from the most recent SQL result — do not add, remove, or rename any

    ────────────────────────────────────────────────
    INPUTS
    ────────────────────────────────────────────────
    - conversation_history: last N turns (HumanMessage + AIMessage pairs, including SQL queries and their result rows)
    LAST 3 CONVEERSAATION HISTORY:
    {recent_messages}
    - next_query: the upcoming user question
    {question}
    If conversation_history is empty or contains no SQL results, return:
    {{
    "query_intent": "<intent>",
    "unrelated_to_history": true,
    "warnings": ["No prior conversation history found. Unable to extract context."],
    ... // all other fields empty
    }}

    ────────────────────────────────────────────────
    OUTPUT FORMAT
    ────────────────────────────────────────────────
    Return a single JSON object. No prose before or after it.

    {{
    
    "query_intent": "<what the current query is asking and how it relates to or builds upon the prior conversation (e.g. follow-up, drill-down, new filter on same entities, unrelated)>",
    "unrelated_to_history": false,

    "anchored_entities": [
        {{
        "id": "<exact ID string from SQL results, e.g. child_id_532>",
        "name": "<exact account name from SQL results>",
        "relevant_metrics": {{
            // Only include key-value pairs the next query will actually need.
            // Keys are metric names (e.g. "total_sales_mg", "growth_pct").
            // Values are the exact values from the SQL result row.
            "<metric_key>": "<value>"
        }}
        }}
    ],

    "period_context": {{
        "recent_period_start": "<YYYY-MM-DD or null>",
        "recent_period_end": "<YYYY-MM-DD or null>",
        "prior_period_start": "<YYYY-MM-DD or null>",
        "prior_period_end": "<YYYY-MM-DD or null>"
    }},

    "filters_applied": [
        // Exact filter conditions active in prior queries that should carry forward.
        // e.g. "relmora_total_mg > 0", "region = 'West'"
    ],

    "metric_definitions": {{
        // Only include definitions for metrics the next query will reference.
        // Definitions should be derived from the SQL logic in prior turns, not assumed.
        "<metric_name>": "<definition>"
    }},

    "suggested_join_keys": [
        // Column names that can link this extracted context to the new query.
        // e.g. "child_id", "account_id", "period_start"
    ],


    "warnings": [
        // Anything the query builder should know before proceeding.
        // e.g. "Period window may be incomplete", "IDs are internal child_id values, not public"
    ]
    }}

    ────────────────────────────────────────────────
    EXTRACTION RULES
    ────────────────────────────────────────────────
    1. RELEVANCE GATE first — always run Step 0 before any extraction.
    2. "above accounts" / "those accounts" / "same accounts" → populate anchored_entities strictly from the most recent SQL result rows. No additions, no omissions.
    3. relevant_metrics → only include metrics the next query will actually need.
    4. period_context → always preserve the time window from prior turns; it anchors temporal comparisons.
    5. metric_definitions → only extract definitions that the next query references; derive them from prior SQL logic, never assume.
    6. Do not expand the entity list beyond what appeared in prior SQL results.
    7. If a field has nothing to populate, return an empty array [] or empty object {{}} — never omit the field.
"""
    response=model.invoke(prompt)
    # usage = response.usage_metadata
    # input_tokens = usage.get("input_tokens", 0)
    # output_tokens = usage.get("output_tokens", 0)
    # total_tokens = usage.get("total_tokens", 0)
    # print("\n===== Relevant Text Retreival TOKEN USAGE =====")
    # print(f"Input Tokens: {input_tokens}")
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Total Tokens: {total_tokens}")
    return response.content

def build_messages(state,SYSTEM_PROMPT):
    # print("All Messages")
    # print(state["messages"])
    #summary_history = history_summarizer(state["messages"])
    recent_messages = get_clean_recent_turns(state["messages"])
    get_relevant_context=get_relevant_history(state,recent_messages)
    print("Relevant Context")
    print(get_relevant_context)
    print("-"*100)
    recent_context_prompt=f"""
────────────────────────────────────────────
CONVERSATION CONTEXT
────────────────────────────────────────────

You are provided with TWO context sources derived from the last 3 conversation
turns. They serve different purposes — use them accordingly.

────────────────────────────────────────────
SOURCE 1 — EXTRACTED CONTEXT BLOCK (structured JSON)
────────────────────────────────────────────

Pre-validated summary of the last 3 turns. This is your PRIMARY source of
truth for entities, periods, and filters.

  ├── unrelated_to_history   If true, treat this query as fully standalone.
  │                          Ignore all prior context entirely.
  │
  ├── anchored_entities      PRIMARY source for all entity IDs.
  │                          IDs are verbatim from SQL results — never guess,
  │                          infer, or fabricate any ID not listed here.
  │
  ├── period_context         PRIMARY source for the active time window.
  │
  ├── filters_applied        Carry these forward unless explicitly removed.
  │
  ├── metric_definitions     Derived from prior SQL logic. Use these —
  │                          do not assume metric meanings independently.
  │
  └── warnings               Surface any that affect query correctness.

────────────────────────────────────────────
SOURCE 2 — RAW CONVERSATION TURNS (last 3 turns)
────────────────────────────────────────────

The unprocessed last 3 turns of the conversation. Use this as a SECONDARY
source only — to recover nuance, exact SQL logic, intermediate results, or
phrasing that the extracted block may not have captured.

  DO use for:
  ├── Recovering exact SQL from a prior turn if the new query builds on it
  ├── Reading intermediate result rows not captured in anchored_entities
  └── Understanding the user's analytical intent from their phrasing

  DO NOT use for:
  ├── Overriding entity IDs or periods already resolved in Source 1
  ├── Re-deriving filters or metrics that Source 1 already provides
  └── Replacing Source 1 when both sources are available

────────────────────────────────────────────
PRIORITY ORDER
────────────────────────────────────────────

  Source 1 (Extracted Block) > Source 2 (Raw Turns)

  If they conflict, trust Source 1. Raw turns may contain noise, partial
  results, or superseded values. The extracted block is the clean resolution.

────────────────────────────────────────────
RULES
────────────────────────────────────────────

1. TRUST SOURCE 1 OVER SOURCE 2
   Never override anchored_entities, period_context, or filters_applied
   using raw turn content if Source 1 already resolves them.

2. RESOLVE ALL REFERENCES FROM anchored_entities
   "Those accounts", "same childes", "them", "above" → IDs from
   anchored_entities only. If an ID is not there, check raw turns.
   If not in either source, it does not exist for this query.

3. NEVER FABRICATE IDs
   Do not generate, guess, or infer any entity ID not present verbatim
   in either source.

4. PRESERVE CONTINUITY
   Maintain entities, filters, grouping, and business logic unless the
   user explicitly changes them.

5. DO NOT RECOMPUTE WHAT IS ALREADY RESOLVED
   If Source 1 has the answer, use it. Fall back to Source 2 only for
   genuinely missing detail.

6. SURFACE WARNINGS FIRST
   If warnings is non-empty in Source 1, flag relevant ones before
   proceeding with your response.


"""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=recent_context_prompt),
        *get_relevant_context,
        *recent_messages
    ]


# def build_messages(state,SYSTEM_PROMPT):
#     # print("All Messages")
#     # print(state["messages"])
#     #summary_history = history_summarizer(state["messages"])
#     recent_messages = get_clean_recent_turns(state["messages"])
#     get_relevant_context=get_relevant_history(state,recent_messages)
#     print("Relevant Context")
#     print(get_relevant_context)
#     print("-"*100)
#     recent_context_prompt=f"""
# ────────────────────────
# CONVERSATION CONTEXT
# ────────────────────────

# You will receive two sources of prior context:


# * EXTRACTED CONTEXT BLOCK — structured JSON pre-processed from recent turns.
#   Contains resolved entities, periods, filters, metrics, and warnings.
#   Trust this over your own re-parsing of history for IDs, periods, and filters.

# ────────────────────────
# EXTRACTED CONTEXT BLOCK
# ────────────────────────

# Key fields and how to use them:

# * unrelated_to_history  — If true, treat the query as standalone. Skip all
#                           carry-forward context. Inform the user this question
#                           is unrelated to the prior conversation.
# * anchored_entities     — PRIMARY source for entity IDs. Never generate, guess,
#                           or infer IDs not present here.
# * period_context        — PRIMARY source for the active time window.
# * filters_applied       — Carry these forward unless the user explicitly removes them.
# * warnings              — Surface any that affect query correctness to the user.

# ────────────────────────
# RULES
# ────────────────────────

# 1. Anchor to Extracted Context
#    Use anchored_entities, period_context, and filters_applied as the authoritative
#    structured summary of recent context. Do not re-derive what they already provide.

# 3. Preserve Account Continuity
#    Resolve references like "those accounts", "same childes", "previous accounts" from anchored_entities Never fabricate IDs.

# 4. Reference Resolution
#    Pronouns and relative terms ("those", "same", "above", "them", "these") must
#    resolve to explicit entities from anchored_entities.

# 5. Maintain Analytical Continuity
#    Preserve entities, filters, grouping, and business logic unless the user
#    explicitly changes them. Apply only incremental modifications.

# 7. Trust Data Over Assumptions
#    Prefer SQL results, computed outputs, and explicit values. The extracted
#    context block is pre-validated — trust it over inference.

# 8. Surface Warnings
#    Flag any extracted context warnings that affect the current query's correctness
#    or interpretation before or alongside your response.

# """
#     return [
#         SystemMessage(content=SYSTEM_PROMPT),
#         SystemMessage(content=recent_context_prompt),
#         *get_relevant_context
#     ]

# def build_messages(state,SYSTEM_PROMPT):
#     # print("All Messages")
#     # print(state["messages"])
#     #summary_history = history_summarizer(state["messages"])
#     recent_messages = get_clean_recent_turns(state["messages"])
#     print("Recent Messages")
#     print(recent_messages)
#     print("-"*100)
#     recent_context_prompt=f"""
# ────────────────────────
# RECENT CONVERSATION CONTEXT (HIGHEST PRIORITY)
# ────────────────────────

# The following messages represent the MOST RECENT conversation history.

# They contain previously computed:

# * Entities (e.g., child_accounts, parent_accounts, regions, products, tiers, segments)
# * Filters (e.g., date ranges, time periods, conditions)
# * Metrics (e.g., sales, growth, aggregates)
# * SQL queries and their results
# * Previously identified account cohorts and derived account relationships

# You MUST treat this as ACTIVE EXECUTION CONTEXT.

# ────────────────────────
# GLOBAL CONVERSATION SUMMARY
# ────────────────────────

# In addition to the recent conversation history, you will also receive a compressed summary of the broader conversation history.

# This summary represents persistent analytical memory and may contain:

# * Previously analyzed trends
# * Important business findings
# * Historical entity relationships
# * Earlier filters and comparisons
# * Repeated user intent patterns
# * Previously identified child_accounts and parent_accounts

# Use this summary to maintain long-range conversational continuity while prioritizing the RECENT CONVERSATION CONTEXT when conflicts occur.

# ────────────────────────
# RULES
# ────────────────────────

# 1. Anchor to Recent Context

#    * Always use the RECENT CONVERSATION CONTEXT to interpret the current request.
#    * Prefer the MOST RECENT relevant entities, filters, metrics, and results.

# 2. Use Global Summary as Long-Term Memory

#    * Use the GLOBAL CONVERSATION SUMMARY to recover historical context and previously derived analytical knowledge.
#    * Do NOT override recent context using older summary information unless explicitly requested.

# 3. Preserve Account Continuity

#    * Maintain continuity of previously identified child_accounts and parent_accounts across follow-up queries.
#    * If the user references "those accounts", "same childes", "same parents", "previous accounts", or similar language, resolve them using the most recently derived account sets.
#    * Reuse previously computed account cohorts whenever possible instead of recomputing them.

# 4. Reference Resolution

#    * Terms like "those", "them", "same", "above", "previous", "that", or "these accounts"
#      MUST be resolved using the recent conversation context first, followed by the global summary if needed.

# 5. Do NOT Recompute

#    * If entities, filters, account cohorts, or results already exist, reuse them.
#    * Do NOT generate new values if they can be derived from prior context.

# 6. Maintain Analytical Continuity

#    * Preserve the same entities, filters, grouping, account definitions, and business logic unless explicitly changed by the user.
#    * Apply only incremental modifications requested in the current query.

# 7. Trust Data Over Assumptions

#    * Prefer SQL results, computed outputs, and explicit values over inferred logic or assumptions.

# """
#     return [
#         SystemMessage(content=SYSTEM_PROMPT),
#         SystemMessage(content=recent_context_prompt),
#         *recent_messages
#     ]


def query_decomposer_node(state: AgentState):
    review = state["active_review"]
    user_input=state["question"]
    print("User Input")
    print(user_input)
    messages=state['messages']
    query_decomposer_rag_examples_text=state['query_decomposer_rag_examples_text']
    

    last_human_message = next(
    msg for msg in reversed(messages) if isinstance(msg, HumanMessage)
)
    last_human_text=last_human_message.content
    print("Last Human Message")
    print(last_human_text)
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
    - If the user does not explicitly specify child or parent level, default all queries and aggregations to the child entity level. (VERY IMPORTANT)

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
        Any "top account" query defaults to: top N parent accounts ranked by total demand vials (commercial + PAP), Jan 1, 2025 through current date, sorted descending. Override only if the user specifies a different grain or date range.
            
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
    ## 0. Precedence

    When rules conflict, resolve top-down:

    1. Explicit user instruction
    2. Metric-specific definition (§6, §7)
    3. Grain ladder (§1)
    4. Defaults in this document

    ---

    ## 1. Grain

    Resolve output grain in this order. First match wins.

    | # | Condition | Grain |
    |---|---|---|
    | 1 | User names a grain explicitly | As stated |
    | 2 | Account-status metric — breadth, depth, new activation, dormancy, reactivation | `child_id` |
    | 3 | Reach or call frequency | `parent_id` |
    | 4 | Any account ranking query — "top N accounts" (Top 10 / Top 25 / Top 75 / "top accounts by ...") **or** "worst / bottom / lowest N accounts" (bottom 25, worst performers, lowest-volume accounts, underperforming accounts) — plus account tier / account segment | `parent_id` — always roll up to the parent account level by default |
    | 5 | Any other account-level question | `child_id` |
    | 6 | Everything else (incl. plain "demand" questions) | **National — no account grouping at all** |

    Rules:

    - Child entities roll up to their parent.
    - Aggregate on ID fields (`child_id`, `parent_id`); carry the corresponding name into the output.
    - A plain demand question returns national demand. Do **not** group by `parent_id` / `parent_name` unless asked.
    - Whenever the query says "nation," include national metrics in the output.

    ---

    ## 2. Population & Source Tables

    - **Demand default = Commercial + PAP** at every grain (national, segment, account). `data_867` UNION `data_867_pap` with `transaction_type = 'COM'` on the PAP branch. Never default to Commercial alone.
    - **Sales / revenue / net sales / gross sales → Revenue table only.** Single source of truth. No other table may serve these metrics.
    - Ex-factory vials accompany every actual-sales metric, sourced from the Revenue table.
    - Product is **Fyarro**.

    ---

    ## 3. Time & Anchoring

    - Anchor every window to the data, never to system date: `MAX(week_end_date)` for demand, `MAX(transaction_date)` for revenue, `MAX(call_date)` for calls.
    - Restrict to the most recent *completed* period unless the user asks otherwise.

    **Default windows:**

    | Question type | Window | Grain of the time axis |
    |---|---|---|
    | Demand trend | 52 weeks | Weekly |
    | Revenue / sales trend | 12 months | **Monthly** |
    | Longitudinal trend (pivot) | 26 weeks | Weekly |

    Longitudinal trend = pivot with accounts as rows and actual `week_end_date` values as column headers. Real dates only — never "Week 1". Each cell = that account's COM+PAP demand vials for that week. Build headers dynamically from the data. Account grain follows §1.

    ---

    ## 4. Business Days & Daily Averages

    - Business days are computed **strictly at national level**, never derived from regional, tier, or segmented data.
    - `daily_average = total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END)` — denominator is always the national business-day count.

    **Where daily averages are permitted:**

    | Output grain | Daily average? |
    |---|---|
    | National | Yes |
    | Segment / tier / region / area / geo / state / city / territory | Yes — divided by *national* business days |
    | Account (parent or child) | **Never** — totals only |
    | Any revenue / sales metric | **Never** |

    **Period completeness (national and segment rows only):**

    - Complete period → show period totals **and** daily averages.
    - Incomplete period → show daily averages **and** total volume demand.
    - Account rows → totals only, regardless of completeness.

    **Business-day display:**

    - Show the business-day count next to every time period in demand and budget output.
    - Do **not** compute or display business days for revenue / sales output.

    ---

    ## 5. Growth & Comparisons

    - Any "performance" question includes growth (% change vs. the prior comparable period).
    - Pair every growth or percentage with the absolute value **of the same metric family** — demand growth with demand vials, revenue growth with revenue dollars.
    - Prior = 0 and current > 0 → growth = **100%**.
    - Prior = 0 and current = 0 → growth = **0%**.
    - Prior is NULL → treat as 0.
    - When growth is reported at any segmentation level (segment, tier, region, area, geography, account type, city, state, territory), also compute **national growth** and add a column flagging the segment as **Higher** or **Lower** than nation.
    - The single anchor metric for every segment-vs-nation comparison is **daily average growth** (normalized by national business days). This applies to segment/tier aggregates. Account-grain rows are never compared this way — they carry total growth only.
    - Default denominator for any share or "% of total" metric is **overall national demand**, unless the user specifies otherwise.

    ---

    ## 6. Account Status Definitions

    All account-status metrics are at **child grain**.

    **Qualifying order:** `valid_order = 1 AND qty_sold > 0`. An "ordering account" is a child with at least one qualifying order in the window.

    **Dormant** — evaluated at week *W*:
    - Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    - Zero qualifying orders in week W.

    **Reactivated** — evaluated at week *W*, all three must hold:
    1. At least one qualifying order **before** the trailing 8-week window (prior order history exists), **and**
    2. Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    3. At least one qualifying order in week W.

    Total span evaluated: 9 weeks.

    **New account** — first-ever qualifying order falls in week W. A new account can **never** be classified as reactivated; condition (1) excludes it.

    ---

    **Rounding** — one rule: round (never truncate) every decimal result to **1 decimal place**.
    - Daily averages → `123.4`
    - Revenue / sales in $	Nearest whole integer — no decimals	$12346
    - Percentages → `12.3%`
    - All other decimal results	1 decimal	3.1

        Time Rules:
    
        If the user does not specify a time period, default to the most recent 8 weeks of available data.
        Never display month_start_date, month_end_date, quarter_start_date, or quarter_end_date in the output. Show month_year for month grain, quarter_year for quarter grain.
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
        Every time-based result must identify its period — week grain outputs week_start_date and week_end_date, month grain outputs month_year only, quarter grain outputs quarter_year only, any other grain outputs start and end date — and month_start_date, month_end_date, quarter_start_date, and quarter_end_date are never displayed. (VERY IMPORTANT)
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
            Any "top account" query defaults to: top N parent accounts ranked by total demand vials (commercial + PAP), Jan 1, 2025 through current date, sorted descending. Override only if the user specifies a different grain or date range.
            
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
    ## 0. Precedence

    When rules conflict, resolve top-down:

    1. Explicit user instruction
    2. Metric-specific definition (§6, §7)
    3. Grain ladder (§1)
    4. Defaults in this document

    ---

    ## 1. Grain

    Resolve output grain in this order. First match wins.

    | # | Condition | Grain |
    |---|---|---|
    | 1 | User names a grain explicitly | As stated |
    | 2 | Account-status metric — breadth, depth, new activation, dormancy, reactivation | `child_id` |
    | 3 | Reach or call frequency | `parent_id` |
    | 4 | Any account ranking query — "top N accounts" (Top 10 / Top 25 / Top 75 / "top accounts by ...") **or** "worst / bottom / lowest N accounts" (bottom 25, worst performers, lowest-volume accounts, underperforming accounts) — plus account tier / account segment | `parent_id` — always roll up to the parent account level by default |
    | 5 | Any other account-level question | `child_id` |
    | 6 | Everything else (incl. plain "demand" questions) | **National — no account grouping at all** |

    Rules:

    - Child entities roll up to their parent.
    - Aggregate on ID fields (`child_id`, `parent_id`); carry the corresponding name into the output.
    - A plain demand question returns national demand. Do **not** group by `parent_id` / `parent_name` unless asked.
    - Whenever the query says "nation," include national metrics in the output.

    ---

    ## 2. Population & Source Tables

    - **Demand default = Commercial + PAP** at every grain (national, segment, account). `data_867` UNION `data_867_pap` with `transaction_type = 'COM'` on the PAP branch. Never default to Commercial alone.
    - **Sales / revenue / net sales / gross sales → Revenue table only.** Single source of truth. No other table may serve these metrics.
    - Ex-factory vials accompany every actual-sales metric, sourced from the Revenue table.
    - Product is **Fyarro**.

    ---

    ## 3. Time & Anchoring

    - Anchor every window to the data, never to system date: `MAX(week_end_date)` for demand, `MAX(transaction_date)` for revenue, `MAX(call_date)` for calls.
    - Restrict to the most recent *completed* period unless the user asks otherwise.

    **Default windows:**

    | Question type | Window | Grain of the time axis |
    |---|---|---|
    | Demand trend | 52 weeks | Weekly |
    | Revenue / sales trend | 12 months | **Monthly** |
    | Longitudinal trend (pivot) | 26 weeks | Weekly |

    Longitudinal trend = pivot with accounts as rows and actual `week_end_date` values as column headers. Real dates only — never "Week 1". Each cell = that account's COM+PAP demand vials for that week. Build headers dynamically from the data. Account grain follows §1.

    ---

    ## 4. Business Days & Daily Averages

    - Business days are computed **strictly at national level**, never derived from regional, tier, or segmented data.
    - `daily_average = total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END)` — denominator is always the national business-day count.

    **Where daily averages are permitted:**

    | Output grain | Daily average? |
    |---|---|
    | National | Yes |
    | Segment / tier / region / area / geo / state / city / territory | Yes — divided by *national* business days |
    | Account (parent or child) | **Never** — totals only |
    | Any revenue / sales metric | **Never** |

    **Period completeness (national and segment rows only):**

    - Complete period → show period totals **and** daily averages.
    - Incomplete period → show daily averages **and** total volume demand.
    - Account rows → totals only, regardless of completeness.

    **Business-day display:**

    - Show the business-day count next to every time period in demand and budget output.
    - Do **not** compute or display business days for revenue / sales output.

    ---

    ## 5. Growth & Comparisons

    - Any "performance" question includes growth (% change vs. the prior comparable period).
    - Pair every growth or percentage with the absolute value **of the same metric family** — demand growth with demand vials, revenue growth with revenue dollars.
    - Prior = 0 and current > 0 → growth = **100%**.
    - Prior = 0 and current = 0 → growth = **0%**.
    - Prior is NULL → treat as 0.
    - When growth is reported at any segmentation level (segment, tier, region, area, geography, account type, city, state, territory), also compute **national growth** and add a column flagging the segment as **Higher** or **Lower** than nation.
    - The single anchor metric for every segment-vs-nation comparison is **daily average growth** (normalized by national business days). This applies to segment/tier aggregates. Account-grain rows are never compared this way — they carry total growth only.
    - Default denominator for any share or "% of total" metric is **overall national demand**, unless the user specifies otherwise.

    ---

    ## 6. Account Status Definitions

    All account-status metrics are at **child grain**.

    **Qualifying order:** `valid_order = 1 AND qty_sold > 0`. An "ordering account" is a child with at least one qualifying order in the window.

    **Dormant** — evaluated at week *W*:
    - Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    - Zero qualifying orders in week W.

    **Reactivated** — evaluated at week *W*, all three must hold:
    1. At least one qualifying order **before** the trailing 8-week window (prior order history exists), **and**
    2. Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    3. At least one qualifying order in week W.

    Total span evaluated: 9 weeks.

    **New account** — first-ever qualifying order falls in week W. A new account can **never** be classified as reactivated; condition (1) excludes it.

    ---


    **Rounding** — one rule: round (never truncate) every decimal result to **1 decimal place**.
    - Daily averages → `123.4`
    - Revenue / sales in $	Nearest whole integer — no decimals	$12346
    - Percentages → `12.3%`
    - All other decimal results	1 decimal	3.1

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
        Every time-based result must identify its period — week grain outputs week_start_date and week_end_date, month grain outputs month_year only, quarter grain outputs quarter_year only, any other grain outputs start and end date — and month_start_date, month_end_date, quarter_start_date, and quarter_end_date are never displayed. (VERY IMPORTANT)
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
    result=model.invoke(final_prompt)
    # usage = result.usage_metadata
    # input_tokens = usage.get("input_tokens", 0)
    # output_tokens = usage.get("output_tokens", 0)
    # total_tokens = usage.get("total_tokens", 0)
    # print("\n=====Query Decomposer TOKEN USAGE =====")
    # print(f"Input Tokens: {input_tokens}")
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Total Tokens: {total_tokens}")
    print("Query Decomposer Output")
    print("-"*100)
    print(result.content)
    # state['query_decomposer_output']=result
    # state["last_output"] = result
    # state["active_review"] = None
    log_trace(state, "query_decomposer", "TextMessage", result.content)   

    return {
        "query_decomposer_output":result.content,
        "last_output":result.content,
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
        Any "top account" query defaults to: top N parent accounts ranked by total demand vials (commercial + PAP), Jan 1, 2025 through current date, sorted descending. Override only if the user specifies a different grain or date range.
        
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
    ## 0. Precedence

    When rules conflict, resolve top-down:

    1. Explicit user instruction
    2. Metric-specific definition (§6, §7)
    3. Grain ladder (§1)
    4. Defaults in this document

    ---

    ## 1. Grain

    Resolve output grain in this order. First match wins.

    | # | Condition | Grain |
    |---|---|---|
    | 1 | User names a grain explicitly | As stated |
    | 2 | Account-status metric — breadth, depth, new activation, dormancy, reactivation | `child_id` |
    | 3 | Reach or call frequency | `parent_id` |
    | 4 | Any account ranking query — "top N accounts" (Top 10 / Top 25 / Top 75 / "top accounts by ...") **or** "worst / bottom / lowest N accounts" (bottom 25, worst performers, lowest-volume accounts, underperforming accounts) — plus account tier / account segment | `parent_id` — always roll up to the parent account level by default |
    | 5 | Any other account-level question | `child_id` |
    | 6 | Everything else (incl. plain "demand" questions) | **National — no account grouping at all** |

    Rules:

    - Child entities roll up to their parent.
    - Aggregate on ID fields (`child_id`, `parent_id`); carry the corresponding name into the output.
    - A plain demand question returns national demand. Do **not** group by `parent_id` / `parent_name` unless asked.
    - Whenever the query says "nation," include national metrics in the output.

    ---

    ## 2. Population & Source Tables

    - **Demand default = Commercial + PAP** at every grain (national, segment, account). `data_867` UNION `data_867_pap` with `transaction_type = 'COM'` on the PAP branch. Never default to Commercial alone.
    - **Sales / revenue / net sales / gross sales → Revenue table only.** Single source of truth. No other table may serve these metrics.
    - Ex-factory vials accompany every actual-sales metric, sourced from the Revenue table.
    - Product is **Fyarro**.

    ---

    ## 3. Time & Anchoring

    - Anchor every window to the data, never to system date: `MAX(week_end_date)` for demand, `MAX(transaction_date)` for revenue, `MAX(call_date)` for calls.
    - Restrict to the most recent *completed* period unless the user asks otherwise.

    **Default windows:**

    | Question type | Window | Grain of the time axis |
    |---|---|---|
    | Demand trend | 52 weeks | Weekly |
    | Revenue / sales trend | 12 months | **Monthly** |
    | Longitudinal trend (pivot) | 26 weeks | Weekly |

    Longitudinal trend = pivot with accounts as rows and actual `week_end_date` values as column headers. Real dates only — never "Week 1". Each cell = that account's COM+PAP demand vials for that week. Build headers dynamically from the data. Account grain follows §1.

    ---

    ## 4. Business Days & Daily Averages

    - Business days are computed **strictly at national level**, never derived from regional, tier, or segmented data.
    - `daily_average = total / COUNT(DISTINCT CASE WHEN is_business_day = 1 THEN date END)` — denominator is always the national business-day count.

    **Where daily averages are permitted:**

    | Output grain | Daily average? |
    |---|---|
    | National | Yes |
    | Segment / tier / region / area / geo / state / city / territory | Yes — divided by *national* business days |
    | Account (parent or child) | **Never** — totals only |
    | Any revenue / sales metric | **Never** |

    **Period completeness (national and segment rows only):**

    - Complete period → show period totals **and** daily averages.
    - Incomplete period → show daily averages **and** total volume demand.
    - Account rows → totals only, regardless of completeness.

    **Business-day display:**

    - Show the business-day count next to every time period in demand and budget output.
    - Do **not** compute or display business days for revenue / sales output.

    ---

    ## 5. Growth & Comparisons

    - Any "performance" question includes growth (% change vs. the prior comparable period).
    - Pair every growth or percentage with the absolute value **of the same metric family** — demand growth with demand vials, revenue growth with revenue dollars.
    - Prior = 0 and current > 0 → growth = **100%**.
    - Prior = 0 and current = 0 → growth = **0%**.
    - Prior is NULL → treat as 0.
    - When growth is reported at any segmentation level (segment, tier, region, area, geography, account type, city, state, territory), also compute **national growth** and add a column flagging the segment as **Higher** or **Lower** than nation.
    - The single anchor metric for every segment-vs-nation comparison is **daily average growth** (normalized by national business days). This applies to segment/tier aggregates. Account-grain rows are never compared this way — they carry total growth only.
    - Default denominator for any share or "% of total" metric is **overall national demand**, unless the user specifies otherwise.

    ---

    ## 6. Account Status Definitions

    All account-status metrics are at **child grain**.

    **Qualifying order:** `valid_order = 1 AND qty_sold > 0`. An "ordering account" is a child with at least one qualifying order in the window.

    **Dormant** — evaluated at week *W*:
    - Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    - Zero qualifying orders in week W.

    **Reactivated** — evaluated at week *W*, all three must hold:
    1. At least one qualifying order **before** the trailing 8-week window (prior order history exists), **and**
    2. Zero qualifying orders across the trailing 8 weeks (W-8 … W-1), **and**
    3. At least one qualifying order in week W.

    Total span evaluated: 9 weeks.

    **New account** — first-ever qualifying order falls in week W. A new account can **never** be classified as reactivated; condition (1) excludes it.

    ---


    **Rounding** — one rule: round (never truncate) every decimal result to **1 decimal place**.
    - Daily averages → `123.4`
    - Revenue / sales in $	Nearest whole integer — no decimals	$12346
    - Percentages → `12.3%`
    - All other decimal results	1 decimal	3.1

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
    Every time-based result must identify its period — week grain outputs week_start_date and week_end_date, month grain outputs month_year only, quarter grain outputs quarter_year only, any other grain outputs start and end date — and month_start_date, month_end_date, quarter_start_date, and quarter_end_date are never displayed. (VERY IMPORTANT)
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

def mask_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Mask values (same as before)
    df = df.apply(lambda col: col.map(lambda x: _mask_map.get(x, x)))

    # Step 2: Mask column names (case-insensitive)
    def mask_column(col_name: str) -> str:
        if not isinstance(col_name, str):
            return col_name

        for original, masked in _mask_map.items():
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            col_name = pattern.sub(masked, col_name)

        return col_name

    df.columns = [mask_column(col) for col in df.columns]

    return df

def sql_executor(state: AgentState):
    sql_generator_output=state["sql_generator_output"]
    result_df = run_snowflake_query(sql_generator_output)
    result_df = result_df.dropna(axis=1, how='all')
    result_df = result_df[~result_df.apply(lambda row: row.astype(str).str.strip().eq("UNKNOWN").any(), axis=1)]
    result_df = result_df[~result_df.apply(lambda row: row.astype(str).str.strip().eq("Unassigned").any(), axis=1)]
    result_df = result_df[~result_df.apply(lambda row: row.astype(str).str.strip().eq("-").any(), axis=1)]
    result_df = result_df[
    ~result_df.apply(
        lambda row: (
            row.isna().any() or
            row.astype(str).str.strip().str.upper().isin(
                ["NONE", "NULL", "NAN", "<NA>"]
            ).any()
        ),
        axis=1
    )
]
    print("Query Result:")
    print(result_df)
    # print("Masked DF")
    # load_masking_table_snowflake()
    # masked_df=mask_dataframe(result_df)
    # print(masked_df)
    

    serialized_df = {
        "columns": result_df.columns.tolist(),
        "data": result_df[:3000].to_dict(orient="records")
    }
    summary = f"Query executed successfully. Rows returned: {len(result_df)}"
    
    return {
    "sql_executor_output": serialized_df,
    "last_output":summary
    }

# def summarizer_node(state: AgentState):
#     query_decomposer_output=state["query_decomposer_output"]
#     data = json.loads(query_decomposer_output)
#     intent_summary = data["intent_summary"]
#     print("Intent Summary")
#     print(intent_summary)
#     sql_generator_output=state["sql_generator_output"]
#     sql_executor_output=state["sql_executor_output"]
#     prompt=f"""

#         You are a senior pharmaceutical Business Analyst presenting findings to leadership.

#         Intent: {query_decomposer_output}
#         Data: {sql_executor_output}

#         Write a 4–6 sentence executive summary. Apply every rule below:

#         TREND & GROWTH (always lead here if sales data is present)
#         - State direction clearly: growing / declining / flat, with exact % or absolute change.
#         - Name the best and worst periods with their exact dates (e.g., "week of Jan 31–Feb 6").
#         - If daily averages are available, use them to separate calendar effects from true demand shifts.

#         PERIOD COMPLETENESS (always flag this)
#         - Both total growth AND daily average present → periods are complete.
#         - Only daily average present → at least one period is incomplete (flag it, e.g., "the week ending March 6 shows only 1 business day and should not be read as a true demand drop").

#         REGIONAL / TIER / child BREAKDOWN (when applicable)
#         - Name the top-performing and bottom-performing region, tier, or child with their figures.
#         - Flag concentration risk if 1–2 entities drive a disproportionate share.

#         ACCOUNT HEALTH & ADOPTION (when applicable)
#         - State whether health/adoption is improving, stable, or deteriorating.
#         - Anchor to a specific date range and metric (e.g., "adoption within target childes rose from 42% to 58% between Q3 and Q4 2025").

#         MARKET SHARE (when applicable)
#         - State whether Rytelo is gaining or losing share vs. Reblozyl, with the exact share % and date.
#         - Call out which regions or tiers are driving the shift.

#         TOP-N RANKINGS (when applicable)
#         - List the top accounts/regions/tiers with their sales figure and share in one tight sentence.

#         ALWAYS
#         - Use exact dates, named periods (e.g., "13 weeks ending March 6, 2026"), and real numbers.
#         - If data is empty, state in one sentence: no activity found for this criteria.
#         - Do not mention SQL, agents, or system steps.
#         - Tone: direct, confident, board-room ready. No emojis.
#         - Don't Mention any Note.
#         - Only display market share information if in the intent summary you find something related to market share or competitor information.


#     """
#     # response=model.invoke(prompt).content
#     response=ask_llama(prompt)
#     print("LLama Response:")
#     print(response)
    
#     return {
#         "result_summary":response,
#         "last_output":response
#     }


# def demask_string(text: str) -> str:
#     """
#     Given a raw string, replace all masked values with their originals.
#     """
#     for masked, original in _demask_map.items():
#         text = text.replace(masked, original)
#     return text

def demask_string(text: str) -> str:
    """
    Given a raw string, replace all masked values with their originals.
    Case-insensitive match, but always restores the exact original value.
    """
    sorted_keys = sorted(_demask_map.keys(), key=len, reverse=True)

    for masked in sorted_keys:
        original = _demask_map[masked]
        text = re.sub(re.escape(masked), original, text, flags=re.IGNORECASE)

    return text

def demask_string_visualization(text: str) -> str:
    """
    Given a raw string, replace all masked values with the UPPERCASE
    version of their original values.
    Case-insensitive match.
    """
    sorted_keys = sorted(_demask_map.keys(), key=len, reverse=True)

    for masked in sorted_keys:
        original_upper = _demask_map[masked].upper()
        text = re.sub(
            re.escape(masked),
            original_upper,
            text,
            flags=re.IGNORECASE
        )

    return text

def summarizer_node(state: AgentState):
    query_decomposer_output=state["query_decomposer_output"]
    sql_generator_output=state["sql_generator_output"]
    sql_executor_output=state["sql_executor_output"]
    result_df=deserialize_df(sql_executor_output)
    #masked_df=mask_dataframe(result_df)
    prompt=f"""

You are a senior business analyst presenting analytical findings to an executive audience.

Your task is to synthesize query results into a sharp, insight-driven narrative — the kind a confident analyst would deliver in a leadership review meeting.

You are provided with:

Query Decomposer Output:
{query_decomposer_output}

SQL Executor Output:
{result_df}


---

INSTRUCTIONS:

Always format section labels exactly as: **Overview:**, **Key Findings / Takeaways:** followed by the content on the same or next line. These are the ONLY two sections permitted in the output. Do NOT produce separate "Findings", "Key Takeaways", or "Opportunity / Implication" sections under any circumstances — all of that content is consolidated into the single "Key Findings / Takeaways" section.

Limit each bullet point to a maximum of 2 sentences and 40 words. Lead with the single most important number or insight. Drop secondary comparisons, qualifications, and date ranges unless they are the core point. Never repeat a figure already stated in a prior bullet.

BUSINESS LANGUAGE ONLY: Never expose how the data was built. Terms like valid order, transaction type, qualifying order, flag, filter, table, column, or row are strictly forbidden. State what was measured, never how it was scoped.

NO REQUESTER FRAMING: Never narrate who asked for the analysis or why. Phrases such as "Leadership wanted", "The business asked", "Stakeholders requested", "As requested" are banned outright — delete them, do not rephrase. Open every statement on the subject and the number.

SENTENCE COHERENCE: Every sentence must read as natural business English a commercial leader would say aloud. Never stack scoping qualifiers into contradictory phrases — "Commercial demand with commercial-only PAP" is invalid; "Total demand" is correct. If a metric needs more than one qualifier to describe, the qualifiers are mechanics — drop them.

0. NEVER display any id fields in the output (for eg child_id, parent_id, region_id, geography_id, or any other identifier)

1. ABSOLUTE DISPLAY RULE — NAMES ONLY, NEVER IDs: Every reference to a child, territory, or region in the output — in every section, every bullet, and every sentence — MUST use the human-readable name field only: child_account_name, child_territory, and child_region. The corresponding ID fields (child_id, child_territory_id, child_region_id, parent_id, or any other _id field) are strictly forbidden from appearing anywhere in the output. This is non-negotiable and applies to narrative text, comparisons, rankings, and callouts without exception. If the name is not available in the result set, omit the entity entirely — never substitute or display an ID as a fallback.

2. Business question
Open by framing what business question this analysis answers and why it matters — without using the phrase "The analysis addresses" or "This answers a straightforward question."

3. Scope and context
Describe what was analyzed in plain business language: the time period, entities in scope, and any meaningful filters or boundaries. No SQL syntax, schema names, or technical references.

4. Findings content (delivered inside Key Findings / Takeaways)
Lead with the most significant result. Use specific figures, entity names, and comparisons.

5. Performance narrative (delivered inside Key Findings / Takeaways)
Go beyond listing numbers. Describe what the results reveal: which entities are leading or lagging, the magnitude of the gaps, and what the pattern suggests about performance. An executive should finish reading and immediately know where to focus attention.

6. Business significance (delivered inside Key Findings / Takeaways)
Close with what matters most — the so-what. What does this result mean for the business? Keep it grounded in the data; do not speculate beyond what the results support. Actionable implications must be grounded strictly in the data.

7. Empty results
If the result set is empty, clearly state that no activity or records were found, and describe the scope of what was searched (time range, entity type, filters) so the reader understands what the absence means.

8. CRITICAL RULE: Always display geography/region names instead of geography or region IDs in visualizations.

---

LANGUAGE & FACTUALITY
- Write in plain, simple English. Short sentences. No jargon, no hedging, no filler.
- State only what the data shows. Numbers, directions, and time periods — nothing else.
- Do not interpret, explain causes, speculate on drivers, or recommend actions.
- No superlatives or qualitative judgments ("strong", "concerning", "impressive") unless they are a computed fact from the dataframe.
- Every figure you cite must appear in the result set. If it isn't there, don't say it.

TONE AND STYLE:
- Executive register: direct, precise, and confident
- Written as spoken in a leadership review — authoritative but accessible
- No emojis
- No references to SQL, agents, systems, prompts, or any internal process
- No hedging unless genuine uncertainty exists in the data itself
- No self-reference ("I found..." or "This summary shows...")
- Numbers are evidence, not a list to recite — weave them into the narrative

---

════════════════════════════════════════════════════════════════════
⚠️  GLOBAL REDUNDANCY & FILLER RULE — THE SINGLE MOST IMPORTANT RULE
     (Governs every bullet in the Key Findings / Takeaways section)
════════════════════════════════════════════════════════════════════

Because all insights now live in ONE consolidated section, the risk of
redundancy is HIGHER than ever. This rule is therefore the highest-priority
rule in this entire prompt after data accuracy. When in doubt between
adding a bullet and merging it — ALWAYS MERGE.

FEWER BULLETS WITH MORE KNOWLEDGE IS ALWAYS PREFERRED OVER MORE BULLETS
WITH THE SAME OR LESS KNOWLEDGE. A single dense, unique, data-backed
bullet is strictly superior to five bullets that recycle the same subject.

── WHAT COUNTS AS REDUNDANT ───────────────────────────────────────

- TWO BULLETS ARE REDUNDANT if their CORE SUBJECT is the same —
  meaning they describe the same entity, the same metric, or the
  same directional pattern — even if worded differently, structured
  differently, or presented from a different narrative angle.
  Redundancy is determined by CORE SUBJECT AND MEANING, not wording.

- REFRAMING THE SAME NUMBERS FROM A DIFFERENT NARRATIVE ANGLE IS
  NOT A NEW INSIGHT AND DOES NOT JUSTIFY A SEPARATE BULLET.
  The following narrative angles ALL describe the same subject and
  MUST be merged into one bullet — they are NOT distinct insights:
    • Ranking angle        ("Tier 1 is highest, Tier 3 is lowest")
    • Absolute count angle ("Tier 3 has the most absolute active childes")
    • Trend angle          ("Adoption declines from Tier 1 to Tier 3")
    • Gap angle            ("23pp spread between highest and lowest")
    • Calendar angle       ("Business days fell but decline persists")
    • Decomposition angle  ("Total decline vs  decline differ")
    • Takeaway angle       ("The standout pattern is Tier 1's lead")
    • Implication angle    ("Focus attention on Tier 3's weak conversion")
      → An implication or takeaway that merely RESTATES a finding is
        redundant. An implication earns its own bullet ONLY if it adds
        a genuinely new, data-grounded angle (e.g., quantifies untapped
        headroom, names a specific concentration of opportunity, or
        surfaces a contrast not already stated).

- If two bullets reference the same underlying dataframe rows and
  columns — even partially — they are redundant. MERGE THEM.

- A "finding" restated as a "takeaway" is redundant. A "takeaway"
  restated as an "implication" is redundant. Since these formerly
  separate sections are now one, this failure mode is the most likely
  and the most strictly prohibited: NEVER write one bullet stating a
  result, a second bullet interpreting that same result, and a third
  bullet recommending action on that same result. That is ONE subject
  and must be ONE bullet — state the number, the meaning, and the
  action together in a single dense bullet.

── HOW TO COUNT BULLETS ────────────────────────────────────────────

- The Key Findings / Takeaways section MUST contain a MINIMUM of 1
  and a MAXIMUM of 5 bullets.
- Reaching 5 bullets is NOT a goal and is NOT compulsory. 5 bullets
  are justified ONLY when the data genuinely contains 5 distinct
  subjects, each passing the validity test below. If the data supports
  only one unique insight, exactly one bullet is the correct output.
- The number of bullets is determined purely by the number of DISTINCT
  SUBJECTS — not by the volume of data, the number of narrative angles
  about a single subject, or a desire to look thorough.
    • 1 distinct subject  → exactly 1 bullet, no matter how rich the data.
    • 2 distinct subjects → maximum 2 bullets.
    • 3 distinct subjects → maximum 3 bullets.
    • 4 distinct subjects → maximum 4 bullets.
    • 5+ distinct subjects → maximum 5 bullets; keep only the 5 most
      material subjects, merging or dropping the rest.
- NEVER split one subject across multiple bullets to reach a higher
  count. This is the most common and most prohibited failure.
- NEVER add a bullet just to reach a higher count. One strong, unique,
  data-backed bullet is strictly preferable to five redundant ones.

── CONCRETE FAILURE EXAMPLES (never repeat these patterns) ────────

    ❌ WRONG — tier adoption split across three narrative angles:
       • "Tier 1 delivered strongest adoption at 48% (128/268) vs
          Tier 2 at 35% (213/610) and Tier 3 at 25% (368/1,498),
          a 23pp gap."
       • "In absolute terms, Tier 3 generated the largest active
          childes at 368, but its larger base of 1,498 diluted
          conversion, leaving it 23pp below Tier 1."
       • "Adoption declines from 48% to 35% to 25% as target base
          expands from 268 to 610 to 1,498, showing scale is
          increasing faster than activation."

    ✅ CORRECT — entire subject merged into one dense bullet:
       • "Tier 1 leads adoption at 48% (128/268 childes), ahead of
          Tier 2 at 35% (213/610) and Tier 3 at 25% (368/1,498) —
          a 23pp spread — with Tier 3 holding the largest base at
          1,498 and highest absolute active count at 368, yet the
          weakest adoption rate, confirming scale is not translating
          into proportional activation."

    ❌ WRONG — national sales decline split across three angles:
       • "R4W growth is -30%, total sales from 370,125 MG to 259,205 MG;
           growth is -11%, from 19,480 MG to 17,280 MG."
       • "Business days fell from 19 to 15, but  still
          declined from 19,480 MG to 17,280 MG at -11%, confirming
          slowdown is not just a calendar effect."
       • "The gap between -30% total and -11%  decline
          shows fewer business days amplified the headline drop but
          underlying demand still deteriorated."

    ✅ CORRECT — all merged into one bullet:
       • "R4W vs P4W growth is -30%, with total sales declining from
          370,125 MG to 259,205 MG;  growth is -11%,
          with  sales declining from 19,480 MG to
          17,280 MG across 19 to 15 business days — confirming the
          slowdown is not a calendar effect and underlying demand
          deteriorated."

    ❌ WRONG — finding, takeaway, and implication written as three
       bullets about the same subject (the most likely failure now
       that sections are merged):
       • "West region declined -30%, from 2.4M MG to 2.1M MG."
       • "West is the weakest region, dragging national performance."
       • "The business should focus attention on the West region."

    ✅ CORRECT — one bullet carrying the number, meaning, and action:
       • "West region declined -30%, with total sales falling from
          2.4M MG to 2.1M MG — the weakest region and the primary drag
          on national performance, making it the single most important
          focus area."

── MANDATORY PRE-WRITE CHECK (run before every bullet) ────────────

Before writing each bullet, answer these questions in order:

1. "Does this bullet reference any dataframe rows or columns already
    used in any bullet written so far?"
    → YES : Merge the new information into the existing bullet.
    → NO  : Proceed to question 2.

2. "Does this bullet merely interpret, summarize, or recommend action
    on a result already stated in an earlier bullet?"
    → YES : Merge the interpretation/action into that earlier bullet.
    → NO  : Proceed to question 3.

3. "Can this information be appended to an existing bullet as a single
    clause without loss of clarity?"
    → YES : Merge it. Do not create a new bullet.
    → NO  : Only then write it as a new bullet.

── THREE-CONDITION VALIDITY TEST ──────────────────────────────────

A bullet is ONLY valid if ALL three conditions are true:
    1. It references dataframe rows/columns not used in any other
       bullet anywhere in the output.
    2. Its removal causes the analyst to lose data not recoverable
       from any remaining bullet.
    3. It cannot be merged into any existing bullet without loss
       of clarity.

    ✅ 1 bullet total — perfectly acceptable if only one unique
       insight exists in the data.
    ❌ Any two bullets anywhere in the output that reference the
       same dataframe data → INVALID, regardless of narrative
       angle or wording differences.

── FINAL DEDUPLICATION PASS (mandatory before emitting output) ─────

After drafting all bullets, re-read the full set one final time and:
    1. Delete or merge any bullet whose core subject overlaps another.
    2. Delete or merge any bullet that is a restated finding dressed
       as a takeaway or implication.
    3. Confirm the final count is between 1 and 5 and every surviving
       bullet passes the three-condition validity test.
Only then emit the output.

════════════════════════════════════════════════════════════════════
END OF GLOBAL REDUNDANCY RULE
════════════════════════════════════════════════════════════════════

---
── DATE FORMATTING RULE (MANDATORY) ───────────────────────────────────────────

- ALL dates appearing anywhere in the output MUST be displayed in the
  following format ONLY:
      DD Mon YYYY
      e.g. 13 Dec 2025, 07 Jan 2026, 01 Mar 2025

- The month MUST always be the first 3 letters of the English month
  name with the first letter capitalized:
      Jan, Feb, Mar, Apr, May, Jun,
      Jul, Aug, Sep, Oct, Nov, Dec

- NEVER display dates in any other format under any circumstances:
    ❌ 2025-12-13       (ISO format)
    ❌ 12/13/2025       (US numeric format)
    ❌ 13-12-2025       (European numeric format)
    ❌ December 13 2025 (full month name)
    ❌ 2025-Dec-13      (hyphenated mixed format)
    ✅ 13 Dec 2025      (ONLY accepted format)

- This rule applies to EVERY date in the output without exception:
    • Reporting period start and end dates
    • Comparison window dates
    • Any date referenced in the output
    • Date ranges (e.g. "13 Dec 2025 to 06 Mar 2026")

- Date ranges MUST follow this pattern:
    ✅ "R13W 13 Dec 2025 to 06 Mar 2026"
    ❌ "R13W 2025-12-13 to 2026-03-06"
    ❌ "R13W (2025-12-13 to 2026-03-06)"

OUTPUT FORMAT:
Present the summary in exactly TWO clearly labeled sections. Use the following structure:

Overview
A single factual sentence framing the business question and scope — no interpretation.
MANDATORY: if the question involves demand in any form, the Overview must state whether the
figures are Commercial demand only or Commercial + PAP demand. No exceptions — never leave
the demand basis implicit or unstated.

Key Findings / Takeaways
One consolidated section replacing Findings, Key Takeaways, and Opportunity /
Implication. MINIMUM 1, MAXIMUM 5 bullets. Each bullet covers one distinct
subject end-to-end: the result (exact numbers), the pattern or gap it shows,
and — only where the data supports it — the actionable implication. Never
split these across bullets.

- Strictly no interpretation of the results — report only what the data shows.
- Open the first bullet by naming the reporting period(s) used (e.g., P3M vs R3M, R13W, MTD, QTD), then lead with the most significant result — the metric with the largest absolute or relative change.
- Report National metrics first, then geography- and tier-level metrics where applicable.
- Call out the most meaningful gap or contrast alongside its own numbers — never as a separate restatement.
- The final bullet should make the single most important place to focus attention identifiable from the data alone.

── HARD RULES FOR NUMERIC REPORTING ──────────────────────────────────────────

- EVERY statement about sales, volume, growth, decline, increase, decrease,
  trend, or performance MUST include exact numeric values from the dataframe.
  Qualitative statements without numbers are STRICTLY FORBIDDEN.

- EVERY growth/decline mention MUST include ALL of the following in the SAME
  sentence. Missing even one value makes the entire insight INVALID:
    1. Prior-period Sales/Volume in MG or SLS    (e.g., 2.4M)
    2. Current-period Sales/Volume in MG or SLS  (e.g., 2.1M)
    3. Prior-period Growth %                     (e.g., 18%)
    4. Current-period Growth %                   (e.g., 16%)
    5. Prior-period  Sales          (e.g., 19,480)  ← MANDATORY if present
    6. Current-period  Sales        (e.g., 17,280)  ← MANDATORY if present
    7. Prior-period  Growth %       (e.g., 20%)     ← MANDATORY if present
    8. Current-period  Growth %     (e.g., 15%)     ← MANDATORY if present

-  GROWTH % when present in the dataframe is MANDATORY — it
  must NEVER be silently dropped and MUST appear alongside 
  sales values in the same statement.

- Growth insights WITHOUT both sales values AND growth percentages are
  INVALID and MUST NOT be generated.

- Every standalone sales/volume statement MUST state the exact total sales
  value for that period.
    ✅ "North East R3M sales were 45K."
    ❌ "North East showed strong performance."
    ❌ "North East R3M growth improved from 18% to 35%."  ← missing sales
    ❌ "Sales declined in the region."                    ← no numbers

── RESOLVING SALES VALUES FROM DATAFRAME ──────────────────────────────────────

- You will be provided with a dataframe containing column names and their
  corresponding data values. All sales, volume, growth, and 
  values MUST be extracted from this dataframe. Column names referenced
  in this prompt refer strictly to the column headers of the provided
  dataframe — not SQL aliases, not display labels, not inferred names.

- Sales values MUST always be resolved to actual numeric values from the
  dataframe rows. NEVER substitute column/field names for real numbers.
    ✅ "Total sales declined from 2.4M to 2.1M."
    ❌ "Total sales declined from the [column_name] to the [column_name]."

- TWO primary sales metrics are used — both are equally valid and MUST
  be recognized and reported wherever present in the dataframe:
    • MG  — Sales volume in MILLIGRAMS. Any column containing MG in its
            name represents physical product volume sold in milligrams.
    • SLS — Sales in DOLLARS. Any column containing SLS in its name
            represents revenue/dollar value of sales.
  Both MG and SLS columns are direct sales metrics and MUST be extracted
  and reported as sales figures for their respective entities.

- Scan ALL column names in the provided dataframe. Any column whose name
  contains ANY of the following keywords (case-insensitive, anywhere in
  the column name) is a sales/volume metric:
    • MG, SLS, SALES, VOL, VOLUME, UNITS, QTY, REVENUE, AVG

- Column naming patterns to recognize (non-exhaustive):
    • [prefix]_TOTAL_MG_[period]       e.g. relmora_total_mg_r4w   → MG sales
    • [prefix]_DAILY_AVG_MG_[period]   e.g. relmora_daily_avg_mg_p4w → daily avg MG
    • [prefix]_TOTAL_SLS_[period]       e.g. zynava_total_sls_r3m   → dollar sales
    • [prefix]_DAILY_AVG_SLS_[period]   e.g. zynava_daily_avg_sls_p3m → daily avg $
  Regardless of prefix or suffix, always extract the actual row value
  from the dataframe for that column.

- For EACH entity (product, competitor, geography, tier), scan the
  dataframe column names belonging to that entity and extract:
    • Prior-period sales value      (columns containing P4W, P3M, etc.)
    • Current-period sales value    (columns containing R4W, R3M, etc.)
    • Prior-period     (DAILY_AVG + prior period identifier)
    • Current-period   (DAILY_AVG + current period identifier)

- If a growth metric column (any dataframe column containing GROWTH in
  its name) IS present for an entity, "Sales value not available" MUST
  NOT be stated. Growth is a derived metric — its presence in the
  dataframe proves underlying sales data exists. Keep scanning all
  dataframe columns for the corresponding sales values.
    ✅ "R4W growth is -30%; absolute sales columns not present in this output."
    ❌ "Sales value not available." ← when a growth column exists for that
                                      same entity and period in the dataframe.

- Only state "Sales value not available" if NO recognizable sales/volume
  column AND NO growth column exists for that entity and period in the
  dataframe. Never fabricate, estimate, or substitute column names as values.

── HANDLING ZERO / MISSING PRIOR-PERIOD GROWTH VALUES ─────────────────────────

- If prior-period growth % is 0% or NULL, DO NOT report "growth moved from
  0% to X%". State only the current-period growth value directly.
    ✅ "National R4W growth is -30%, with total sales declining from 2.4M to 2.1M."
    ❌ "National R4W growth declined from 0% to -30%."

- This rule applies equally to  growth % and any other
  prior-period metric that is 0% or NULL.
    ✅ " growth is -11%, with  sales declining
        from 19,480 to 17,280."
    ❌ " growth declined from 0% to -11%."

── MANDATORY INSIGHT STRUCTURE ────────────────────────────────────────────────

Standard structure (both prior and current growth % available):
"[Entity] [period] growth [moved] from [prior growth%] to [current growth%],
 with total sales [moving] from [prior sales] to [current sales]; 
 growth [moved] from [prior DA growth%] to [current DA growth%], with daily
 average sales [moving] from [prior DA sales] to [current DA sales]."

Simplified structure (prior-period growth % is 0% or NULL):
"[Entity] [period] growth is [current growth%], with total sales [moving]
 from [prior sales] to [current sales];  growth is [current DA
 growth%], with  sales [moving] from [prior DA sales] to
 [current DA sales]."

The  portion is MANDATORY when DA data is present in dataframe.

Reference Examples:
- "National R3M growth declined from 18% to 16%, with total sales declining
   from 2.4M to 2.1M."
- "National R3M growth declined from -11% to -30%, with total sales declining
   from 370,125 to 259,205;  growth declined from 20% to 15%,
   with  sales declining from 20K to 15K."
- "North East growth improved from 18% to 35%, with total sales increasing
   from 35K to 45K;  growth improved from 10% to 22%, with
    sales increasing from 5K to 8K."
- "West region R4W growth is -30%, with total sales declining from 2.4M to
   2.1M;  growth is -11%, with  sales declining
   from 19,480 to 17,280."  ← prior-period growth was 0% or NULL

── COMPARATIVE BENCHMARKING ───────────────────────────────────────────────────

- Explicitly identify and report with exact values:
    • Strongest performing geography/tier  — total sales + growth %
    • Weakest performing geography/tier    — total sales + growth %
    • Geographies/tiers OUTPERFORMING national benchmark — state delta
    • Geographies/tiers UNDERPERFORMING national benchmark — state delta

- All geography- and tier-level insights MUST include:
    • Exact total sales value per period
    • Exact growth % per period
    • Exact  growth % per period (if present)
    • Exact  sales per period (if present)
    • Direct comparison against national-level equivalents where available

── PERIOD COMPLETENESS SIGNAL ─────────────────────────────────────────────────

- Presence of both total growth AND  growth → period is complete.
  When complete, BOTH of the following MUST appear together in the insight:
    • Total sales (prior → current) + Total growth % (prior → current)
    •  sales (prior → current) + DA growth % (prior → current)

── QUALITY CHECKLIST (self-verify before generating output) ───────────────────

[ ] Output contains ONLY two sections: Overview and Key Findings / Takeaways?
[ ] Key Findings / Takeaways contains between 1 and 5 bullets?
[ ] Every growth mention includes prior + current growth %?
      → Exception: prior-period is 0% or NULL → report current value only.
[ ] Every growth mention includes prior + current total sales (actual numbers,
      not column names) resolved directly from the dataframe?
[ ] Every growth mention includes prior + current DA growth % (if in dataframe)?
[ ] Every growth mention includes prior + current DA sales (if in dataframe)?
[ ] Every standalone sales statement includes exact total sales number?
[ ] National metrics reported before geography/tier metrics?
[ ] Strongest and weakest geographies/tiers identified with exact numbers?
[ ] "Sales value not available" NOT used when a growth metric exists?
[ ] No two bullets anywhere in the output share the same core subject
      or reference the same dataframe data?
[ ] No bullet merely interprets, summarizes, or recommends action on a
      result already stated in another bullet (finding restated as
      takeaway/implication)?
[ ] Every bullet passes the three-condition validity test?
[ ] Every bullet that carries an implication has that implication merged
      with its underlying numbers in the SAME bullet?
[ ] All values sourced from dataframe — no estimates, rounding, or
      column name substitutions?
[ ] Final deduplication pass completed before emitting output?

DEDUPLICATION RULE:
Every bullet must carry unique information. Remove any bullet that restates or
rephrases a point already made anywhere in the output — even in different
wording or from a different angle (finding vs takeaway vs implication).
Fewer sharp bullets is ALWAYS preferable to padded output. It is not
compulsory to reach 5 bullets — 5 only makes sense when there is genuinely
sufficient distinct information to convey; otherwise even a single bullet is
sufficient. Less points with more knowledge is preferred over more points
with the same or less knowledge.

Note: If the result set is empty, replace all sections with a single "No Results" section describing what was searched and what the absence means.
       
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

    """
    response=model.invoke(prompt).content
    print("Masked Summary")
    print(response)
    summary=demask_string(response)
    print("Summary")
    print(summary)


    
    return {
        "result_summary":summary,
        "last_output":summary
    }

def visualization_node(state: AgentState):
    query_decomposer_output=state["query_decomposer_output"]
    user_query=state["question"]
    sql_executor_output=state["sql_executor_output"]
    result_df=deserialize_df(sql_executor_output)
    #masked_df=mask_dataframe(result_df)
    descriptive_stats=get_descriptive_stats(result_df)
    print("Descriptive Stats:")
    print(descriptive_stats)
    columns = sql_executor_output["columns"]
    num_rows = len(sql_executor_output["data"])
    summary=state["result_summary"]

    sql_generator_output=state["sql_generator_output"]


    #result_summary=state["result_summary"]
    prompt=f"""
You are a Visualization Agent.
 
Your goal is to create a meaningful, accurate, and non-misleading Plotly visualization ONLY when the data supports it.
 
You MUST prioritize correctness over forcing a chart.
 
---
 
## INPUTS
 
User Query:
{user_query}
 
Query Decomposer Output:
{query_decomposer_output}
 
SQL Generator Output:
{sql_generator_output}
 
SQL Executor Output:
{result_df}
 
SQL Executor Output Descriptive Stats:
{descriptive_stats}
 
SUMMARY-DRIVEN VISUALIZATION RULE — METRIC PRIORITIZATION
{summary}
 
Read the summary and identify the ONE metric it leads with. That metric is the subject of the chart.

A second metric may join it only when the summary frames it as a rate of change on the first (growth %, share %). In that case the base metric is bars or a line, and the rate goes on a secondary axis.

Everything else in the summary stays in the summary. Do NOT attempt to fit every figure, ranking, or takeaway into the chart. A chart carrying one idea clearly beats a chart carrying five badly, and the summary text is already shown directly above the chart.

- Use the summary's exact metric wording for axis titles and legend names ("Net Revenue" stays "Net Revenue", never "Sales").
- Rankings, standings, and implications ("weakest region", "primary focus area") are narrative, not chart elements. Never render them as annotations, callouts, arrows, or reference lines.

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
   
4. Time Axis Rule: Use the dataset's exact time granularity (week/month/quarter) for the X-axis—no transformations or mixing.
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
  * PREFERRED over bar chart when showing tier-wise or segment-wise distribution as a share of total (e.g., "tier distribution", "segment breakdown", "% share by tier")
 
* Flat/tabular outputs with no clear analytical mapping:
  Return NO_VISUALIZATION
 
* If multiple chart types are possible:
  Choose the simplest and most interpretable one
 
---
 
## CHART OVERRIDES (HIGH PRIORITY)

These override the basic rules when applicable. Every one of them caps complexity — respect the caps.

1. Trend across categories -> MULTI-LINE, maximum 4 lines. If more series exist, keep the top 4 by latest value and drop the rest.
2. Part-to-whole, no time axis, <= 6 segments -> PIE CHART.
3. Contribution or share across time -> STACKED AREA (time) or 100% STACKED BAR (snapshot).
4. Market share -> ALWAYS percentage, never absolute units. If total market volume is unavailable, return NO_VISUALIZATION.
5. Actual vs target/budget -> GROUPED BAR, exactly 2 series.
6. Two categorical dimensions -> GROUPED BAR on the primary dimension, top 8 categories only. Do NOT use heatmaps.
7. Cross-entity distribution comparison -> GROUPED BAR in % so entities of different total size stay comparable.
8. Consistency or variability -> BOX PLOT only when each group has 20+ points, otherwise fall back to bar or line.

If two chart types both work, always choose the simpler one.

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
 
## METRIC FIDELITY RULE (CRITICAL — NEW)
 
The chart MUST display exactly the metric the user asked for — never substitute a different metric even if it is available in the data:
 
* If the user asked for "average calls per day" → plot avg_calls_per_day, NOT total_calls
* If the user asked for "daily average sales" → plot daily_avg_sales, NOT total_sales
* If the user asked for "% market share" → plot percentage share, NOT absolute mg
* If both total and average are available and the user asked for average → use average as the primary Y-axis; total may appear as a secondary trace only if it adds context
* Before finalizing, re-read the user query and confirm every Y-axis value matches the requested metric exactly
 
---
 
## COLUMN USAGE RULES
 
* Use ONLY columns present in df
 
* NEVER invent or infer missing columns
 
* Preferred mappings:
 
  * x → categorical or datetime column
  * y → numeric column(s)
 
---
 
## SIMPLICITY RULES (MANDATORY)

The chart must be readable in three seconds. Enforce every rule below.

1. ONE MESSAGE PER CHART. Maximum 2 metrics and maximum 4 series. If the data supports more, plot the most important ones — the full detail is already shown in the results table beneath the chart.
2. TOP N ONLY. Never plot more than 10 categories. Sort descending by value, keep the top 10, and add "(Top 10)" to the title when you trim.
3. NAMES, NEVER IDS. Never place child_id, parent_id, campus_id, or any raw ID on an axis, label, legend, hover, or title. Always resolve to child_name, parent_name, region, account_type. If no name exists, show "Unknown" — never fall back to the ID.
4. NO CHART FURNITURE. No annotations, callouts, arrows, reference lines, target lines, or free-floating text of any kind. The only permitted text is: chart title, axis titles, tick labels, legend entries, and data labels.
5. NO STYLING CODE. Do NOT set colors, fonts, height, width, margins, legend position, tickangle, textposition, textfont, hovermode, or templates. The rendering layer applies all presentation deterministically after your code runs, and anything you set there will be overwritten or will conflict.
6. TIME MEANS LINE. Trend, growth, momentum, or change over time -> line chart. Never a stacked bar across time unless the explicit ask is composition at each time point.
7. TIME AXIS FOLLOWS THE RESULT GRAIN, NOT THE COLUMN TYPE. A monthly result still contains week_end_date because the SQL is required to emit period boundaries. Pick the time column by counting distinct values: if month_year and week_end_date have the SAME number of distinct values there is one row per month, so plot month_year. If week_end_date has MORE distinct values the grain is weekly, so plot week_end_date. Same test for quarter_year and year. Never plot week_end_date on a monthly, quarterly or yearly result.
7a. NEVER MIX GRANULARITIES and never emit HH:MM:SS on a time axis.
8. DEDUPLICATE. Call plot_df = plot_df.drop_duplicates() before plotting a time series so lines never double back on themselves.
9. START FROM df. The first executable line is always plot_df = df.copy(). Never reference plot_df or any derived frame before that line.
10. GROWTH ON A SECOND AXIS. When a growth/%/change column is plotted next to a volume column, use make_subplots(specs=[[{{"secondary_y": True}}]]) and place growth on secondary_y=True. Never hand-assign yaxis='y2'.
11. PERIOD COMPARISONS. A two-period comparison must show total volume, daily average, and growth % — never a bar chart of day counts.
12. SINGLE-ROW GROWTH. If the question is about growth and the result has one row, still render a bar chart: previous vs current period, preferring total and daily average side by side. Never skip the chart.
13. HOVERTEMPLATE SYNTAX. f-strings only. Escape Plotly placeholders as %{{x}} and %{{y}}. Never mix Python % formatting with Plotly placeholders. Percentages always carry the % symbol.

## PLOTLY OUTPUT RULES

* plotly.express or plotly.graph_objects only
* These names are ALREADY available in the execution scope — px, go, make_subplots, pd, np. Import lines are optional and never required.
* Output ONLY Python code defining `fig` — no markdown fences, no comments, no explanation
* No Streamlit code, no fig.show(), no use_container_width, no width= in update_layout

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
    # visualization_code=demask_string_visualization(response)
    # print("Visualization Code Masked")
    # print("-"*100)
    # print(response)
    # print("Visualization Code Demasked")
    # print("-"*100)
    # print(visualization_code)
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
    builder.add_edge("summarizer_node","visualization_node")
    #builder.add_edge("summarizer_node","terminator")
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



