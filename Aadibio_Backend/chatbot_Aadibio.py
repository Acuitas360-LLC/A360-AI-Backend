from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from subgraph_Aadibio import build_graph
from datetime import datetime, UTC
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List
import numpy as np
import base64
import json
import os
from openai import OpenAI
import plotly.express as px
import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from typing import TypedDict, Literal, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# Access the key
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
model = ChatOpenAI(model="gpt-5.4")
openai_api_key = os.getenv("OPENAI_API_KEY")
from rapidfuzz import process, fuzz
from collections import defaultdict

def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _load_private_key() -> bytes:
    private_key_b64 = os.getenv("SNOWFLAKE_PRIVATE_KEY_B64")
    private_key_pem = os.getenv("SNOWFLAKE_PRIVATE_KEY_PEM")
    private_key_path = _env_or_default(
        "SNOWFLAKE_PRIVATE_KEY_PATH",
        os.path.join(os.path.dirname(__file__), "snowflake_keys", "rsa_key.p8"),
    )
    private_key_passphrase = _env_or_default(
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "Murtaza@1971",
    )

    if private_key_b64:
        key_data = base64.b64decode(private_key_b64)
    elif private_key_pem:
        key_data = private_key_pem.encode()
    else:
        with open(private_key_path, "rb") as key_file:
            key_data = key_file.read()

    private_key = serialization.load_pem_private_key(
        key_data,
        password=private_key_passphrase.encode(),
        backend=default_backend(),
    )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


SNOWFLAKE_CONFIG = {
    "user": _env_or_default("SNOWFLAKE_USER", "ahusain"),
    "account": _env_or_default("SNOWFLAKE_ACCOUNT", "ua60309.south-central-us.azure"),
    "private_key": _load_private_key(),
    "warehouse": _env_or_default("SNOWFLAKE_WAREHOUSE", "AADIBIO_COMPUTE"),
    "database": _env_or_default("SNOWFLAKE_DATABASE", "AADIBIO_CAI"),
    "schema": _env_or_default("SNOWFLAKE_SCHEMA", "AADIBIO_CAI_SCHEMA"),
    "role": _env_or_default("SNOWFLAKE_ROLE", "CONVERSATIONAL_AI"),
    "client_session_keep_alive": True,
}

# ---------- Global Dictionary ----------
MASKING_TABLE_DICT = {}

def load_masking_table(table_name: str = "MASK_MAPPING") -> dict:
    """
    Loads the masking table from Snowflake and converts it into:
    {
        "territory_name": ["North Territory", "South Territory", ...],
        "state_name":     ["Telangana", "Maharashtra", ...],
        "city_name":      ["Hyderabad", "Mumbai", ...],
        ...
    }
    """
    global MASKING_TABLE_DICT

    try:
        with snowflake.connector.connect(**SNOWFLAKE_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT column_name, original_value FROM {table_name}")
                rows = cursor.fetchall()

        # Build dictionary — group original_values under each column_name
        result = defaultdict(list)
        for column_name, original_value in rows:
            result[column_name].append(original_value)

        # Convert to regular dict and store globally
        MASKING_TABLE_DICT = dict(result)

        print(f"✅ Masking table loaded: {len(MASKING_TABLE_DICT)} columns, "
              f"{sum(len(v) for v in MASKING_TABLE_DICT.values())} total values")

    except Exception as e:
        print(f"❌ Failed to load masking table: {e}")
        raise

    return MASKING_TABLE_DICT

# Define your fallback column priority order here
COLUMN_FALLBACK_ORDER = [
    "region",
    "speciality_distributor",
    "parent_name"
    # add more columns in priority order as needed
]

def fallback_column_search(entity_value, masking_table, threshold=70):
    """
    When column_name is unknown, try each column in priority order.
    Returns first confident match found.
    """
    for column in COLUMN_FALLBACK_ORDER:
        if column not in masking_table:
            continue

        corrected_value, score, status = fuzzy_correct(column, entity_value, masking_table, threshold)

        if status in ("exact", "case_corrected", "fuzzy_corrected"):
            print(f"🔍 Fallback matched '{entity_value}' → '{corrected_value}' in column '{column}' (score: {score})")
            return column, corrected_value, score, status

    print(f"⚠️ Fallback exhausted all columns for '{entity_value}', no match found")
    return None, entity_value, None, "no_match"

def extract_entities_from_query(user_query, valid_columns):
    prompt = f"""
    Available columns: {valid_columns}

    From the user query below, extract ALL entities that correspond to 
    any of the available columns above.

    Return ONLY a JSON array like:
    [
        {{"column_name": "state_name", "entity_value": "Telangana"}},
        {{"column_name": null,         "entity_value": "Sttle"}}
    ]

    Rules:
    - Only if in the Query it is expliciittly mentioned mentioned about the column_name then only take that as a column name or else mark it as null (VERY IMPORTANT). For Exxample west region, then onlyy consider westt to be region if nothing is mention cconsider it to be null.
    - column_name must always be one of the available columns listed above, or null if unsure
    - Extract as many entities as present in the query
    - Don't add any prefix or suffix to the entity name
    - If no entity is found for a column, skip it
    - Return empty array [] if nothing relevant is found

    User query: "{user_query}"
    """
    response = model.invoke(prompt).content

    try:
        clean    = response.strip().replace("```json", "").replace("```", "")
        entities = json.loads(clean)
        return entities if isinstance(entities, list) else []
    except json.JSONDecodeError:
        print("⚠️ Failed to parse LLM response as JSON")
        return []

# ---------- Step 2: Fuzzy match a single entity ----------
def fuzzy_correct(column_name, entity_value, masking_table, threshold=70):
    valid_values = masking_table.get(column_name, [])
    print("Valid Values")
    print(valid_values)

    if not valid_values:
        return entity_value, None, "unknown_column"

    # Exact match — no correction needed
    if entity_value in valid_values:
        return entity_value, 100, "exact"

    # Case-insensitive exact match
    lower_map = {v.lower(): v for v in valid_values}
    if entity_value.lower() in lower_map:
        return lower_map[entity_value.lower()], 100, "case_corrected"

    # Fuzzy match
    result = process.extractOne(
        entity_value,
        valid_values,
        scorer=fuzz.token_set_ratio
    )

    print("Result")
    print(result)

    if result:
        match, score, _ = result
        if score >= threshold:
            return match, score, "fuzzy_corrected"

    return entity_value, None, "no_match"


# ---------- Step 3: Correct ALL entities ----------
def correct_all_entities(entities, masking_table):
    corrections = []

    for item in entities:
        col             = item["column_name"]
        value           = item["entity_value"]
        was_column_null = col is None  # ✅ capture before any resolution

        if col is None:
            print(f"🔎 Column unknown for '{value}', trying fallback column search...")
            col, corrected_value, score, status = fallback_column_search(value, masking_table)
        else:
            corrected_value, score, status = fuzzy_correct(col, value, masking_table)

        corrections.append({
            "column_name":     col,
            "original_value":  value,
            "corrected_value": corrected_value,
            "score":           score,
            "status":          status,
            "was_column_null": was_column_null  # ✅ store the flag
        })

        if status == "exact":
            print(f"✅ '{value}' → exact match in '{col}'")
        elif status in ("case_corrected", "fuzzy_corrected"):
            print(f"🔧 '{value}' → '{corrected_value}' in '{col}' (score: {score})")
        elif status == "no_match":
            print(f"⚠️ '{value}' → no match found anywhere, flagging for LLM fallback")
        elif status == "unknown_column":
            print(f"❌ '{col}' not found in masking table")

    return corrections


# ---------- Step 4: Handle no-match cases via LLM fallback ----------
def llm_fallback_correction(no_match_items, masking_table):
    fallback_results = []

    for item in no_match_items:
        col   = item["column_name"]
        value = item["original_value"]

        if col is not None:
            columns_to_check = [col]
        else:
            columns_to_check = COLUMN_FALLBACK_ORDER

        corrected    = None
        resolved_col = None

        for current_col in columns_to_check:
            valid_values = masking_table.get(current_col, [])
            if not valid_values:
                continue

            print(f"🔎 Checking all {len(valid_values)} values in '{current_col}'")


            prompt = f"""
                The user mentioned "{value}" in their query.

                These are ALL the valid values for column "{current_col}":
                {valid_values}

                Does any of these CLOSELY match what the user meant? 
                Only return a match if you are highly confident it is a typo or abbreviation of a valid value.
                For example "Hydrabad" → "Hyderabad" is a valid correction.
                But "Sttle" → "South East" is NOT a valid correction as they are not similar at all.

                If nothing closely matches, return null — do not force a match.

                Return ONLY JSON:
                {{
                    "corrected_value": "exact value from the list above or null if no good match",
                    "reason": "brief reason"
                }}
            """
            response = model.invoke(prompt).content
            print("Response")
            print(response)

            try:
                clean     = response.strip().replace("```json", "").replace("```", "")
                result    = json.loads(clean)
                corrected = result.get("corrected_value")

                if corrected:
                    resolved_col = current_col
                    print(f"✅ LLM confirmed: '{value}' → '{corrected}' in '{resolved_col}' | {result.get('reason')}")
                    break  # ← Stop as soon as LLM confirms a match
                else:
                    print(f"⏭️ LLM rejected all values in '{current_col}', moving to next...")

            except:
                print(f"⚠️ Failed to parse LLM response for column '{current_col}'")
                continue

        # After going through all columns
        if corrected:
            fallback_results.append({
                "column_name":     resolved_col,
                "original_value":  value,
                "corrected_value": corrected,
                "status":          "llm_fallback"
            })
        else:
            print(f"❌ '{value}' unresolved after checking all fallback columns")
            fallback_results.append({
                "column_name":     col,
                "original_value":  value,
                "corrected_value": value,
                "status":          "unresolved"
            })

    return fallback_results

import re

# def apply_corrections_to_query(user_query, corrections):
#     corrected_query = user_query

#     for item in corrections:
#         original        = item["original_value"]
#         corrected       = item["corrected_value"]
#         status          = item["status"]
#         column_name     = item.get("column_name")
#         was_column_null = item.get("was_column_null", False)

#         if status in ("case_corrected", "fuzzy_corrected", "llm_fallback") and original != corrected:
#             # ✅ corrected value always wrapped in quotes
#             replacement = f'"{corrected}" in "{column_name}"' if was_column_null and column_name else f'"{corrected}"'

#             pattern         = r'\b' + re.escape(original) + r'\b'
#             corrected_query = re.sub(pattern, replacement, corrected_query, flags=re.IGNORECASE)
#             print(f"🔁 Replaced '{original}' → '{replacement}'")

#     return corrected_query

def apply_corrections_to_query(user_query, corrections):
    corrected_query = user_query

    for item in corrections:
        original    = item["original_value"]
        corrected   = item["corrected_value"]
        status      = item["status"]
        column_name = item.get("column_name")

        if status in ("exact", "case_corrected", "fuzzy_corrected", "llm_fallback"):
            # ✅ Always append column_name regardless of whether it was null or not
            replacement     = f'"{corrected}" in "{column_name}"' if column_name else f'"{corrected}"'
            pattern         = r'\b' + re.escape(original) + r'\b'
            corrected_query = re.sub(pattern, replacement, corrected_query, flags=re.IGNORECASE)
            print(f"🔁 Replaced '{original}' → '{replacement}'")

    return corrected_query

# ---------- Master Orchestrator ----------
def process_user_query(user_query):
    masking_table = load_masking_table()
    valid_columns = list(masking_table.keys())

    print(f"\n🔍 Query: {user_query}")
    print("-" * 50)

    # Step 1 — Extract all entities
    entities = extract_entities_from_query(user_query, valid_columns)
    # print("Entities")
    # print(entities)
    print(f"📦 Extracted {len(entities)} entities: {entities}\n")

    if not entities:
        print("No entities found, proceeding with raw query")
        return user_query

    # Step 2 & 3 — Fuzzy correct all entities
    corrections = correct_all_entities(entities, masking_table)

    # Step 4 — LLM fallback for unresolved ones
    no_matches = [c for c in corrections if c["status"] == "no_match"]
    if no_matches:
        print(f"\n🔄 Sending {len(no_matches)} unresolved entities to LLM fallback...")
        fallback = llm_fallback_correction(no_matches, masking_table)

        # ✅ Fix: merge by original_value instead of column_name
        # column_name can be None so it's not a reliable key
        fallback_map = {f["original_value"]: f for f in fallback}
        for c in corrections:
            if c["status"] == "no_match" and c["original_value"] in fallback_map:
                c.update(fallback_map[c["original_value"]])

    # Step 5
    return apply_corrections_to_query(user_query, corrections)


def run_snowflake_query(query: str) -> pd.DataFrame:
    with snowflake.connector.connect(**SNOWFLAKE_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
            return pd.DataFrame(rows, columns=columns)



def sql_generator_build_rag_examples_block(results):
    if not results:
        return "No RAG examples found."

    blocks = []
    blocks.append(f"""
    ────────────────────────────────────────────
    RAG EXAMPLES (HIGH PRIORITY: USAGE + JSON → SQL MAPPING)
    ────────────────────────────────────────────
    If RAG examples are provided and relevant:

    Reuse the closest example’s SQL structure/style

    Reuse patterns for:

    monthwise/quarterwise formatting (long vs pivot)

    conditional aggregations (CASE WHEN)

    max-date / rolling-window subquery patterns

    Only deviate if the Decomposer JSON forces it

""")
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"""
                ### Example #{i}
                Score: {r['score']:.4f}

                User Question:
                {r['matched_question']}

                Query Decomposition:
                {json.dumps(r["query_decomposition"], indent=2, ensure_ascii=False)}

                Final SQL:

                {r["final_sql"]}
                """

            )
    return "\n".join(blocks)
        
def query_decomposer_build_rag_examples_block(results):
    if not results:
        return "No RAG examples found."

    blocks = []
    blocks.append(f"""
        ────────────────────────
        RAG EXAMPLES (HIGH PRIORITY — MUST FOLLOW IF RELEVANT)
        ────────────────────────
        RAG examples are NOT optional reference. They are HIGH PRIORITY patterns.
                                            
        If RAG examples are provided and relevant to the user query: - You MUST follow the closest example’s decomposition style and logic 
        - You MUST reuse the same grouping/aggregation strategy where applicable 
        - You MUST prefer RAG-derived patterns over generic reasoning RAG Alignment Requirements: 
        - If RAG examples are provided, set rag_alignment.rag_provided = true 
        - used_examples MUST list the example identifiers you followed (e.g., "Example #1") 
        - borrowed_patterns MUST describe what you reused (e.g., "monthwise grouping using month_year", "group_by parent_eid,parent_name") - differences_from_examples MUST be empty unless schema/intent forces deviation - If you deviate, differences_from_examples MUST clearly state why (schema mismatch, different intent, missing columns, etc.)


        RAG EXAMPLES:
""")
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"""
                ### Example #{i}
                Score: {r['score']:.4f}

                User Question:
                {r['matched_question']}

                Query Decomposition:
                {json.dumps(r["query_decomposition"], indent=2, ensure_ascii=False)}

                """
                    
            )
    return "\n".join(blocks)





# ---------- Step 1: LLM extracts ALL entities from query ----------




def get_intent_summary(user_query):
    prompt=f"""
    You are an Intent Extraction agent for an analytics question-answering system.

        Your task is to read a user’s natural-language question and produce a single
        clear, canonical intent_summary that describes WHAT analytical computation
        is being requested.

        Rules:
        - Do NOT generate SQL or pseudo-SQL.
        - Do NOT mention tables, joins, or implementation details unless necessary
        to disambiguate the intent.
        - The intent_summary must be a single sentence or two concise sentences.
        - Use precise analytical language (e.g., compute, compare, aggregate, growth).
        - Normalize vague phrases (e.g., "recent", "last", "latest") into clear analytical meaning.
        - If multiple computations are requested, clearly enumerate them in one intent_summary.
        - Always assume time-based calculations are anchored to the maximum available date
        unless explicitly stated otherwise.
        - Prefer declarative phrasing over question form.

        User Query INPUT (VERY IMPORTANT):
        {user_query}

        Examples:

        Input:
        "How is the demand trending?"
        Output:
        {{
        "intent_summary": "Show the national commercial demand trend over the most recent 52 weeks of available data, using weekly granularity based on week_end_date, combining commercial demand from data_867 and data_867_pap, and including weekly daily average demand, business day counts, and explicit week start and end boundaries."
        }}

        Input:
        "What is the YTD demand with pap?"
        Output:
        {{
        "intent_summary": "Calculate national year-to-date demand including PAP, using combined demand from data_867 and data_867_pap, filtered to valid orders only, for the current year determined from the latest available week_end_date. Because YTD is incomplete unless data reaches December 31, return total demand volume and daily average demand with business day count, explicit period boundaries, and the latest available date."
        }}
        Input:
        "What is the QTD budget with pap?" 
        Output:
        {{
        "intent_summary": "Calculate the current quarter-to-date demand budget including PAP at the national level using the forecast table. Since the query explicitly asks for budget with PAP, use forecast records scoped to Commercial plus PAP. Compute QTD budget as the sum of full prior months in the current quarter plus the prorated month-to-date budget for the current month based on business days elapsed in the current month. Return a single national-level row with period boundaries and business day counts."
        }}

        Input:
        "What is the YTD budget?"
        Output:
        {{
        "intent_summary": "Calculate the current year-to-date budget at the national parent level using the forecast table, defaulting to Commercial scope only. YTD budget must equal the sum of full monthly budget values for all prior months in the current year plus the prorated current-month budget based on business days elapsed in the current month. Return a single row with the YTD budget, current year boundaries, max week end date, and total YTD business days elapsed derived from data_867. Do not display current month boundary details."
        }}

        Input:
        "What is the YTD attainment?" 
        Output:
        {{
        "intent_summary": "Calculate national parent-level year-to-date attainment percentage for Commercial demand only by dividing YTD actual demand by YTD forecast/budget, using the current year determined from the latest available demand week_end_date and applying YTD forecast logic for an incomplete current month. Also calculate and display YTD business days elapsed from data_867 at the national level."
        }}

        Input:
        "give me the top 25 accounts?"
        Output:
        {{
        "intent_summary": "Return the top 25 parent accounts ranked by total Commercial + PAP demand vials from 2025-01-01 through the current available date in the data."
        }}

        Input:
        "give me longitudinal trend of the top 25 accounts" 
        Output:
        {{
        "intent_summary": " Produce a longitudinal weekly trend for the top 25 parent accounts, where top 25 is defined by total Commercial + PAP demand vials from 2025-01-01 through the current available date, and the trend is shown as a pivoted weekly view over the most recent 26 weeks with accounts as rows and actual week_end_date values as dynamic columns."
        }}

        Input:
        "Give me the list of reactivated accounts in the recent 4 weeks" 
        Output:
        {{
        "intent_summary": "Return the child-account level list of reactivated accounts for each of the 4 most recent completed weeks, using combined Commercial + PAP activity from data_867 and data_867_pap and applying the strict reactivated-account definition independently for each week, while also including the account's latest ordered date, latest ordered vials, parent name, region, and account_type"
        }}

        Input:
        "Give me the trend of new dormant accounts addition?"
        Output:
        {{
        "intent_summary": "Calculate the weekly trend of new dormant account additions for the most recent 52 weeks of available data, using child account activity from the combined Commercial + PAP demand sources. For each analysis week, compute weekly dormant accounts based on the exact 8-week dormancy rule, then derive new dormant additions as the count of accounts that are dormant in the current week but were not dormant in the immediately previous analysis week."
        }}

        Input:
        "What are the number of patients alerts by mtor therapy?" 
        Output:
        {{
        "intent_summary": "Count the number of MTOR patient therapy alert records by mtor therapy, using the mtor_alerts table and the default YTD period from January 1 of the current calendar year through the latest available date in the table."
        }}

        Input:
        "What is the growth of the top 75 accounts?" 
        Output:
        {{
        "intent_summary": "Calculate the combined growth for the top 75 parent accounts, where 'Top 75 accounts' is anchored to the parent_marketing_target table and growth defaults to Recent 8 Weeks versus Prior 8 Weeks. Because this is an account-level query, include Commercial + PAP demand by combining data_867 with all records from data_867_pap without transaction_type restriction. Rank parent accounts by combined actual demand in the recent 8-week period, keep the top 75 parent accounts, then sum demand across those top 75 accounts for both recent and prior 8-week periods and compute combined growth. Also include the corresponding absolute demand volumes for both periods and the combined total growth percentage."
        }}

        Input:
        "What is the account depth across regions?" 
        Output:
        {{
        "intent_summary": "Calculate account depth across regions for the default most recent 8 weeks of available data, using Commercial + PAP demand at the child account level, where depth equals total demand divided by the number of ordering child accounts."
        }}

        Input:
        "WHat is the revenue contribution across wholesalers?" 
        Output:
        {{
        "intent_summary": " Calculate revenue contribution across wholesalers using the revenue table as the source of truth, defaulting to the most recent 8 weeks of available data, grouped by wholesaler, with each wholesaler's net sales revenue and its share of total net sales revenue over the same period. Include corresponding ex-factory vials quantity alongside revenue metrics"
        }}

        Input:
        "WHat is the balance to go for the current year?" 
        Output:
        {{
        "intent_summary": "Calculate the current year balance to go for revenue at the national parent level as forecasted net sales for the current year minus actual net sales for the current year, using the current year determined from the most recent available week_end_date in the revenue table."
        }}

        Input:
        "What is the sales forecast for the current quarter?" 
        Output:
        {{
        "intent_summary": "Calculate the full current quarter sales forecast at the national level using the revenue_forecast table, where the current quarter is determined from the most recent available week_end_date in the revenue table. Because the user asked for the sales forecast for the current quarter and the prior quarter-to-date interpretation was rejected, return the full-quarter forecast by summing all daily net_sales_forecast rows that belong to the derived current quarter. Include the current quarter calendar boundaries and report a single row."
        }}

        Input:
        "Give me the new account activation trend for the recent 52 weeks" 
        Output:
        {{
        "intent_summary": "Calculate the weekly trend of new account activations for the most recent 52 weeks using the first observed valid commercial demand week for each child account as the activation event, then count how many accounts were newly activated in each week, while ensuring every week in the 52-week window appears in the output with 0 when there are no activations."
        }}

        Input:
        "How is the daily average demand trending across east region?" 
        Output:
        {{
        "intent_summary": "Calculate the weekly daily average commercial demand trend for the East region over the most recent 8 weeks of available data, using total commercial demand from data_867 plus commercial records from data_867_pap, and normalize each week's demand by the national-level count of distinct business days within that week."
        }}
        

        Return JSON only:
        {{"intent_summary": "<canonical intent>"}}
    """
    response = model.invoke([HumanMessage(content=prompt)])
    # usage = response.usage_metadata
    # input_tokens = usage.get("input_tokens", 0)
    # output_tokens = usage.get("output_tokens", 0)
    # total_tokens = usage.get("total_tokens", 0)
    # print("\n===== Intent Summary TOKEN USAGE =====")
    # print(f"Input Tokens: {input_tokens}")
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Total Tokens: {total_tokens}")
    raw_text = response.content.strip()

    try:
        parsed = json.loads(raw_text)
        return parsed["intent_summary"]
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse intent JSON: {raw_text}") from e
    except KeyError:
        raise ValueError(f"'intent_summary' missing in response: {raw_text}")

def search_snowflake(user_query, intent, top_k=7):
    intent_summary = intent

    print("Intent Summary")
    print(intent_summary)

    # Build embedding text (same as FAISS logic)
    final_query = f"""Intent: {intent_summary}
    User Question: {user_query}"""

    # SQL query
    sql = f"""
    SELECT
        run_id,
        question AS matched_question,
        query_decomposition,
        final_sql,
        VECTOR_COSINE_SIMILARITY(
            embedding,
            SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                $$ {final_query} $$
            )
        ) AS score
    FROM rag_payload
    ORDER BY score DESC
    LIMIT {top_k}
    """

    df = run_snowflake_query(sql)

    # Convert to FAISS-like output format
    results = []
    for _, row in df.iterrows():
        results.append({
            "score": float(row["SCORE"]),
            "run_id": row["RUN_ID"],
            "matched_question": row["MATCHED_QUESTION"],
            "query_decomposition": row["QUERY_DECOMPOSITION"],
            "final_sql": row["FINAL_SQL"],
        })

    return results


def build_rag_examples(user_input, intent):
    results = search_snowflake(
        user_query=user_input,
        intent=intent,
        top_k=7
    )

    sql_generator_rag_examples_text=f"""
        ────────────────────────
        EXAMPLES (FOR GUIDANCE NOT GENERATED BY RAG)
        ────────────────────────

        Example 1:
        User Question:
        "Total quantity sold by region in Q4-24"

        Expected SQL Output:
        SELECT campus_region, SUM(relmora_total_mg) AS total_qty
        FROM drug_sales
        WHERE quarter_year = 'Q4-24'
        GROUP BY campus_region;

        ────────────────────────

        Example 2:
        User Question:
        "Monthly quantity sold for Academic accounts"

        Expected SQL Output:
        SELECT month_year, SUM(relmora_total_mg) AS total_qty
        FROM drug_sales
        WHERE campus_account_type = 'Academic'
        GROUP BY month_year
        ORDER BY month_year;

   
"""
    query_decomposer_rag_examples_text=f"""
        ────────────────────────
        FINAL FULL EXAMPLE (NOT GENERATED BY RAG)
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
            "column": "relmora_total_mg",
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
"""
    relevant_questions=[]
    if results:
        threshold_index=-1
        print("---RAG Output---")
        for i, r in enumerate(results[0:3], start=1):
            if r['score']>=0.7:
                threshold_index=i
            print(f"#{i}  Score: {r['score']:.4f}")
            #print(f"Run ID: {r['run_id']}")
            print(f"Matched Question: {r['matched_question']}")
            # print("\n--- Query Decomposition ---")
            # qd = r["query_decomposition"]
            # # Convert if it's a string
            # if isinstance(qd, str):
            #     qd = json.loads(qd)
            # print(json.dumps(qd, indent=2, ensure_ascii=False))
            print("\n--- Final SQL ---")
            print(r["final_sql"])
            print("\n----------------------------\n")


        for it in results[3:]:
            relevant_questions.append(it["matched_question"].capitalize())

        if threshold_index==-1:
            results=None
        elif threshold_index>0:
            threshold_index=max(threshold_index,3)
            sql_generator_rag_examples_text = sql_generator_build_rag_examples_block(results[0:threshold_index])
            query_decomposer_rag_examples_text = query_decomposer_build_rag_examples_block(results[0:threshold_index])
    

        
        


    else:
        print("No RAG Examples Were Found for the Given Query")

    return sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions

from typing import Dict, Any


def build_chat_response(result: Dict[str, Any], relevant_questions, preview_rows: int = 10) -> str:
    """
    Builds a user-facing chat response string from LangGraph result state.

    Includes:
    - Generated SQL
    - Result summary
    - SQL executor output preview (tabular)

    Returns:
        str: Content safe to pass to AIMessage(content=...)
    """
    parts = []

    # 1. SQL Generator Output
    sql_query = result.get("sql_generator_output")
    if sql_query:
        parts.append("SQL Query Executed:")
        parts.append(sql_query)

    # 2. Result Summary
    result_summary = result.get("result_summary")
    if result_summary:
        parts.append("\nResult Summary:")
        parts.append(result_summary)

    # # 3. SQL Executor Output (preview)
    # executor_output = result.get("sql_executor_output")
    # if executor_output:
    #     df = pd.DataFrame(
    #     executor_output["data"],
    #     columns=executor_output["columns"]
    #     )
    #     # print("Result Data Frame")
    #     # print("-"*100)
    #     # print(df)
    #     content = (
    #         "Query Results:\n\n"
    #         + df.to_markdown(index=False)
            
    #     )

    #     parts.append(content)
    
    # visualization_code=result.get("visualization_code")
    # if visualization_code:
    #     parts.append("Visualization Code:")
    #     # print("Visualization Code from Build Chat Response")
    #     # print(visualization_code)
    #     parts.append(visualization_code)
    if len(relevant_questions)>0:
        formatted_questions = "\n".join(
            f"- {q}" for q in relevant_questions
        )
        parts.append("\nRelevant Questions:")
        parts.append(formatted_questions)

    return "\n".join(parts) if parts else "Completed"


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    



def build_chatbot(checkpointer):
    subgraph = build_graph(checkpointer=None)

    def chat_node(state: ChatState, config):
        messages = state["messages"]
        query=process_user_query(messages[-1].content)
        print("Corrected Query")
        print(query)
        intent=get_intent_summary(query)
        sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions =build_rag_examples(query,intent)


        initial_state = {
            "question": query,
            "messages": messages,
            "run_id": datetime.now(UTC).isoformat() + "Z",
            "last_output": "",
            "query_decomposer_output": None,
            "sql_generator_output": None,
            "sql_reviewer_output": None,
            "human_reviewer_output": None,
            "active_review": None,
            "query_decomposer_rag_examples_text":query_decomposer_rag_examples_text,
            "sql_generator_rag_examples_text":sql_generator_rag_examples_text,
            "result_summary":None,
            "sql_executor_output":None,
            "visualization_code":None,
            "trace": []
        }

        result = subgraph.invoke(initial_state, config=config)
        new_messages = []

        # 1️⃣ Main assistant response
        content = build_chat_response(result, relevant_questions)
        new_messages.append(AIMessage(content=content))

        # 2️⃣ SQL result as a structured message
        if result.get("sql_executor_output") is not None:
            new_messages.append(
                AIMessage(
                    content="SQL query results",
                    additional_kwargs={
                        "type": "sql_result",
                        "data": result["sql_executor_output"]
                    }
                )
            )

        # 3️⃣ Visualization as a structured message
        if result.get("visualization_code") is not None:
            new_messages.append(
                AIMessage(
                    content="Visualization",
                    additional_kwargs={
                        "type": "visualization",
                        "code": result["visualization_code"]
                    }
                )
            )

        return {
            "messages": new_messages
        }

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)
