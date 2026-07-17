from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from subgraph_Crinetics import build_graph
from datetime import datetime, UTC
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List
import numpy as np
import json
import os
from openai import OpenAI
import plotly.express as px
import snowflake.connector
from typing import TypedDict, Literal, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# Access the key
load_dotenv()
model = ChatOpenAI(model="gpt-5.4")
openai_api_key = os.getenv("OPENAI_API_KEY")
from rapidfuzz import process, fuzz
from collections import defaultdict

def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


SNOWFLAKE_CONFIG = {
        "user": _env_or_default("SNOWFLAKE_USER", "ahusain"),
        "password": _env_or_default("SNOWFLAKE_PASSWORD", "Murtaza@40401059"),
        "account": _env_or_default("SNOWFLAKE_ACCOUNT", "ua60309.south-central-us.azure"),
        "warehouse": _env_or_default("SNOWFLAKE_WAREHOUSE", "PALSONIFY_COMPUTE"),
        "database": _env_or_default("SNOWFLAKE_DATABASE", "PALSONIFY"),
        "schema": _env_or_default("SNOWFLAKE_SCHEMA", "PALSONIFY_SCHEMA")
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
    conn = None
    cursor = None

    try:
        conn   = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()

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

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return MASKING_TABLE_DICT

# Define your fallback column priority order here
COLUMN_FALLBACK_ORDER = [
    "region",
    "area",
    "territory", 
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
        "Give me the total number of enrollments"
        Output:
        {{
        "intent_summary": "Calculate the total number of enrollments for the default quarter-to-date period anchored to the most recent available week_end_date, using COUNT of crinetics_id from the enrollments dataset."
        }}

        Input:
        ""How are my enrollments trending?"
        Output:
        {{
        "intent_summary": "Show the year-to-date weekly trend of enrollments using count of crinetics_id, anchored to the year containing the latest available week_end_date, and include both weekly enrollment volume and the cumulative sum of weekly enrollments with explicit weekly time boundaries."
        }}

        Input:
        "Give me the total number of dispenses" 
        Output:
        {{
        "intent_summary": "Calculate the total number of dispenses using the Dispense table, defaulting to quarter-to-date for the most recent available quarter, and summing bottles_dispensed."
        }}

        Input:
        "How are the dispenses trending?"
        Output:
        {{
        "intent_summary": "Show the year-to-date weekly trend of total dispenses using the Dispense table, where total dispenses are calculated as the sum of bottles_dispensed, and also include the cumulative year-to-date running total by week, anchored to the current year in the data with explicit weekly time boundaries."
        }}

        Input:
        "How does enrollments look like across regions?" 
        Output:
        {{
        "intent_summary": "Show the year-to-date weekly trend of total dispenses using the Dispense table, where total dispenses are calculated as the sum of bottles_dispensed, and also include the cumulative year-to-date running total by week, anchored to the current year in the data with explicit weekly time boundaries."
        }}

        Input:
        "how does the dispenses contribution look like by each region"
        Output:
        {{
        "intent_summary": "Calculate dispense contribution by region for the default quarter-to-date reporting period, where dispense contribution is defined as total bottles from both Dispense and SD_SHIPMENTS combined, and report each region's share of the combined national total along with absolute volume and time boundaries."
        }}

        Input:
        "Top 10 territories by enrollments." 
        Output:
        {{
        "intent_summary": "Return the top 10 territories ranked by quarter-to-date enrollments from the most recent available quarter in the enrollments dataset, using count of crinetics_id and including quarter boundary dates."
        }}

        Input:
        "What is the activation trend of non targeted HCPs?" 
        Output:
        {{
        "intent_summary": "Calculate the year-to-date weekly activation trend for non targeted HCPs from the enrollments dataset, where activation is defined as an HCP whose first-ever enrollment transaction date falls within the current year-to-date period, restricted to non targeted HCPs with tier = 'N', and include the cumulative sum alongside the weekly trend. Week boundaries must align to the week_end_date values present in the data, with week_start_date defined as week_end_date minus 6 days, except when January 1st of the current year falls inside that weekly boundary, in which case week_start_date must be overridden to January 1st of the current year."
        }}

        Input:
        "Give me the list of top63 accounts not activated so far"
        Output:
        {{
        "intent_summary": "Return the list of Top 63 parent accounts that are not activated so far, using the Top 63 target/master list anchored to parent_marketting_target and identifying activated accounts as the union of parent_ids from enrollments and sd_shipments across all available history. Preserve all target-list accounts and return one row per parent_id."
        }}

        Input:
        "Give me the list of hcps that have adopted our drug so far." 
        Output:
        {{
        "intent_summary": "Return the list of HCPs who have adopted the drug so far, interpreting 'adopted' as having at least one enrollment record and 'so far' as Launch to Date across the full available transaction date range. Output one row per unique HCP with required HCP-level fields and the number of enrollments brought."
        }}

        Input:
        "What is the reach across territories?" 
        Output:
        {{
        "intent_summary": "Calculate call-based reach across territories using the target HCP universe from marketting_target and reached HCPs from calls_data, defaulting to quarter-to-date for the most recent quarter because no time period was specified."
        }}

        Input:
        "What is the nation reach?" 
        Output:
        {{
        "intent_summary": "Calculate national reach as the distinct count of target HCP NPIs reached through calls divided by the distinct count of total target HCP NPIs, using the default quarter-to-date period anchored to the most recent quarter available in the data."
        }}

        Input:
        "What is the call frequency across regions?" 
        Output:
        {{
        "intent_summary": "Calculate call frequency by region using calls_data and marketting_target, where frequency is total calls to reached target HCPs divided by distinct reached target HCP NPIs. Because no time period is specified, default to quarter-to-date using the most recent quarter_year. Include only Face To Face, Phone, and Video channels by default and restrict to target HCPs in tiers 1-4."

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

import pandas as pd
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
        #query=process_user_query(messages[-1].content)
        query=(messages[-1].content)
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
