from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from subgraph_7 import build_graph
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




def load_payload_store(payload_path: str) -> Dict[str, Any]:
    with open(payload_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
        "Compare the growth of the month Jan'26 and Dec'25?"
        Output:
        {{
        "intent_summary": "Compare relmora sales growth between Dec'25 and Jan'26, using total monthly sales if both months are complete; otherwise normalize both months to weekly average sales before calculating percent growth, and include absolute sales values for both months."
        }}

        Input:
        "How does our breadth look like?"
        Output:
        {{
        "intent_summary": "Assess breadth of relmora distribution/coverage by counting active campus accounts (relmora_total_mg > 0) and total shipped volume over the most recent 52 weeks of available data, at the national level (no campus grouping)."
        }}

        Input:
        "How are we doing in terms of adding new businesses?" 
        Output:
        {{
        "intent_summary": "Assess performance in adding new campus (child) businesses by counting newly active campus_ids in the most recent 13 weeks versus the prior 13 weeks, including absolute counts and growth percent, using only accounts with relmora activity (relmora_total_mg > 0)."
        }}

        Input:
        "How does new account addition look like across regions?"
        Output:
        {{
        "intent_summary": "Show new campus account additions across campus regions over the most recent 52 weeks of available data, where a new account is defined as a campus_id whose first observed week_end_date with relmora_total_mg > 0 occurs within the window."
        }}

        Input:
        "Compare the growth in Q3'25 with Q4'25?" 
        Output:
        {{
        "intent_summary": "Compare relmora sales growth between calendar Q3-25 and Q4-25, reporting both absolute sales and growth percentage using totals if both quarters are complete, otherwise using weekly-average normalized sales for both quarters."
        }}

        Input:
        "Is short-term sales performance consistent across regions?"
        Output:
        {{
        "intent_summary": "Assess whether short-term relmora sales performance is consistent across regions by comparing recent 4 week_end_date periods versus prior 4 week_end_date periods at the campus-region level, including regional and national sales metrics, period boundaries based strictly on week_end_date values, and a region-versus-nation performance flag calculated using daily average growth."
        }}

        Input:
        "How does account adoption look like within target campuses across regions?" 
        Output:
        {{
        "intent_summary": "Assess account adoption within target campuses across regions at the campus entity level using the most recent 13 weeks of available data. Adoption must be expressed as adoption_rate_r13w = active_accounts_r13w / total_target_accounts_r13w, where active accounts are distinct target campus_ids with positive relmora volume during the period and total target accounts are all distinct target campus_ids in each target campus region. Remove all sales metrics from the output."
        }}

        Input:
        "Are we gaining or losing share by tier across regions?" 
        Output:
        {{
        "intent_summary": "Assess whether market share is being gained or lost by campus tier across regions by comparing recent 3 months versus prior 3 months using the market share dataset, and add a performance-vs-nation column based on whether each tier-region segment's share change is higher or lower than the national share change."
        }}


        Return JSON only:
        {{"intent_summary": "<canonical intent>"}}
    """
    response = model.invoke([HumanMessage(content=prompt)])
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
        intent=get_intent_summary(messages[-1].content)
        sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions =build_rag_examples(messages[-1].content,intent)

        initial_state = {
            "question": messages[-1].content,
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
