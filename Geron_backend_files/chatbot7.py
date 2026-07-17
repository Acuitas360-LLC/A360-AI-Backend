from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from subgraph5 import build_graph
from datetime import datetime, UTC
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List
import numpy as np
import faiss
import json
import os
from openai import OpenAI
import plotly.express as px
from typing import TypedDict, Literal, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# Access the key
load_dotenv()
model = ChatOpenAI(model="gpt-5.2")
openai_api_key = os.getenv("OPENAI_API_KEY")

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

def openai_embed_query(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Embed a single query using OpenAI and return a normalized float32 vector (1, dim)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


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
        "how much drug grown in recent 4 weeks and how that compared with growth in recent 8 weeks"
        Output:
        {{
        "intent_summary": "Compute drug unit sales growth for the most recent 4 weeks versus the preceding 4 weeks and for the most recent 8 weeks versus the preceding 8 weeks (anchored to the max week_end_date), then compare the two growth rates by reporting their percentage-point difference and indicating whether the 4-week growth rate is higher or the 8-week growth rate is higher."
        }}

        Input:
        "what is the sales contribution of parent accounts which are academic, IDNs and Community"
        Output:
        {{
        "intent_summary": "Compute the percentage contribution to national sales for parent account types Academic, IDN (including Other), and Community (including Unknown) over the most recent 52 weeks."
        }}

        Input:
        "Among the top 15 parent accounts by sales contribution in the most recent 52 weeks, estimate the growth in recent 13 weeks and compare it to nation" 
        Output:
        {{
        "intent_summary": "Identify the top 15 parent accounts by 52-week sales contribution (most recent 52 weeks), then calculate each parent’s 13-week growth and compare it to national growth, including underlying sales volumes for 52-week and 13-week periods."
        }}

        Input:
        "how many accounts ordered our product in LTD, YTD, MTD and QTD" 
        Output:
        {{
        "intent_summary": "Return the number of distinct parent accounts that ordered the product for Life-To-Date (LTD), Year-To-Date (YTD), Quarter-To-Date (QTD), and Month-To-Date (MTD), each bounded by the dataset's most recent week_end_date and aligned to the month_year and quarter_year labels."
        }}

        Input:
        "Among the top 15 parent accounts by sales contribution in the most recent 52 weeks, identify any accounts that have declined in the most recent 13 weeks compared to the prior 13 weeks." 
        Output:
        {{
        "intent_summary": "Identify parent accounts that are in the top 15 by national sales contribution over the most recent 52 weeks and have declined in the most recent 13 weeks versus the prior 13 weeks."
        }}
        
        Input:
        "Estimate the sales growth by Area and Parent Account Type in recent 4 weeks compared to previous 4 weeks" 
        Output:
        {{
        "intent_summary": "Compute sales growth percentage by Area and Parent Account Type comparing Recent 4 Weeks (R4W) versus Previous 4 Weeks (P4W), using week_end_date anchored to the most recent week in the dataset, applying parent account type mergers"
        }}

        Input:
        "Show Q4 2024 new accounts by account type with contribution to total?" 
        Output:
        {{
        "intent_summary": "Identify new parent accounts whose first-ever shipment occurred in Q4 2024, aggregate the count by parent account type (with required type merging), and compute each type's contribution to the total new accounts."
        }}

        Input:
        "what is the sales growth in recent 4 weeks compared to previous 5 weeks  by BC potential segment and how does that compare with nation?" 
        Output:
        {{
        "intent_summary": "Compute sales growth by parent_bc_segment for the most recent 4 weeks versus the immediately preceding 5 weeks, normalize to weekly averages due to unequal period lengths, compare each segment's growth to national growth, and include the weekly average values for both periods at both the segment and national levels in the output."
        }}

        Input:
        "Provide weekly sales trend by area and nation" 
        Output:
        {{
        "intent_summary": "Produce a weekly sales trend for the most recent 52 weeks with one row per week_end_date and sales by geography (areas and national total) presented as separate columns."
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

def search_faiss(
    query: str,
    index_path: str = "faiss_index3.bin",
    payload_path: str = "payload_store3.json",
    top_k: int = 2
) -> List[Dict[str, Any]]:
    """
    Returns top-k retrieved examples with their payloads.
    """
    payload_obj = load_payload_store(payload_path)
    embedding_model = payload_obj.get("embedding_model", "text-embedding-3-small")

    ids_in_order = payload_obj["ids_in_faiss_order"]
    payload_store = payload_obj["payload_store"]

    # Load FAISS index
    index = faiss.read_index(index_path)

    intent_summary=get_intent_summary(query)
    final_query=f"Intent: {intent_summary}\nUser question: {query}"



    # Embed query
    qvec = openai_embed_query(final_query, model=embedding_model)

    # Search
    distances, indices = index.search(qvec, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if score>=0.7:
            run_id = ids_in_order[idx]
            payload = payload_store.get(run_id, {})

            results.append({
                "score": float(score),
                "run_id": run_id,
                "matched_question": payload.get("question", ""),
                "query_decomposition": payload.get("query_decomposition"),
                "final_sql": payload.get("final_sql", ""),
                "sql_reviewer_text": payload.get("sql_reviewer_text", ""),
                "human_verdict": payload.get("human_verdict", "")
            })

    return results

def build_rag_examples(user_input):
    results = search_faiss(
        query=user_input,
        index_path="faiss_index3.bin",
        payload_path="payload_store3.json",
        top_k=4
    )

    sql_generator_rag_examples_text=f"""
        ────────────────────────
        EXAMPLES (FOR GUIDANCE NOT GENERATED BY RAG)
        ────────────────────────

        Example 1:
        User Question:
        "Total quantity sold by region in Q4-24"

        Expected SQL Output:
        SELECT parent_regn_nm, SUM(qty_sold_pu) AS total_qty
        FROM drug_sales
        WHERE quarter_year = 'Q4-24'
        GROUP BY parent_regn_nm;

        ────────────────────────

        Example 2:
        User Question:
        "Monthly quantity sold for high business segment accounts"

        Expected SQL Output:
        SELECT month_year, SUM(qty_sold_pu) AS total_qty
        FROM drug_sales
        WHERE parent_bc_segment = 'High'
        GROUP BY month_year
        ORDER BY month_year;

   
"""
    query_decomposer_rag_examples_text=f"""
        ────────────────────────
        FINAL FULL EXAMPLE (NOT GENERATED BY RAG)
        ────────────────────────

        {{
        "intent_summary": "Calculate total sales for the last 13 weeks based on the most recent date available in the dataset.",
        "tables": ["drug_sales"],
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
            "column": "qty_sold_pu",
            "group_level": "none"
            }}
        ],
        "subqueries": [
            {{
            "name": "max_date_cte",
            "purpose": "Identify the most recent week_end_date in the dataset",
            "logic": "Compute MAX(week_end_date) from drug_sales"
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
        print("---RAG Output---")
        for i, r in enumerate(results[0:3], start=1):
            print(f"#{i}  Score: {r['score']:.4f}")
            print(f"Run ID: {r['run_id']}")
            print(f"Matched Question: {r['matched_question']}")
            print(f"Human Verdict: {r['human_verdict']}")
            print("\n--- Query Decomposition ---")
            print(json.dumps(r["query_decomposition"], indent=2, ensure_ascii=False))
            print("\n--- Final SQL ---")
            print(r["final_sql"])
            print("\n--- SQL Reviewer ---")
            print(r["sql_reviewer_text"])
            print("\n----------------------------\n")

        sql_generator_rag_examples_text = sql_generator_build_rag_examples_block(results[0:3])
        query_decomposer_rag_examples_text = query_decomposer_build_rag_examples_block(results[0:3])
        for it in results:
            relevant_questions.append(it["matched_question"].capitalize())


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
        sql_generator_rag_examples_text, query_decomposer_rag_examples_text, relevant_questions =build_rag_examples(messages[-1].content)


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
            "visualization_spec":None,
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

        # 4️⃣ Visualization spec as a structured message (new separate node output)
        if result.get("visualization_spec") is not None:
            new_messages.append(
                AIMessage(
                    content="Visualization Spec",
                    additional_kwargs={
                        "type": "visualization_spec",
                        "spec": result["visualization_spec"]
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
