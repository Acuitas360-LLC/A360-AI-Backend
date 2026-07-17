import os
import json
import argparse
from typing import Any, Dict, List
from dotenv import load_dotenv
import numpy as np
import faiss
from openai import OpenAI
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
model = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
)
# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

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
        "how much trodelvy grown in recent 4 weeks and how that compared with growth in recent 8 weeks"
        Output:
        {{
        "intent_summary": "Compute Trodelvy unit sales growth for the most recent 4 weeks versus the preceding 4 weeks and for the most recent 8 weeks versus the preceding 8 weeks (anchored to the max week_end_date), then compare the two growth rates by reporting their percentage-point difference and indicating whether the 4-week growth rate is higher or the 8-week growth rate is higher."
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
    index_path: str = "faiss_index2.bin",
    payload_path: str = "payload_store2.json",
    top_k: int = 3
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
    print("Intent Summary")
    print(intent_summary)
    final_query=f"Intent: {intent_summary}\nUser question: {query}"
    # Embed query
    qvec = openai_embed_query(final_query, model=embedding_model)

    # Search
    distances, indices = index.search(qvec, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

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


def main():
    parser = argparse.ArgumentParser()
    query=input("Enter your Query: ")
    parser.add_argument("--index_path", default="faiss_index2.bin", help="FAISS index file path")
    parser.add_argument("--payload_path", default="payload_store2.json", help="Payload store JSON path")
    parser.add_argument("--top_k", type=int, default=3, help="Number of nearest matches")
    args = parser.parse_args()

    results = search_faiss(
        query=query,
        index_path=args.index_path,
        payload_path=args.payload_path,
        top_k=args.top_k
    )


    print("RAG Retrieval Results")
    

    for i, r in enumerate(results, start=1):
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


if __name__ == "__main__":
    main()
