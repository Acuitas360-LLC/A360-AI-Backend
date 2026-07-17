import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import numpy as np
import faiss
from openai import OpenAI
load_dotenv()

# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

# =========================================================
# 1) Extraction Helpers (JSON + SQL)
# =========================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from:
    - ```json ... ``` blocks
    - raw JSON string
    - text that contains embedded JSON
    """
    if not text:
        return None

    raw = text.strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            return None

    return None


def extract_sql(text: str) -> str:
    """
    Extract SQL from:
    - ```sql ... ```
    - ``` ... ```
    - plain SQL
    """
    if not text:
        return ""

    raw = text.strip()

    m = re.search(r"```sql\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()

    return raw


# =========================================================
# 2) Trace utilities (last SUCCESS + closest previous outputs)
# =========================================================

def find_last_success_index(trace: List[Dict[str, Any]]) -> Optional[int]:
    last_idx = None
    for i, ev in enumerate(trace):
        if ev.get("agent") == "human_reviewer":
            if str(ev.get("text", "")).strip().upper() == "SUCCESS":
                last_idx = i
    return last_idx


def find_last_event_before(trace: List[Dict[str, Any]], idx: int, agent_name: str) -> Optional[Dict[str, Any]]:
    for j in range(idx, -1, -1):
        if trace[j].get("agent") == agent_name:
            return trace[j]
    return None


def extract_final_success_bundle(chat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trace = chat.get("trace", [])
    success_idx = find_last_success_index(trace)
    if success_idx is None:
        return None

    decomposer_ev = find_last_event_before(trace, success_idx, "query_decomposer")
    sqlgen_ev = find_last_event_before(trace, success_idx, "SQL_Generator")
    reviewer_ev = find_last_event_before(trace, success_idx, "sql_reviewer")
    executor_ev = find_last_event_before(trace, success_idx, "sql_executor")

    decomposer_json = extract_json_from_text(decomposer_ev.get("text", "")) if decomposer_ev else None
    final_sql = extract_sql(sqlgen_ev.get("text", "")) if sqlgen_ev else ""

    return {
        "run_id": chat.get("run_id", ""),
        "question": chat.get("question", "").strip(),
        "decomposer_json": decomposer_json,
        "final_sql": final_sql,
        "sql_reviewer_text": reviewer_ev.get("text", "").strip() if reviewer_ev else "",
        "sql_executor_text": executor_ev.get("text", "").strip() if executor_ev else "",
        "human_verdict": "SUCCESS"
    }


# =========================================================
# 3) Build embedding text
# =========================================================

def build_embedding_text(bundle: Dict[str, Any]) -> str:
    """
    Build embedding text with rules:
    - include group_by/order_by only if non-empty
    - include limit only if not null
    """
    lines = []

    question = bundle.get("question", "")
    if question:
        lines.append(f"User Question: {question}")

    decomposer = bundle.get("decomposer_json") or {}

    intent = decomposer.get("intent_summary")
    if intent:
        lines.append(f"Intent: {intent}")

    tables = decomposer.get("tables") or []
    if tables:
        lines.append(f"Tables: {', '.join(tables)}")

    filters = decomposer.get("filters") or []
    if filters:
        lines.append("Filters:")
        for f in filters:
            col = f.get("column", "")
            op = f.get("operator", "")
            val = f.get("value", "")
            if col and op and val != "":
                lines.append(f"- {col} {op} {val}")

    aggs = decomposer.get("aggregations") or []
    if aggs:
        lines.append("Aggregations:")
        for a in aggs:
            fn = a.get("function", "")
            col = a.get("column", "")
            metric = a.get("metric_name", "")
            group_level = a.get("group_level", "")
            s = f"- {fn}({col})"
            if metric:
                s += f" AS {metric}"
            if group_level:
                s += f" | group_level={group_level}"
            lines.append(s)

    subs = decomposer.get("subqueries") or []
    if subs:
        lines.append("Subquery Patterns:")
        for s in subs:
            name = s.get("name", "")
            purpose = s.get("purpose", "")
            logic = s.get("logic", "")
            parts = []
            if name:
                parts.append(f"name={name}")
            if purpose:
                parts.append(f"purpose={purpose}")
            if logic:
                parts.append(f"logic={logic}")
            if parts:
                lines.append("- " + " | ".join(parts))

    # group_by (skip if empty)
    group_by = decomposer.get("group_by")
    if group_by:
        if isinstance(group_by, list) and len(group_by) > 0:
            lines.append(f"Group By: {', '.join(group_by)}")
        elif isinstance(group_by, str) and group_by.strip():
            lines.append(f"Group By: {group_by.strip()}")

    # order_by (skip if empty)
    order_by = decomposer.get("order_by")
    if order_by:
        if isinstance(order_by, list) and len(order_by) > 0:
            lines.append(f"Order By: {', '.join(map(str, order_by))}")
        elif isinstance(order_by, str) and order_by.strip():
            lines.append(f"Order By: {order_by.strip()}")

    # limit (skip if null)
    limit_val = decomposer.get("limit", None)
    if limit_val is not None:
        lines.append(f"Limit: {limit_val}")

    final_output = decomposer.get("final_output") or {}
    if final_output:
        cols = final_output.get("columns") or []
        gran = final_output.get("row_granularity")
        parts = []
        if cols:
            parts.append(f"columns=[{', '.join(cols)}]")
        if gran:
            parts.append(f"row_granularity={gran}")
        if parts:
            lines.append("Final Output: " + " | ".join(parts))

    rules = decomposer.get("validation_rules") or []
    if rules:
        lines.append("Validation Rules:")
        for r in rules:
            lines.append(f"- {r}")

    sql = bundle.get("final_sql", "")
    if sql:
        lines.append("SQL:")
        lines.append(sql)

    reviewer = bundle.get("sql_reviewer_text", "")
    if reviewer:
        lines.append("Reviewer:")
        lines.append(reviewer)

    lines.append("Human Verdict: SUCCESS")

    return "\n".join(lines).strip()


# =========================================================
# 4) OpenAI Embeddings
# =========================================================

def openai_embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Returns float32 numpy array of shape (N, dim)
    Normalized for cosine similarity usage with IndexFlatIP.
    """
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client=OpenAI()

    vectors = []
    # batching helps performance + cost
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vectors = [d.embedding for d in resp.data]
        vectors.extend(batch_vectors)

    vecs = np.array(vectors, dtype="float32")

    # Normalize so cosine similarity works with inner product
    faiss.normalize_L2(vecs)
    return vecs


# =========================================================
# 5) Build FAISS index + Payload store
# =========================================================

def build_faiss_index(agent_trace_path: str,
                     out_index_path: str,
                     out_payload_path: str,
                     embedding_model: str = "text-embedding-3-small") -> None:

    with open(agent_trace_path, "r", encoding="utf-8") as f:
        chats = json.load(f)

    embedding_texts = []
    ids = []
    payload_store = {}

    for chat in chats:
        bundle = extract_final_success_bundle(chat)
        if not bundle:
            continue

        run_id = bundle["run_id"]
        embedding_text = build_embedding_text(bundle)

        ids.append(run_id)
        embedding_texts.append(embedding_text)

        payload_store[run_id] = {
            "run_id": run_id,
            "question": bundle["question"],
            "query_decomposition": bundle["decomposer_json"],
            "final_sql": bundle["final_sql"],
            "sql_reviewer_text": bundle["sql_reviewer_text"],
            "sql_executor_text": bundle["sql_executor_text"],
            "human_verdict": "SUCCESS",
            "embedding text":embedding_text
        }

    if not embedding_texts:
        raise ValueError("No SUCCESS runs found. FAISS index not built.")

    # Create embeddings using OpenAI
    vectors = openai_embed_texts(embedding_texts, model=embedding_model)
    dim = vectors.shape[1]

    # Cosine similarity index
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, out_index_path)

    # Save payload store (IMPORTANT: keep ids order)
    out_obj = {
        "embedding_model": embedding_model,
        "faiss_index_type": "IndexFlatIP (cosine via L2 normalization)",
        "ids_in_faiss_order": ids,
        "payload_store": payload_store
    }

    with open(out_payload_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS index saved: {out_index_path}")
    print(f"✅ Payload store saved: {out_payload_path}")
    print(f"✅ Total indexed SUCCESS records: {len(ids)}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_agent_trace.json", help="Path to agent_trace.json")
    parser.add_argument("--index_out", default="faiss_index.bin", help="Output FAISS index file")
    parser.add_argument("--payload_out", default="payload_store.json", help="Output payload store file")
    parser.add_argument("--embedding_model", default="text-embedding-3-small",
                        help="OpenAI embedding model name")
    args = parser.parse_args()

    build_faiss_index(
        agent_trace_path=args.input,
        out_index_path=args.index_out,
        out_payload_path=args.payload_out,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
