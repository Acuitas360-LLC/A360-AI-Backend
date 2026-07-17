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

    decomposer = bundle.get("decomposer_json") or {}

    intent = decomposer.get("intent_summary")
    if intent:
        lines.append(f"Intent: {intent}")

    question = bundle.get("question", "")
    if question:
        lines.append(f"User Question: {question}")

    return "\n".join(lines).strip()


# =========================================================
# 5) Build FAISS index + Payload store
# =========================================================

def build_payload(agent_trace_path: str,
                     out_payload_path: str) -> None:

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
            # "sql_reviewer_text": bundle["sql_reviewer_text"],
            # "sql_executor_text": bundle["sql_executor_text"],
            "human_verdict": "SUCCESS",
            "embedding text":embedding_text
        }

    if not embedding_texts:
        raise ValueError("No SUCCESS runs found. FAISS index not built.")


    # Save payload store (IMPORTANT: keep ids order)
    out_obj = {
        "ids_in_order": ids,
        "payload_store": payload_store
    }

    with open(out_payload_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    
    print(f"✅ Payload store saved: {out_payload_path}")
    print(f"✅ Total indexed SUCCESS records: {len(ids)}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_agent_trace.json", help="Path to agent_trace.json")
    parser.add_argument("--payload_out", default="payload_store_snowflake.json", help="Output payload store file")
    args = parser.parse_args()

    build_payload(
        agent_trace_path=args.input,
        out_payload_path=args.payload_out,    
    )


if __name__ == "__main__":
    main()
