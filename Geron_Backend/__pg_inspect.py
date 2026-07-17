import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv(Path('.env'))

db_uri = (
    os.getenv('DB_URI', '').strip()
    or os.getenv('POSTGRES_URL', '').strip()
    or os.getenv('POSTGRES_URI', '').strip()
)
print('DB_URI configured:', bool(db_uri))

conn = psycopg.connect(db_uri)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
print('TABLES:', tables)

targets = ['thread_registry', 'hidden_threads', 'thread_message_cache', 'message_feedback']
for t in targets:
    if t in tables:
        cur.execute(f'SELECT COUNT(*) FROM {t}')
        print(f'{t} count:', cur.fetchone()[0])

# print('\nLatest thread_registry rows:')
# if 'thread_registry' in tables:
#     cur.execute("SELECT * FROM thread_registry ORDER BY created_at DESC NULLS LAST LIMIT 4")
    
#     rows = cur.fetchall()
#     columns = [desc[0] for desc in cur.description]

#     for idx, row in enumerate(rows, start=1):
#         print(f"\n========== Row {idx} ==========")
#         for col, val in zip(columns, row):
#             print(f"{col}: {val}")


# print('\nLatest hidden_threads rows:')
# if 'hidden_threads' in tables:
#     cur.execute("SELECT user_id, thread_id, hidden_at FROM hidden_threads ORDER BY hidden_at DESC NULLS LAST LIMIT 10")
#     for row in cur.fetchall():
#         print(row)

# print('\nLatest thread_message_cache rows:')
# if 'thread_message_cache' in tables:
#     cur.execute("SELECT * FROM thread_message_cache ORDER BY updated_at DESC LIMIT 4")
    
#     rows = cur.fetchall()
#     columns = [desc[0] for desc in cur.description]

#     for idx, row in enumerate(rows, start=1):
#         print(f"\n========== Row {idx} ==========")
#         for col, val in zip(columns, row):
#             print(f"{col}: {val}")

# print('\nLatest message_feedback rows:')

if 'message_feedback' in tables:
    # cur.execute("""
    #     SELECT
    #         id,
    #         thread_id,
    #         message_id,
    #         user_query,
    #         assistant_response,
    #         rating,
    #         created_at,
    #         user_id,
    #         feedback_text,
    #         updated_at,
    #         enriched_at,
    #         feedback_query_message_id,
    #         feedback_response_message_id,
    #         followup_questions,
    #         enrich_status,
    #         enrich_attempts
    #     FROM message_feedback
    #     ORDER BY created_at DESC
    #     LIMIT 1
    # """)
    cur.execute("""
        SELECT feedback_text, enrich_status, enrich_attempts, created_at
        FROM message_feedback
        ORDER BY created_at DESC
        LIMIT 1
    """)

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    for idx, row in enumerate(rows, start=1):
        print(f"\n========== Row {idx} ==========")
        for col, val in zip(columns, row):
            print(f"{col}: {val}")

conn.close()
