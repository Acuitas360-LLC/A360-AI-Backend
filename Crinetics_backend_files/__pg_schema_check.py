import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

load_dotenv(Path('backend/.env'))

db_uri = (
    os.getenv('DB_URI', '').strip()
    or os.getenv('POSTGRES_URL', '').strip()
    or os.getenv('POSTGRES_URI', '').strip()
)
conn = psycopg.connect(db_uri)
cur = conn.cursor()

tables = ['thread_registry', 'hidden_threads', 'thread_message_cache', 'message_feedback']
print('Columns (name:type) by table:')
for t in tables:
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        ORDER BY ordinal_position
        """,
        (t,),
    )
    cols = cur.fetchall()
    print(f'\n{t}:')
    for c in cols:
        print(' ', c)

print('\nIndexes:')
cur.execute(
    """
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE schemaname='public'
      AND tablename IN ('thread_registry','hidden_threads','thread_message_cache','message_feedback')
    ORDER BY tablename, indexname
    """
)
for idx in cur.fetchall():
    print(idx[0])

conn.close()
