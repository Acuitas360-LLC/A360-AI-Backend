from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://snowflake_admin:D6jHttcNb8T1zXa4dWMwHwQiuGhtD0m2mljCdgKIfMI4Y9vTwrRxuenEUNBCdBDd@6vdjofao3jarzfssqxyteizsfq.skondys-et17731.southcentralus.azure.postgres.snowflake.app:5432/postgres"

try:
    import psycopg
    conn = psycopg.connect(DB_URI, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    print("✅ Connection and setup successful!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")