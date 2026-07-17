from langgraph.checkpoint.postgres import PostgresSaver

# DB_URI = "postgres://snowflake_admin:JNFdgf8aMEaawpuuce29tXV3jTHiicdoh5XZiMednIXFaJZs24Rlg7YpeBhMFRzJ@q6ayf7u5tbf6bob22izfsd4gga.skondys-et17731.southcentralus.azure.postgres.snowflake.app:5432/postgres?sslmode=require"

DB_URI="postgres://snowflake_admin:sGTA5xZq2QMAg6i1ITE2TbhjMHkJRd0Cd3JW1u4lFg36YYjmkLQ78ahIy6fh6hwU@2fnho2lztja33mfqwyh3q62lxm.skondys-et17731.southcentralus.azure.postgres.snowflake.app:5432/postgres?sslmode=require"

try:
    import psycopg
    conn = psycopg.connect(DB_URI, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    print("✅ Connection and setup successful!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")