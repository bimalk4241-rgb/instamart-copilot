
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("data/instamart.db")

# Read all SQL queries
with open("sql/queries.sql", "r", encoding="utf-8") as file:
    sql_script = file.read()

queries = sql_script.split(";")

# Execute each query separately
for i, query in enumerate(queries, start=1):
    if query.strip():
        print(f"\n{'=' * 50}")
        print(f"QUERY {i}")
        print(f"{'=' * 50}")

        result = pd.read_sql_query(query, conn)

        print(result.to_string(index=False))
        print(f"\nNumber of rows: {len(result)}")

conn.close()

