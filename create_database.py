
import pandas as pd
import sqlite3
from pathlib import Path

# 1. Load the cleaned customer and order datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Products have no cleaning changes, but still need a clean copy
products = pd.read_csv("data/raw/products.csv")

Path("data/clean").mkdir(parents=True, exist_ok=True)
products.to_csv("data/clean/products.csv", index=False)

# 2. Create the SQLite database
connection = sqlite3.connect("data/instamart.db")

# 3. Store the DataFrames as SQL tables
customers.to_sql("customers", connection, if_exists="replace", index=False)
products.to_sql("products", connection, if_exists="replace", index=False)
orders.to_sql("orders", connection, if_exists="replace", index=False)

# 4. Confirm records were stored
for table in ["customers", "products", "orders"]:
    result = pd.read_sql_query(
        f"SELECT COUNT(*) AS total FROM {table}",
        connection
    )
    print(f"{table}: {result['total'].iloc[0]} rows")

connection.close()

print("\nDatabase created successfully!")
