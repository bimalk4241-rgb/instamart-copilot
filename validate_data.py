
import pandas as pd
import sqlite3

# Load cleaned data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

# Validate customer records
assert len(customers) == 1200
assert customers["city"].nunique() == 6
assert customers["age"].isna().sum() == 0
assert customers["gender"].isna().sum() == 0

# Validate product records
assert len(products) == 40

# Validate cleaned orders
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309
assert len(completed) == 11110
assert len(cancelled) == 1199
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert completed["delivery_minutes"].isna().sum() == 0
assert cancelled["delivery_minutes"].isna().sum() == 1199

# Validate revenue
assert abs(
    completed["order_amount"].sum() - 2998668.80
) < 0.01

# Validate SQLite database
with sqlite3.connect("data/instamart.db") as conn:
    for table, expected in [
        ("customers", 1200),
        ("products", 40),
        ("orders", 12309)
    ]:
        actual = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        assert actual == expected, f"{table}: {actual} rows"

print("ALL CHECKPOINTS PASSED")
