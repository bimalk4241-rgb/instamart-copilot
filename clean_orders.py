
import pandas as pd
from pathlib import Path

# Load raw data
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# 1. Convert timestamps
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Remove exact duplicates
orders = orders.drop_duplicates()

# 3. Keep the latest record for each order
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# 4. Convert order amounts to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Add product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Fill missing order amounts using the formula
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# 5. Clean delivery times
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

# Calculate category medians from completed orders
completed = orders["status"] == "completed"

category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders have no delivery time
orders.loc[
    orders["status"] == "cancelled",
    "delivery_minutes"
] = float("nan")

# Remove temporary product columns
orders = orders.drop(columns=["price_inr", "category"])

# Save cleaned data
Path("data/clean").mkdir(parents=True, exist_ok=True)

orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

orders.to_csv("data/clean/orders.csv", index=False)

# Verify results
print("Total cleaned orders:", len(orders))

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:",
      orders["order_amount"].isna().sum())

print("\nMissing delivery times among completed:",
      orders.loc[completed, "delivery_minutes"].isna().sum())

print("\nCompleted-order revenue:",
      round(orders.loc[completed, "order_amount"].sum(), 2))
