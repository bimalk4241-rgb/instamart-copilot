
import pandas as pd

# Load cleaned datasets
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# Use only completed orders for revenue analysis
completed = orders[orders["status"] == "completed"].copy()

# Join customers and products
df = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
).merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# 1. Overall business performance
print("\nOVERALL BUSINESS PERFORMANCE")
print("Completed orders:", len(completed))
print("Total revenue:", round(completed["order_amount"].sum(), 2))
print("Average order value:", round(completed["order_amount"].mean(), 2))

# 2. Revenue by city
print("\nREVENUE BY CITY")
city_revenue = (
    df.groupby("city")["order_amount"]
    .agg(["count", "sum", "mean"])
    .round(2)
    .sort_values("sum", ascending=False)
)
print(city_revenue)

# 3. Revenue by product category
print("\nREVENUE BY CATEGORY")
category_revenue = (
    df.groupby("category")["order_amount"]
    .sum()
    .sort_values(ascending=False)
    .round(2)
)
print(category_revenue)

# 4. Monthly revenue
df["order_ts"] = pd.to_datetime(df["order_ts"])
df["month"] = df["order_ts"].dt.to_period("M")

print("\nMONTHLY REVENUE")
monthly_revenue = (
    df.groupby("month")["order_amount"]
    .sum()
    .round(2)
)
print(monthly_revenue)

# 5. Delivery performance
print("\nDELIVERY PERFORMANCE")
print("Median delivery time:", completed["delivery_minutes"].median())
print("Average delivery time:", round(completed["delivery_minutes"].mean(), 2))
print(
    "Orders delivered after 45 minutes:",
    (completed["delivery_minutes"] > 45).sum()
)

# 6. Save results
city_revenue.to_csv("data/clean/eda_city_revenue.csv")
category_revenue.to_csv("data/clean/eda_category_revenue.csv")
monthly_revenue.to_csv("data/clean/eda_monthly_revenue.csv")

print("\nEDA COMPLETED SUCCESSFULLY")