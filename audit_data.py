
import pandas as pd

# Load the three raw datasets
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Display basic information about each dataset
for name, df in [
    ("CUSTOMERS", customers),
    ("PRODUCTS", products),
    ("ORDERS", orders),
]:
    print(f"\n{'=' * 50}")
    print(f"{name} DATA AUDIT")
    print(f"{'=' * 50}")

    print("\nShape (rows, columns):", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    audit = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2)
    })
    print(audit)

# Specific order checks
print("\nORDER QUALITY CHECKS")
print("Raw rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

print("Non-numeric amounts:", amount.isna().sum())
print("Negative delivery times:", (delivery < 0).sum())
print("Blank delivery times:", delivery.isna().sum())

# Specific customer checks
print("\nCUSTOMER QUALITY CHECKS")

ages = pd.to_numeric(customers["age"], errors="coerce")
print("Blank ages:", ages.isna().sum())
print("Ages outside 10–100:", ((ages < 10) | (ages > 100)).sum())
print("Raw city spellings:", customers["city"].nunique())
print("Raw gender spellings:", customers["gender"].dropna().nunique())
print("Blank genders:", customers["gender"].isna().sum())
