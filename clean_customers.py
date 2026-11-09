
import pandas as pd
from pathlib import Path

# Load the original customer data
customers = pd.read_csv("data/raw/customers.csv")

print("Original customer count:", len(customers))

# 1. Standardize city names
customers["city"] = customers["city"].str.strip().str.title()

# 2. Standardize gender values
gender_map = {
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female"
}

customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("Unknown")
)

# 3. Convert ages to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Treat ages outside 10–100 as invalid
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing and invalid ages with the median
median_age = customers["age"].median()
customers["age"] = customers["age"].fillna(median_age).astype(int)

# 4. Convert signup dates to dates
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# 5. Save the cleaned dataset
Path("data/clean").mkdir(parents=True, exist_ok=True)

customers.to_csv("data/clean/customers.csv", index=False)

# 6. Validate the results
print("Cleaned customer count:", len(customers))
print("Median valid age:", median_age)
print("Unique cities:", sorted(customers["city"].unique()))
print("Gender counts:")
print(customers["gender"].value_counts())
print("Missing values:")
print(customers.isna().sum())
