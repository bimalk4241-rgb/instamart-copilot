
import pandas as pd
import numpy as np
from pathlib import Path

# Load cleaned datasets from Part A
orders = pd.read_csv(
    "data/clean/orders.csv",
    parse_dates=["order_ts"]
)

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# Define the snapshot date
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# Select only completed orders
completed = orders[
    orders["status"] == "completed"
].copy()

# Build the RFM table
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (
    snapshot - rfm["last_order"]
).dt.days

# Keep the required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Print reference statistics
print("\nRFM ANALYSIS")
print("=" * 50)

print("Customers:", len(rfm))

print("\nRecency:")
print("Median:", rfm["recency"].median())
print("Minimum:", rfm["recency"].min())
print("Maximum:", rfm["recency"].max())

print("\nFrequency:")
print("Mean:", round(rfm["frequency"].mean(), 2))
print("Maximum:", rfm["frequency"].max())

print("\nMonetary:")
print("Mean:", round(rfm["monetary"].mean(), 2))
print("Maximum:", round(rfm["monetary"].max(), 2))

# Save intermediate RFM results
Path("outputs").mkdir(parents=True, exist_ok=True)
rfm.to_csv("outputs/rfm.csv", index=False)

print("\nRFM TABLE CREATED SUCCESSFULLY")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# TASK 2: K-MEANS CLUSTERING
# Standardize RFM features
features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

print("\nK-MEANS CLUSTERING RESULTS")
print("=" * 50)

results = []
models = {}

# Test K values from 2 to 8
for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": round(inertia, 1),
        "Silhouette": round(silhouette, 3)
    })

    models[k] = model

print(pd.DataFrame(results).to_string(index=False))

# Choose K = 4 for interpretable customer segments
final_k = 4
rfm["segment_id"] = models[final_k].labels_

# TASK 3: CUSTOMER SEGMENT PROFILES
profiles = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nCUSTOMER SEGMENT PROFILES")
print(profiles)

# Premium is always the highest-spending cluster
premium_id = profiles["mean_monetary"].idxmax()

rfm["is_premium"] = rfm["segment_id"] == premium_id

# Assign descriptive names based on monetary value
sorted_segments = profiles.sort_values(
    "mean_monetary", ascending=False
).index.tolist()

segment_names = [
    "Premium",
    "High Value",
    "Regular",
    "Occasional"
]

name_mapping = dict(zip(sorted_segments, segment_names))

rfm["segment_name"] = rfm["segment_id"].map(name_mapping)

# PCA analysis
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA EXPLAINED VARIANCE RATIOS")
print(pca.explained_variance_ratio_)
print("Total explained variance:",
      round(pca.explained_variance_ratio_.sum(), 3))

# Save the customer segment file
rfm.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nCUSTOMER SEGMENTATION COMPLETED")


# TASK 4: LAPSE PREDICTION DATASET

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

cutoff = pd.Timestamp("2025-09-01")

# Training features use ONLY orders before September
historical = orders[orders["order_ts"] < cutoff].copy()

historical_completed = historical[
    historical["status"] == "completed"
].copy()

# Customers who completed at least one order before cutoff
cohort_ids = historical_completed["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

# Label: 1 if customer has no completed orders during September
september_completed = completed[
    (completed["order_ts"] >= cutoff) &
    (completed["order_ts"] < snapshot)
]

active_ids = set(september_completed["customer_id"])

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_ids)
).astype(int)

# Historical RFM features
history_features = historical_completed.groupby(
    "customer_id"
).agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

history_features["recency"] = (
    cutoff - history_features["last_order"]
).dt.days

# Cancellation rate: cancelled / all final-status orders
final_orders = historical[
    historical["status"].isin(["completed", "cancelled"])
].copy()

final_orders["is_cancelled"] = (
    final_orders["status"] == "cancelled"
).astype(int)

cancellation = final_orders.groupby(
    "customer_id"
)["is_cancelled"].mean().reset_index(
    name="cancellation_rate"
)

# Most frequently purchased product category
historical_products = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = historical_products.groupby(
    ["customer_id", "category"]
).size().reset_index(name="count")

favourite_category = (
    category_counts
    .sort_values(
        ["customer_id", "count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Customer tenure at cutoff
cohort["signup_date"] = pd.to_datetime(
    cohort["signup_date"]
)

cohort["tenure_days"] = (
    cutoff - cohort["signup_date"]
).dt.days

# Combine features
model_data = (
    cohort
    .merge(history_features, on="customer_id")
    .merge(cancellation, on="customer_id", how="left")
    .merge(favourite_category, on="customer_id", how="left")
)

# Select only pre-cutoff features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

X = model_data[numeric_features + categorical_features]
y = model_data["lapsed"]

print("\nLAPSE DATASET")
print("=" * 50)
print("Cohort customers:", len(model_data))
print("Lapsed customers:", y.sum())
print("Lapse percentage:", round(y.mean() * 100, 2))

# TASK 5: TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Preprocessing is fitted ONLY on training data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nFEATURE PREPROCESSING READY")


# TASK 6: TRAIN AND EVALUATE MODELS

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import joblib

# Baseline: always predict "not lapsed"
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\nBASELINE MODEL")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
print("Lapsed F1:", f1_score(y_test, baseline_predictions, zero_division=0))

# Logistic Regression
lr_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTRAINING LOGISTIC REGRESSION")
lr_grid.fit(X_train, y_train)

print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 3))

# Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTRAINING RANDOM FOREST")
rf_grid.fit(X_train, y_train)

print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 3))

# Evaluate on the untouched test set
def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_eval, predictions), 3))
    print("Precision:", round(precision_score(y_eval, predictions, zero_division=0), 3))
    print("Recall:", round(recall_score(y_eval, predictions, zero_division=0), 3))
    print("F1:", round(f1_score(y_eval, predictions, zero_division=0), 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_eval, predictions))

    return f1_score(y_eval, predictions, zero_division=0)

lr_f1 = evaluate_model(
    "LOGISTIC REGRESSION",
    lr_grid.best_estimator_,
    X_test, y_test
)

rf_f1 = evaluate_model(
    "RANDOM FOREST",
    rf_grid.best_estimator_,
    X_test, y_test
)

# Select model using cross-validation, not test performance
if rf_grid.best_score_ >= lr_grid.best_score_:
    selected_model = rf_grid.best_estimator_
    selected_name = "Random Forest"
else:
    selected_model = lr_grid.best_estimator_
    selected_name = "Logistic Regression"

print("\nSELECTED MODEL:", selected_name)

# TASK 7: CLASS BALANCING EXPERIMENT

# Oversample only the training data
train_data = X_train.copy()
train_data["lapsed"] = y_train

majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

minority_oversampled = minority.sample(
    n=len(majority),
    replace=True,
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

balanced_lr = clone(lr_grid.best_estimator_)
balanced_lr.fit(X_train_balanced, y_train_balanced)

print("\nCLASS BALANCING EXPERIMENT")
print("Balanced training class counts:")
print(y_train_balanced.value_counts())

evaluate_model(
    "BALANCED LOGISTIC REGRESSION",
    balanced_lr,
    X_test, y_test
)

# TASK 8: INTERPRET AND SAVE

print("\nTOP FIVE PERMUTATION IMPORTANCES")
importance = permutation_importance(
    selected_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=5,
    random_state=42
)

importances = pd.Series(
    importance.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print(importances.head(5))

print("\nLOGISTIC REGRESSION COEFFICIENTS")
lr_model = lr_grid.best_estimator_
feature_names = lr_model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = pd.Series(
    lr_model.named_steps["model"].coef_[0],
    index=feature_names
)

print(coefficients.sort_values().head(5))
print(coefficients.sort_values().tail(5))

# Save model
joblib.dump(selected_model, "outputs/lapse_model.joblib")

# Reload and verify identical predictions
loaded_model = joblib.load("outputs/lapse_model.joblib")

assert np.array_equal(
    selected_model.predict(X_test),
    loaded_model.predict(X_test)
)

print("\nMODEL SAVED AND RELOAD VERIFIED")

# Retention list: Premium customers in the test set
# Premium flag comes from full-window RFM and is illustrative
test_customers = model_data.loc[X_test.index, [
    "customer_id", "lapsed"
]].copy()

premium_lookup = rfm.set_index("customer_id")["is_premium"]

test_customers["is_premium"] = (
    test_customers["customer_id"]
    .map(premium_lookup)
    .fillna(False)
)

test_customers["lapse_probability"] = (
    loaded_model.predict_proba(X_test)[:, 1]
)

retention = (
    test_customers[test_customers["is_premium"]]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    .copy()
)

retention = retention.rename(
    columns={"lapsed": "actually_lapsed"}
)

retention = retention[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nPREMIUM RETENTION LIST")
print(retention.to_string(index=False))

print("\nPART B COMPLETED")
