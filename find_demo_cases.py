# find_demo_cases.py
# Finds LOW / MEDIUM / HIGH demo transactions from your real dataset
# using your trained LightGBM model.

import pandas as pd
import pickle

# -----------------------------
# Load Model
# -----------------------------
with open("models/lightgbm_1m.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("research/graph_augmented_sample.csv")

# -----------------------------
# Exact Features Used in Retraining
# -----------------------------
FEATURES = [
    "BlockHeight",
    "TimeStamp",
    "Value",
    "from_in_degree",
    "from_out_degree",
    "from_pagerank",
    "to_in_degree",
    "to_out_degree",
    "to_pagerank"
]

# -----------------------------
# Predict Scores
# -----------------------------
X = df[FEATURES]

df["probability"] = model.predict_proba(X)[:, 1]

# -----------------------------
# Buckets
# -----------------------------
low_df = df[df["probability"] < 0.20]
med_df = df[(df["probability"] >= 0.45) & (df["probability"] < 0.75)]
high_df = df[df["probability"] >= 0.90]

# -----------------------------
# Select one row each
# -----------------------------
low_case = low_df.sample(1, random_state=42)
med_case = med_df.sample(1, random_state=42)
high_case = high_df.sample(1, random_state=42)

# -----------------------------
# Columns to Show
# -----------------------------
cols = [
    "BlockHeight",
    "TimeStamp",
    "From",
    "To",
    "Value",
    "isError",
    "probability"
]

print("\n==============================")
print("LOW DEMO CASE")
print("==============================")
print(low_case[cols].to_string(index=False))

print("\n==============================")
print("MEDIUM DEMO CASE")
print("==============================")
print(med_case[cols].to_string(index=False))

print("\n==============================")
print("HIGH DEMO CASE")
print("==============================")
print(high_case[cols].to_string(index=False))