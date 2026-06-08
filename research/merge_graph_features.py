# Merge Graph Features Back Into Dataset

import pandas as pd
import pickle
import os

RAW_PATH = "data/raw/second_order_df.csv"
GRAPH_METRICS_PATH = "research/graph_metrics.pkl"
OUTPUT_PATH = "research/graph_augmented_sample.csv"

SAMPLE_SIZE = 400_000

print("Loading dataset...")
df = pd.read_csv(RAW_PATH)

print("Sampling same subset...")
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()

print("Loading graph metrics...")
with open(GRAPH_METRICS_PATH, "rb") as f:
    graph_metrics = pickle.load(f)

in_degree = graph_metrics["in_degree"]
out_degree = graph_metrics["out_degree"]
pagerank = graph_metrics["pagerank"]

print("Mapping graph features...")

# Sender features
df_sample["from_in_degree"] = df_sample["From"].map(in_degree).fillna(0)
df_sample["from_out_degree"] = df_sample["From"].map(out_degree).fillna(0)
df_sample["from_pagerank"] = df_sample["From"].map(pagerank).fillna(0)

# Receiver features
df_sample["to_in_degree"] = df_sample["To"].map(in_degree).fillna(0)
df_sample["to_out_degree"] = df_sample["To"].map(out_degree).fillna(0)
df_sample["to_pagerank"] = df_sample["To"].map(pagerank).fillna(0)

print("Saving graph-augmented dataset...")
os.makedirs("research", exist_ok=True)
df_sample.to_csv(OUTPUT_PATH, index=False)

print("Graph-augmented dataset saved successfully.")