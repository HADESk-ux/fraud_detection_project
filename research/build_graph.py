import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import os

# ===== CONFIG =====
RAW_PATH = "data/raw/second_order_df.csv" 
SAMPLE_SIZE = 400_000
GRAPH_SAVE_PATH = "research/transaction_graph.pkl"

print("Loading dataset...")

df = pd.read_csv(RAW_PATH)

print("Sampling dataset...")
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

print("Building directed graph...")

G = nx.DiGraph()

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    sender = row["From"]
    receiver = row["To"]
    
    if pd.notna(sender) and pd.notna(receiver):
        G.add_edge(sender, receiver)

print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

os.makedirs("research", exist_ok=True)

with open(GRAPH_SAVE_PATH, "wb") as f:
    pickle.dump(G, f)

print("Graph saved successfully.")