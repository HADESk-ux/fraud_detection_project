import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os

GRAPH_PATH = "research/transaction_graph.pkl"
FEATURE_SAVE_PATH = "research/graph_metrics.pkl"

print("Loading graph...")

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

print("Computing graph metrics...")

# Basic degrees
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

print("Computing PageRank...")
pagerank = nx.pagerank(G, alpha=0.85)

print("Packaging metrics...")

graph_metrics = {
    "in_degree": in_degree,
    "out_degree": out_degree,
    "pagerank": pagerank
}

with open(FEATURE_SAVE_PATH, "wb") as f:
    pickle.dump(graph_metrics, f)

print("Graph metrics saved successfully.")