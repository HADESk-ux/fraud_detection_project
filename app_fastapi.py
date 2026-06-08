# app_fastapi.py

import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import Optional

# -----------------------------
# Initialize API
# -----------------------------
app = FastAPI(title="AI-Powered Blockchain Fraud Detection API")

# -----------------------------
# Load Retrained LightGBM Model
# -----------------------------
try:
    with open("models/lightgbm_1m.pkl", "rb") as f:
        model = pickle.load(f)

    print("✔ LightGBM model loaded.")

except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# -----------------------------
# Correct Feature Order
# Matches retrained model exactly
# -----------------------------
FEATURE_ORDER = [
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

print("✔ Feature order loaded.")

# -----------------------------
# Load Transaction Graph
# -----------------------------
try:
    with open("research/transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)

    pagerank = nx.pagerank(G)

    print("✔ Transaction graph loaded.")

except Exception as e:
    print("⚠ Warning loading graph:", e)
    G = None
    pagerank = {}

# -----------------------------
# Request Schema
# -----------------------------
class TxRaw(BaseModel):
    BlockHeight: int
    TimeStamp: int
    From: str
    To: Optional[str] = None
    Value: float
    model_type: Optional[str] = "Hybrid Model"

# -----------------------------
# Graph Feature Builder
# -----------------------------
def get_graph_features(from_addr, to_addr):

    if G is None:
        return {
            "from_in_degree": 0,
            "from_out_degree": 0,
            "from_pagerank": 0,
            "to_in_degree": 0,
            "to_out_degree": 0,
            "to_pagerank": 0
        }

    return {
        "from_in_degree": G.in_degree(from_addr) if from_addr in G else 0,
        "from_out_degree": G.out_degree(from_addr) if from_addr in G else 0,
        "from_pagerank": pagerank.get(from_addr, 0),

        "to_in_degree": G.in_degree(to_addr) if to_addr in G else 0,
        "to_out_degree": G.out_degree(to_addr) if to_addr in G else 0,
        "to_pagerank": pagerank.get(to_addr, 0)
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(tx: TxRaw):

    try:
        # Raw Inputs
        bh = int(tx.BlockHeight)
        ts = int(tx.TimeStamp)
        from_addr = tx.From
        to_addr = tx.To
        val = float(tx.Value)

        # Graph Features
        gf = get_graph_features(from_addr, to_addr)

        # Final Input Features
        features_dict = {
            "BlockHeight": bh,
            "TimeStamp": ts,
            "Value": val,

            "from_in_degree": gf["from_in_degree"],
            "from_out_degree": gf["from_out_degree"],
            "from_pagerank": gf["from_pagerank"],

            "to_in_degree": gf["to_in_degree"],
            "to_out_degree": gf["to_out_degree"],
            "to_pagerank": gf["to_pagerank"]
        }

        # Ordered Vector
        x = np.array(
            [features_dict[f] for f in FEATURE_ORDER]
        ).reshape(1, -1)

        # Predict Probability
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[0][1])
        else:
            pred = model.predict(x)
            prob = float(np.asarray(pred).ravel()[0])

        label = int(prob >= 0.5)

        # Risk Classification
        if prob >= 0.98:
            risk = "HIGH"
        elif prob >= 0.85:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Optional Business Overrides
        if val >= 1000:
            risk = "HIGH"

        if from_addr.startswith("0x000") or str(to_addr).startswith("0xffff"):
            risk = "HIGH"

        print("DEBUG probability:", prob)
        print("DEBUG risk:", risk)

        return {
            "probability": prob,
            "label": label,
            "risk": risk,
            "model_used": "LightGBM Hybrid Retrained"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running!",
        "model": "LightGBM Hybrid Retrained"
    }