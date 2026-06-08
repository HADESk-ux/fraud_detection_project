What this project does
This project implements a blockchain fraud detection pipeline that combines transaction data, graph analytics, and a machine learning model.

1. Data & Graph Feature Engineering
Uses raw Ethereum transaction data from second_order_df.csv and first_order_df.csv.
build_graph.py
samples transactions
builds a directed graph where nodes are addresses and edges are transactions
saves the graph to transaction_graph.pkl
graph_features.py
loads the graph
computes network metrics:
sender/receiver in-degree
sender/receiver out-degree
PageRank
saves those metrics for later use
2. Model Training / Research
The repository contains multiple model artifacts:
lightgbm_1m.pkl
xgboost_1m.joblib
random_forest_1m.joblib
model_meta.json documents the chosen model as LightGBM with a reported ROC-AUC of 0.972.
feature_manifest.json lists a different baseline feature set, showing the project also experiments with frequency and time-based features.
3. Inference & Deployment
app_fastapi.py
loads the deployed LightGBM model
loads the transaction graph and PageRank values
defines a prediction endpoint at POST /predict
for each incoming transaction:
reads raw fields: block height, timestamp, from/to addresses, value
computes graph features for sender and receiver
builds a feature vector in fixed order
predicts probability of fraud
converts probability into:
label (0 or 1)
risk category (LOW, MEDIUM, HIGH)
applies business overrides:
large value (>= 1000) => HIGH
suspicious address patterns => HIGH
4. Demo & Visualization
dashboard_streamlit.py
provides a Streamlit UI for:
showing model comparison results
inputting transaction details
running fraud prediction via the API
tuning risk thresholds
viewing recent transaction history
uses example transactions for normal / moderate / suspicious cases
5. Supporting Scripts
find_demo_cases.py
loads the trained LightGBM model and a graph-augmented dataset
computes fraud probability for all rows
selects sample low, medium, and high probability demo cases
Summary of the actual workflow
Build a transaction graph from raw data
Compute network features like degrees and PageRank
Train or use a hybrid LightGBM model that includes these graph features
Deploy the model behind a FastAPI service
Provide a Streamlit dashboard for interactive prediction and risk assessment
