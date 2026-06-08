import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

print("Loading dataset...")

df = pd.read_csv("research/graph_augmented_sample.csv")

target = "isError"

tabular_features = [
    "BlockHeight",
    "TimeStamp",
    "Value"
]

graph_features = [
    "from_in_degree",
    "from_out_degree",
    "from_pagerank",
    "to_in_degree",
    "to_out_degree",
    "to_pagerank"
]

features_hybrid = tabular_features + graph_features

# -------------------------
# Split
# -------------------------
X = df[features_hybrid]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Handle Imbalance
# -------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("scale_pos_weight =", scale_pos_weight)

# -------------------------
# Train Final Hybrid Model
# -------------------------
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    class_weight="balanced",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

print("Training Hybrid Fraud Model...")

model.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

# -------------------------
# Evaluate
# -------------------------
auc = roc_auc_score(y_test, prob)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("\n===== FINAL MODEL RESULTS =====")
print("AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# -------------------------
# Save Model
# -------------------------
os.makedirs("models", exist_ok=True)

with open("models/lightgbm_1m.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved model -> models/lightgbm_1m.pkl")

# -------------------------
# Feature Importance
# -------------------------
importance = pd.DataFrame({
    "feature": features_hybrid,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=True)

importance.plot(
    x="feature",
    y="importance",
    kind="barh",
    figsize=(8,5),
    title="Feature Importance"
)

os.makedirs("results", exist_ok=True)

plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()