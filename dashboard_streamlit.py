# dashboard_streamlit.py

import streamlit as st
import requests
import pandas as pd

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI Blockchain Fraud Detection",
    layout="wide"
)

# ----------------- HEADER -----------------
st.title("🔍 AI-Powered Blockchain Fraud Detection Dashboard")
st.caption("Research-backed Ethereum fraud detection using LightGBM + Graph Features")

# ----------------- SESSION STATE -----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "demo_tx" not in st.session_state:
    st.session_state.demo_tx = {}

# ----------------- RESEARCH RESULTS -----------------
st.markdown("---")
st.subheader("📊 Research Results")

try:
    df = pd.read_csv("results/model_comparison.csv")
    st.dataframe(df, use_container_width=True)

    st.image(
        "results/model_comparison.png",
        caption="Performance Comparison: Tabular vs Graph vs Hybrid"
    )

except:
    st.info("Research results files not found.")

# ----------------- DEPLOYED MODEL -----------------
st.markdown("---")
st.subheader("🤖 Production Model")
st.success("LightGBM (Best Performing Hybrid Model)")

# ----------------- RISK THRESHOLD CONTROLS -----------------
st.markdown("---")
st.subheader("⚙️ Risk Threshold Settings")

high_risk_threshold = st.slider(
    "High Risk Threshold (ML Probability)",
    0.50, 0.99, 0.80, 0.01
)

medium_risk_threshold = st.slider(
    "Medium Risk Threshold (ML Probability)",
    0.10, high_risk_threshold, 0.20, 0.01
)

# ----------------- DEMO SCENARIOS -----------------
st.markdown("---")
st.subheader("🧪 Example Transaction Scenarios")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("🟢 Normal Example"):
        st.session_state.demo_tx = {
            "BlockHeight": 7892587,
            "TimeStamp": 1559647956,
            "From": "0xaceda257bd41fbf10e63d7f2edbc0fd6508b046b",
            "To": "0xb130b3b3e699f94f49734dc4435279fd7b4b8f00",
            "Value": 8.0
        }

with c2:
    if st.button("🟠 Moderate Example"):
        st.session_state.demo_tx = {
            "BlockHeight": 8189110,
            "TimeStamp": 1563646301,
            "From": "0x39e6db77941463eea0b323f66509eadf0bf0bf1b",
            "To": "0x0fe07dbd07ba4c1075c1db97806ba3c5b113cee0",
            "Value": 0.000019
        }

with c3:
    if st.button("🔴 Suspicious Example"):
        st.session_state.demo_tx = {
            "BlockHeight": 4958556,
            "TimeStamp": 1516717361,
            "From": "0x9d5c623cfc8bb14bd3a25b76c8d52b507a85160c",
            "To": "0x6267b5376c809445c9432bd9f14a3808b00eae2c",
            "Value": 0.031881
        }

# ----------------- INPUT FORM -----------------
st.markdown("---")
st.subheader("🧾 Enter Ethereum Transaction Details")

blockheight = st.number_input(
    "BlockHeight",
    value=st.session_state.demo_tx.get("BlockHeight", 5800000)
)

timestamp = st.number_input(
    "Timestamp",
    value=st.session_state.demo_tx.get("TimeStamp", 1518200000)
)

from_addr = st.text_input(
    "From Address",
    value=st.session_state.demo_tx.get("From", "0xabc123")
)

to_addr = st.text_input(
    "To Address",
    value=st.session_state.demo_tx.get("To", "0xdef456")
)

value_eth = st.number_input(
    "Transaction Value (ETH)",
    value=float(st.session_state.demo_tx.get("Value", 1.5)),
    format="%.6f"
)

# ----------------- PREDICT -----------------
if st.button("🚀 Predict Fraud"):

    payload = {
        "BlockHeight": int(blockheight),
        "TimeStamp": int(timestamp),
        "From": from_addr,
        "To": to_addr,
        "Value": float(value_eth),
        "model_type": "Hybrid Model"
    }

    try:
        with st.spinner("🔍 Analyzing transaction..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10
            )

        if response.status_code != 200:
            st.error(f"❌ API Error: {response.status_code}")
            st.stop()

        result = response.json()

        prob = float(result["probability"])
        label = int(result.get("label", 0))

        # ----------------- RISK ENGINE -----------------
        risk = "LOW RISK"
        reason = "Low fraud probability."

        if prob >= high_risk_threshold:
            risk = "HIGH RISK"
            reason = "Fraud probability exceeded high threshold."

        elif prob >= medium_risk_threshold:
            risk = "MEDIUM RISK"
            reason = "Fraud probability exceeded medium threshold."

        # ----------------- DISPLAY -----------------
        st.markdown("---")
        st.subheader("🔎 Prediction Result")

        if risk == "HIGH RISK":
            st.error("🚨 HIGH RISK TRANSACTION")
        elif risk == "MEDIUM RISK":
            st.warning("⚠️ MEDIUM RISK TRANSACTION")
        else:
            st.success("✅ LOW RISK TRANSACTION")

        st.metric("Fraud Probability", f"{prob:.3%}")
        st.progress(min(int(prob * 100), 100))

        st.caption(f"🧠 Decision Logic: {reason}")

        st.info(
            "The deployed system uses the best-performing LightGBM model. "
            "Thresholds convert probability into operational risk categories."
        )

        # ----------------- HISTORY -----------------
        st.session_state.history.append({
            "From": from_addr,
            "To": to_addr,
            "Value (ETH)": value_eth,
            "Probability": round(prob, 4),
            "Label": label,
            "Risk": risk
        })

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network/API request failed: {e}")

# ----------------- HISTORY -----------------
st.markdown("---")
st.subheader("📜 Recent Transaction Analysis")

if st.session_state.history:
    st.dataframe(
        st.session_state.history[::-1][:5],
        use_container_width=True
    )
else:
    st.info("No transactions analyzed yet.")