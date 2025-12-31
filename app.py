from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/fraud_model.pkl"

# Load ML Model
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_fraud():
    try:
        data = request.json

        # ✅ Step 1: Validate Input Consistency
        validation = validate_transaction(data)
        if validation != "VALID":
            return fraud_response(
                fraud=1,
                probability=99.99,
                risk=100,
                reason=validation
            )

        # ✅ Step 2: Extract ML Features
        features = extract_features(data)

        # ✅ Step 3: ML Prediction
        if model:
            prob = float(model.predict_proba([features])[0][1])
            pred = int(model.predict([features])[0])
        else:
            prob = calculate_risk_score(data) / 100
            pred = 1 if prob >= 0.5 else 0

        # ✅ Step 4: Rule-based Score
        rule_risk = calculate_risk_score(data)

        # ✅ Step 5: Final Decision (ML + Rules)
        final_label = 1 if (pred == 1 or rule_risk >= 50) else 0

        # ✅ Send Final Response
        return fraud_response(
            fraud=final_label,
            probability=prob * 100,
            risk=rule_risk,
            reason="Fraud detected" if final_label else "Safe Transaction"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ✅ FINAL RESPONSE FORMAT
def fraud_response(fraud, probability, risk, reason):
    return jsonify({
        "fraud_prediction": fraud,
        "fraud_probability": round(probability, 3),
        "risk_score": risk,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    })


# ✅ FEATURE ENGINE
def extract_features(data):
    type_map = {
        "CASH_OUT": 0,
        "PAYMENT": 1,
        "CASH_IN": 2,
        "TRANSFER": 3,
        "DEBIT": 4
    }

    t = type_map.get(data.get("transaction_type"), 0)

    amt = float(data.get("amount", 0))
    old_s = float(data.get("old_balance_sender", 0))
    new_s = float(data.get("new_balance_sender", 0))
    old_r = float(data.get("old_balance_recipient", 0))
    new_r = float(data.get("new_balance_recipient", 0))

    features = [
        t, amt, old_s, new_s, old_r, new_r,
        old_s - new_s,      # sender difference
        new_r - old_r       # recipient difference
    ]

    return features


# ✅ FIXED — STRONG VALIDATION ENGINE
def validate_transaction(data):

    try:
        amt = float(data.get("amount", 0))
        old_s = float(data.get("old_balance_sender", 0))
        new_s = float(data.get("new_balance_sender", 0))
        old_r = float(data.get("old_balance_recipient", 0))
        new_r = float(data.get("new_balance_recipient", 0))
        t = data.get("transaction_type")
    except:
        return "Invalid numeric values."

    if amt <= 0:
        return "Amount cannot be zero or negative."

    # ✅ Sender must have required balance
    if t in ["TRANSFER", "CASH_OUT", "DEBIT"]:
        if old_s < amt:
            return "Sender does not have enough balance."

    # ✅ Sender balance must match transaction
    expected_new_sender = old_s - amt
    if abs(new_s - expected_new_sender) > 1:
        return "Sender balance mismatch."

    # ✅ Recipient balance must match amount received
    if t in ["CASH_IN", "TRANSFER", "PAYMENT"]:
        expected_new_recipient = old_r + amt
        if abs(new_r - expected_new_recipient) > 1:
            return "Recipient balance mismatch."

    # ✅ Additional Fraud Patterns
    if new_s == 0 and old_s > amt:
        return "Sender suspiciously went to zero."

    if old_r == 0 and new_r > amt and t != "CASH_IN":
        return "Recipient balance indirectly changed."

    return "VALID"


# ✅ RULE-BASED RISK ENGINE
def calculate_risk_score(data):
    amt = float(data.get("amount", 0))
    old_s = float(data.get("old_balance_sender", 0))
    new_s = float(data.get("new_balance_sender", 0))
    old_r = float(data.get("old_balance_recipient", 0))
    new_r = float(data.get("new_balance_recipient", 0))
    t = data.get("transaction_type")

    risk = 0

    if amt > 200000:
        risk += 40
    elif amt > 100000:
        risk += 25

    if abs((old_s - amt) - new_s) > 50:
        risk += 30

    if abs((old_r + amt) - new_r) > 50:
        risk += 30   # ✅ FIXED — now checks recipient properly

    if new_s == 0 and old_s > amt:
        risk += 15

    if t in ["TRANSFER", "CASH_OUT"]:
        risk += 10

    return min(risk, 100)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("\n✅ Fraud Detection System Running…")
    app.run(debug=True, host="0.0.0.0", port=5000)
