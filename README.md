# 🛡️ Fraud Detection System

Advanced payment fraud detection using XGBoost Machine Learning

## Features

- Real-time transaction fraud detection
- Balance consistency checking
- XGBoost ML model (95%+ accuracy)
- Transaction type analysis
- Sender/Recipient balance tracking

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run application:
```bash
python app.py
```

4. Open browser:
```
http://127.0.0.1:5000
```

## How to Use

1. Select transaction type (CASH_OUT, PAYMENT, etc.)
2. Enter transaction amount
3. Enter sender's old and new balance
4. Enter recipient's old and new balance
5. Click "Check Fraud"

## Test Cases

### Fraudulent Transaction:
- Type: CASH_OUT
- Amount: 500000
- Old Balance (Sender): 550000
- New Balance (Sender): 0
- Old Balance (Recipient): 0
- New Balance (Recipient): 550000

### Legitimate Transaction:
- Type: PAYMENT
- Amount: 5000
- Old Balance (Sender): 10000
- New Balance (Sender): 5000
- Old Balance (Recipient): 2000
- New Balance (Recipient): 7000

## Model Details

- Algorithm: XGBoost
- Features: 8 (transaction type, amount, balances, balance differences)
- Training samples: 10,000
- Accuracy: 95%+

## Technology Stack

- Python 3.8+
- Flask
- XGBoost
- Pandas & NumPy
- HTML/CSS/JavaScript"# Fraud-Detection-in-Online-Payments-using-Machine-Learning" 
