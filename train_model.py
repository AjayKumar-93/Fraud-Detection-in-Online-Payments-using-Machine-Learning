import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pickle
import os

def generate_sample_data(n_samples=10000):
    np.random.seed(42)
    data = []
    
    transaction_types = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
    
    for _ in range(n_samples):
        # Determine if fraud
        is_fraud = np.random.random() < 0.15  # 15% fraud rate
        
        # Transaction type
        if is_fraud:
            trans_type = np.random.choice([0, 3], p=[0.7, 0.3])  # Mostly CASH_OUT and TRANSFER
        else:
            trans_type = np.random.randint(0, 5)
        
        # Amount
        if is_fraud:
            amount = np.random.uniform(100000, 10000000)
        else:
            amount = np.random.uniform(100, 100000)
        
        # Old balance sender
        old_balance_sender = np.random.uniform(amount, amount * 3) if not is_fraud else np.random.uniform(amount * 0.8, amount * 1.2)
        
        # New balance sender
        if is_fraud:
            # Fraudulent: often zero balance or inconsistent
            new_balance_sender = 0 if np.random.random() < 0.6 else old_balance_sender - amount + np.random.uniform(-amount*0.1, amount*0.1)
        else:
            new_balance_sender = old_balance_sender - amount
        
        # Old balance recipient
        old_balance_recipient = np.random.uniform(0, 1000000)
        
        # New balance recipient
        if is_fraud:
            # Fraudulent: inconsistent balance changes
            new_balance_recipient = old_balance_recipient + amount + np.random.uniform(-amount*0.2, amount*0.2)
        else:
            new_balance_recipient = old_balance_recipient + amount
        
        # Balance differences
        balance_diff_sender = old_balance_sender - new_balance_sender
        balance_diff_recipient = new_balance_recipient - old_balance_recipient
        
        data.append([
            trans_type, amount, old_balance_sender, new_balance_sender,
            old_balance_recipient, new_balance_recipient,
            balance_diff_sender, balance_diff_recipient, is_fraud
        ])
    
    columns = [
        'transaction_type', 'amount', 'old_balance_sender', 'new_balance_sender',
        'old_balance_recipient', 'new_balance_recipient',
        'balance_diff_sender', 'balance_diff_recipient', 'is_fraud'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df

def train_fraud_model():
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*60)
    print("\n📊 Generating training data...")
    
    df = generate_sample_data(10000)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_data.csv', index=False)
    
    print(f"✅ Training data saved: {len(df)} samples")
    print(f"📈 Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"📉 Legitimate cases: {len(df) - df['is_fraud'].sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    
    # Prepare features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n🤖 Training XGBoost model...")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()  # Handle imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n🔍 Feature Importance:")
    feature_names = [
        'transaction_type', 'amount', 'old_balance_sender', 'new_balance_sender',
        'old_balance_recipient', 'new_balance_recipient',
        'balance_diff_sender', 'balance_diff_recipient'
    ]
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fraud_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print("="*60)
    
    return accuracy

if __name__ == '__main__':
    train_fraud_model()
