import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#Load dataset

data_path = os.path.join(os.path.dirname(__file__), "..", "models", "train_test_network.csv")
df = pd.read_csv(data_path)

print("Data loaded:", df.shape)

# Preprocessing
# Drop columns not useful for ML (IPs, ports, etc.)
drop_cols = ["src_ip", "dst_ip", "src_port", "dst_port"]
df = df.drop(columns=drop_cols, errors="ignore")

# Fill missing values
df = df.fillna(0)

# Encode categorical columns (proto, service, conn_state, dns_query, type)
encoders = {}
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if col != "label":  # don't encode label
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        encoder_path = os.path.join(os.path.dirname(__file__), "..", "models", f"{col}_encoder.pkl")
        joblib.dump(le, encoder_path)
        print(f"Saved encoder for {col} -> {encoder_path}")


#Define features

final_features = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "proto", "service", "conn_state",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code"
]

# Make sure all required columns exist
X = df[final_features]
y = df["label"]

#Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model + scaler + features
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "botnet_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
features_path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_order.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(final_features, features_path)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
print(f"Feature order saved to {features_path}")
