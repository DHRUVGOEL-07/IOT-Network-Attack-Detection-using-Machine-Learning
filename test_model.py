import joblib
import os
import numpy as np

# ============================
# Load trained model & objects
# ============================
base_path = os.path.join(os.path.dirname(__file__), "models")

model = joblib.load(os.path.join(base_path, "botnet_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
feature_order = joblib.load(os.path.join(base_path, "feature_order.pkl"))

proto_encoder = joblib.load(os.path.join(base_path, "proto_encoder.pkl"))
service_encoder = joblib.load(os.path.join(base_path, "service_encoder.pkl"))
conn_state_encoder = joblib.load(os.path.join(base_path, "conn_state_encoder.pkl"))

# ============================
# Example benign traffic values
# ============================
duration = 2.0
src_bytes = 350.0
dst_bytes = 500.0
src_pkts = 10.0
dst_pkts = 12.0
proto = "tcp"        # lowercase to match training data
service = "-"        # if no service
conn_state = "SF"    # e.g., normal successful connection
missed_bytes = 0.0
http_req_len = 200.0
http_resp_len = 1000.0
http_status = 200.0

# ============================
# Encode categorical values
# ============================
try:
    proto = proto_encoder.transform([proto])[0]
except:
    proto = 0
try:
    service = service_encoder.transform([service])[0]
except:
    service = 0
try:
    conn_state = conn_state_encoder.transform([conn_state])[0]
except:
    conn_state = 0

# ============================
# Construct input vector
# ============================
input_dict = {
    "duration": duration,
    "src_bytes": src_bytes,
    "dst_bytes": dst_bytes,
    "src_pkts": src_pkts,
    "dst_pkts": dst_pkts,
    "proto": proto,
    "service": service,
    "conn_state": conn_state,
    "missed_bytes": missed_bytes,
    "http_request_body_len": http_req_len,
    "http_response_body_len": http_resp_len,
    "http_status_code": http_status
}

input_data = [input_dict[feat] for feat in feature_order]

# Debug print
print("Input vector (before scaling):", input_data)

# Scale + Predict
features = scaler.transform([input_data])
print("Input vector (after scaling):", features)

prediction = model.predict(features)[0]
print("\nPrediction:", "BOTNET ATTACK 🚨" if prediction == 1 else "BENIGN ✅")
