from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ===========================
# Load trained model, scaler & encoders
# ===========================
base_path = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(base_path, "botnet_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
feature_order = joblib.load(os.path.join(base_path, "feature_order.pkl"))

# Encoders for categorical features
proto_encoder = joblib.load(os.path.join(base_path, "proto_encoder.pkl"))
service_encoder = joblib.load(os.path.join(base_path, "service_encoder.pkl"))
conn_state_encoder = joblib.load(os.path.join(base_path, "conn_state_encoder.pkl"))

# ===========================
# Routes
# ===========================
@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if all fields are filled
        if not all(request.form.values()):
            return render_template("index.html", prediction="⚠️ Please fill in all fields before predicting!")

        # Collect manual inputs
        duration = float(request.form['duration'])
        src_bytes = float(request.form['src_bytes'])
        dst_bytes = float(request.form['dst_bytes'])
        src_pkts = float(request.form['src_pkts'])
        dst_pkts = float(request.form['dst_pkts'])
        proto = request.form['proto'].lower()
        service = request.form['service']
        conn_state = request.form['conn_state']
        http_req_len = float(request.form['http_request_body_len'])
        http_resp_len = float(request.form['http_response_body_len'])
        http_status = float(request.form['http_status_code'])

        # =====================
        # Encode categorical values
        # =====================
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

        # =====================
        # Default DNS values
        # =====================
        dns_query = 0
        dns_qclass = 0
        dns_qtype = 0
        dns_rcode = 0

        # =====================
        # Construct feature vector in correct order
        # =====================
        input_dict = {
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "src_pkts": src_pkts,
            "dst_pkts": dst_pkts,
            "proto": proto,
            "service": service,
            "conn_state": conn_state,
            "dns_query": dns_query,
            "dns_qclass": dns_qclass,
            "dns_qtype": dns_qtype,
            "dns_rcode": dns_rcode,
            "http_request_body_len": http_req_len,
            "http_response_body_len": http_resp_len,
            "http_status_code": http_status
        }

        input_data = [input_dict[feat] for feat in feature_order]

        # Scale + Predict
        features = scaler.transform([input_data])
        prediction = model.predict(features)[0]

        result = "BOTNET ATTACK 🚨" if prediction == 1 else "normal ✅"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)


# ===========================
# Start Flask app
# ===========================
if __name__ == "__main__":
    app.run(debug=True)
