from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ==============================
# Load Trained Model Components
# ==============================
model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==============================
# Home Route
# ==============================
@app.route('/')
def home():
    return render_template("index.html")


# ==============================
# Prediction API (For Simulation + Real-time)
# ==============================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get JSON data from frontend
        data = request.get_json()

        hr = float(data['hr'])
        spo2 = float(data['spo2'])
        motion = float(data['motion'])
        sleep_time = float(data['sleep_time'])

        # IMPORTANT: Must match training feature order
        input_df = pd.DataFrame(
            [[hr, spo2, motion, sleep_time]],
            columns=["hr", "spo2", "motion", "sleep_time"]
        )

        # Scale input (since model was trained with scaler)
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        # Convert numeric label to actual disorder name
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        confidence = round(max(probability[0]) * 100, 2)

        return jsonify({
            "result": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# Run Application
# ==============================
if __name__ == "__main__":
    app.run(debug=True)