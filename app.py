from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('flood_model_xgb.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        rainfall = data['Rainfall (mm)']
        temperature = data['Temperature (°C)']
        humidity = data['Humidity (%)']
        
        if rainfall is None or temperature is None or humidity is None:
            return jsonify({"error": "Invalid data"}), 400

        input_data = np.array([[rainfall, temperature, humidity]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        
        result = "Flood Alert" if prediction[0] == 1 else "No Flood"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
