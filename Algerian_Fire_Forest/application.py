from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

try:
    # import our custom ridge regressor and standard scaler
    ridge_model = pickle.load(open('models/ridge_cv_model.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except Exception as e:
    app.logger.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load models")

@app.route("/")
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error rendering index.html: {e}")
        return "Error processing the request", 500

@app.route("/predict_datapoint", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extracting features from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Reigon = float(request.form.get('Reigon'))  # Corrected the typo here

        # Assembling the feature array and reshaping for the scaler
        features = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Reigon]
        new_data_scaled = standard_scaler.transform([features])  # Reshape by wrapping in another list

        # Making predictions
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result=result[0])
    else:
        try:
            return render_template('home.html')  # Adjust if your form is in another HTML file
        except Exception as e:
            app.logger.error(f"Error in GET request of /predict_datapoint: {e}")
            return "Error processing the request", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
