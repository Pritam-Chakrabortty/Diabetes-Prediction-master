import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load model and scaler from the same folder
folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, 'model.pkl')
scaler_path = os.path.join(folder, 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        glucose = float(request.form['Glucose'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        age = float(request.form['Age'])

        # Prepare features and scale
        features = np.array([[glucose, insulin, bmi, age]])
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]
        if prediction == 1:
            result = "You have Diabetes. Please consult a doctor."
        else:
            result = "You don't have Diabetes."

        return render_template(
            'index.html',
            prediction_text=result,
            glucose=glucose,
            insulin=insulin,
            bmi=bmi,
            age=age
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
