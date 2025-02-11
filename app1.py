from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load model and scaler
def load_model(filename="best_model.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Define features
        numerical_features = ['Age', 'Billing Amount']
        categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital',
                                'Insurance Provider', 'Admission Type', 'Medication', 'Room Number']
        
        # Convert numerical features to float
        for feature in numerical_features:
            df[feature] = df[feature].astype(float)
        
        # Scale numerical features
        X_scaled = scaler.transform(df[numerical_features])
        
        # Encode categorical features
        X_categorical = []
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))
            X_categorical.append(df[feature].values)
        
        X_categorical = np.array(X_categorical).T
        
        # Combine numerical and categorical features
        X_final = np.concatenate([X_scaled, X_categorical], axis=1).astype(np.float32)
        
        # Predict
        prediction = model.predict(X_final)
        
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
