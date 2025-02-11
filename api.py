from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and scaler
def load_model():
    with open("/Users/sahinbegum/Downloads/docker/best_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    df = pd.read_csv(filepath, encoding='latin-1')
    numerical_features = ['Age', 'Billing Amount']
    categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital', 
                            'Insurance Provider', 'Admission Type', 'Medication', 'Room Number']
    
    X = df[numerical_features + categorical_features].values
    X_scaled = scaler.transform(X[:, :len(numerical_features)])
    X_final = np.concatenate([X_scaled, X[:, len(numerical_features):]], axis=1)
    
    for i in range(len(numerical_features), X_final.shape[1]):
        le = LabelEncoder()
        X_final[:, i] = le.fit_transform(X_final[:, i].astype(str))
    
    X_final = X_final.astype(np.float32)
    predictions = model.predict(X_final)
    df['Predicted Backorder'] = predictions
    
    return jsonify(df[['Predicted Backorder']].to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
