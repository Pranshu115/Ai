# import os
# import pickle
# import pandas as pd
# from flask import Flask, request, render_template, redirect, url_for
# from werkzeug.utils import secure_filename
# from sklearn.preprocessing import LabelEncoder

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model and scaler
# # Load the trained model and scaler
# MODEL_FILE = "best_model.pkl"  # Changed to the correct filename
# with open(MODEL_FILE, "rb") as f:
#     loaded_data = pickle.load(f)
# model, scaler = loaded_data["model"], loaded_data["scaler"]

# # Allowed file extensions
# ALLOWED_EXTENSIONS = {"csv"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_data(csv_file, scaler):
#     """Preprocess the uploaded CSV file."""
#     df = pd.read_csv(csv_file)
#     df.fillna(0, inplace=True)

#     # Define numerical and categorical features
#     numerical_features = ["Age", "Billing Amount"]
#     categorical_features = ["Gender", "Blood Type", "Medical Condition", "Doctor", "Hospital", 
#                             "Insurance Provider", "Admission Type", "Medication", "Room Number"]

#     # Ensure all features exist
#     for feature in numerical_features + categorical_features:
#         if feature not in df.columns:
#             raise ValueError(f"Feature '{feature}' not found in the input CSV file.")
            
#     # Label encode categorical features
#     for feature in categorical_features:
#         le = LabelEncoder()
#         df[feature] = le.fit_transform(df[feature].astype(str))
    
#     # Apply scaling to numerical features
#     df[numerical_features] = scaler.transform(df[numerical_features])
    
#     return df, df[numerical_features + categorical_features]

# @app.route("/", methods=["GET", "POST"])
# def upload_file():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)
        
#         file = request.files["file"]
#         if file.filename == "" or not allowed_file(file.filename):
#             return redirect(request.url)
        
#         filename = secure_filename(file.filename)
#         filepath = os.path.join("uploads", filename)
#         file.save(filepath)

#         # Preprocess data
#         df, X = preprocess_data(filepath, scaler)
        
#         # Make predictions
#         predictions = model.predict(X)
#         df["Backorder Prediction"] = predictions
        
#         # Save output file
#         output_file = os.path.join("uploads", "predictions.csv")
#         df.to_csv(output_file, index=False)
        
#         return render_template("result.html", tables=[df.to_html(classes="table table-striped")])

#     return render_template("upload.html")

# if __name__ == "__main__":
#     app.run(debug=True)







# # import argparse
# # import pandas as pd
# # import pickle
# # from sklearn.preprocessing import StandardScaler, LabelEncoder

# # # ... (rest of your functions)

# # def preprocess_data(csv_file, scaler):
# #     df = pd.read_csv(csv_file)
# #     df.fillna(0, inplace=True)  # Handle missing values
    
# #     # Assuming these are your numerical features
# #     numerical_features = ['Age', 'Billing Amount']  
    
# #     # Assuming these are your categorical features
# #     categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital', 
# #                             'Insurance Provider', 'Admission Type', 'Medication', 'Room Number']
    
# #     # Ensure all features are present in the input CSV
# #     for feature in numerical_features + categorical_features:
# #         if feature not in df.columns:
# #             raise ValueError(f"Feature '{feature}' not found in the input CSV file.")
            
# #     # Apply Label Encoding to categorical features (if necessary)
# #     for feature in categorical_features:
# #         le = LabelEncoder()
# #         df[feature] = le.fit_transform(df[feature].astype(str))
    
# #     # Select all relevant features for prediction
# #     X = df[numerical_features + categorical_features]
    
# #     # Apply scaling only to numerical features
# #     X[numerical_features] = scaler.transform(X[numerical_features]) 
    
# #     return X, df

# # def load_model(model_file):
# #     """Loads the trained model and scaler from a pickle file."""
# #     with open(model_file, "rb") as f:
# #         loaded_data = pickle.load(f)
# #     return loaded_data["model"], loaded_data["scaler"]

# # def predict_backorder(model, X):
# #     """Predicts backorder using the loaded model."""
# #     predictions = model.predict(X)
# #     return predictions

# # def save_predictions(df, predictions, output_file):
# #     """Saves the predictions to a CSV file."""
# #     df["Backorder Prediction"] = predictions
# #     df.to_csv(output_file, index=False)

# # # ... (rest of your code)



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
